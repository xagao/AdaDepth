"""Ground-adherent depth transfer (paper Eq. 15–17).

All geometry targets are detached. No labels or metric depth are used. Unspecified
paper choices are explicit in GACConfig; see docs/GROUND_ADHERENT.md.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GACConfig:
    window_size: int = 9
    contact_radius: int = 2
    min_ground_pixels: int = 3
    normal_y_threshold: float = 0.8
    offset_aggregation: str = 'median'
    reduction: str = 'image_mean'
    smoothness_weight: float = 1.0
    propagation_radius: int = 2
    samples_per_anchor: int = 4
    sampling: str = 'random'
    chunk_size: int = 2048
    eps: float = 1e-7

    def __post_init__(self):
        if self.window_size < 3 or self.window_size % 2 != 1:
            raise ValueError('window_size must be odd and >= 3')
        if self.propagation_radius < 0 or self.samples_per_anchor < 1:
            raise ValueError('Invalid propagation radius or sample count')
        if self.sampling not in {'random', 'all'}:
            raise ValueError('sampling must be random or all')
        if self.contact_radius < 1 or self.min_ground_pixels < 1:
            raise ValueError('contact_radius and min_ground_pixels must be positive')
        if self.min_ground_pixels > self.window_size**2 or self.chunk_size < 1:
            raise ValueError('Invalid ground sample count or chunk_size')
        if not 0 <= self.normal_y_threshold <= 1:
            raise ValueError('normal_y_threshold must be in [0,1]')
        if self.offset_aggregation not in {'median', 'mean'}:
            raise ValueError('offset_aggregation must be median or mean')
        if self.reduction not in {'image_mean', 'valid_mean', 'sum'}:
            raise ValueError('Unknown reduction')
        if self.smoothness_weight < 0 or self.eps <= 0:
            raise ValueError('Invalid loss scale or epsilon')


class GroundAdherentLoss(nn.Module):
    def __init__(self, config=None, contact_provider=None, offset_reducer=None):
        super().__init__()
        self.config = config if isinstance(config, GACConfig) else GACConfig(**(config or {}))
        self.contact_provider = contact_provider
        self.offset_reducer = offset_reducer
        sobel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('sobel_y', sobel.view(1, 1, 3, 3), persistent=False)

    def _check(self, depth, image, ground, uncertain, dynamic):
        if depth.ndim != 4 or depth.shape[1] != 1 or min(depth.shape[2:]) < 2:
            raise ValueError('depth must be Bx1xHxW with H,W >= 2')
        if not depth.is_floating_point() or not (torch.isfinite(depth) & (depth > 0)).all():
            raise ValueError('depth must be finite positive floating point')
        if image.shape != (depth.shape[0], 3, *depth.shape[2:]) or image.device != depth.device:
            raise ValueError('image must be Bx3xHxW on the depth device')
        if not torch.isfinite(image).all():
            raise ValueError('image must be finite')
        masks = []
        for mask in [ground, uncertain, dynamic]:
            if mask.shape != depth.shape or mask.device != depth.device:
                raise ValueError('masks must match Bx1xHxW and depth device')
            if not ((mask == 0) | (mask == 1)).all():
                raise ValueError('masks must be binary')
            masks.append(mask.detach().bool())
        ground, uncertain, dynamic = masks
        if ((ground & uncertain) | (ground & dynamic) | (uncertain & dynamic)).any():
            raise ValueError('Ground, uncertain and dynamic masks must be disjoint')
        return ground, uncertain, dynamic

    @torch.no_grad()
    def _contact_band(self, ground, dtype):
        """Image-plane ground boundary with a near-vertical contour normal.

        Replicate padding prevents image borders from becoming contact contours.
        gy > 0 selects the upper ground boundary (ground below the object).
        """
        g = ground.to(dtype)
        padded = F.pad(g, (1, 1, 1, 1), mode='replicate')
        kernel = self.sobel_y.to(device=g.device, dtype=dtype)
        gy = F.conv2d(padded, kernel)
        gx = F.conv2d(padded, kernel.transpose(2, 3))
        normal_y = gy.abs() / (gx.square() + gy.square()).sqrt().clamp_min(self.config.eps)
        boundary = ground & (gy > self.config.eps) & (normal_y >= self.config.normal_y_threshold)
        radius = self.config.contact_radius
        return F.max_pool2d(boundary.to(dtype), 2*radius+1, 1, radius).bool()

    @torch.no_grad()
    def _ground_medians(self, depth, ground, candidates):
        """Gather only candidate patches, in chunks; never materialize BHWk²."""
        cfg = self.config
        batch, _, height, width = depth.shape
        radius = cfg.window_size // 2
        padded_width = width + 2*radius
        padded_depth = F.pad(depth, (radius,)*4).flatten(2)
        padded_ground = F.pad(ground, (radius,)*4).flatten(2)
        medians = torch.zeros_like(depth)
        valid = torch.zeros_like(ground)
        dy, dx = torch.meshgrid(torch.arange(cfg.window_size, device=depth.device),
                                torch.arange(cfg.window_size, device=depth.device), indexing='ij')
        offsets = (dy*padded_width + dx).flatten()
        for b in range(batch):
            coords = candidates[b, 0].nonzero(as_tuple=False)
            for chunk in coords.split(cfg.chunk_size):
                if not len(chunk): continue
                rows, cols = chunk.unbind(1)
                indices = (rows*padded_width + cols)[:, None] + offsets[None]
                values = padded_depth[b, 0][indices]
                samples = padded_ground[b, 0][indices]
                enough = samples.sum(1) >= cfg.min_ground_pixels
                values = values.masked_fill(~samples, float('nan'))
                # torch.nanmedian uses the lower middle value for an even count.
                median = values.nanmedian(1).values
                enough &= torch.isfinite(median) & (median > cfg.eps)
                medians[b, 0, rows, cols] = torch.where(enough, median, torch.zeros_like(median))
                valid[b, 0, rows, cols] = enough
        return medians, valid

    @torch.no_grad()
    def build_reference(self, depth, ground, uncertain, dynamic, generator=None):
        """Build Eq.15 offsets and Eq.16 references independently per image."""
        cfg = self.config
        source = depth.detach()
        band = self._contact_band(ground, source.dtype) if self.contact_provider is None else self.contact_provider(ground).detach()
        if band.shape != ground.shape or band.device != ground.device or band.dtype != torch.bool:
            raise ValueError('contact_provider must return a bool mask matching ground')
        candidates = (uncertain | dynamic) & band
        ground_median, valid = self._ground_medians(source, ground, candidates)
        source_anchors = uncertain & valid & band
        dynamic_anchors = dynamic & valid & band
        delta = source.new_zeros(source.shape[0])
        has_offset = torch.zeros(source.shape[0], device=source.device, dtype=torch.bool)
        for b in range(source.shape[0]):
            offsets = (source[b]-ground_median[b])[source_anchors[b]]
            if offsets.numel():
                value = self.offset_reducer(offsets) if self.offset_reducer is not None else (offsets.median() if cfg.offset_aggregation == 'median' else offsets.mean())
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError('offset_reducer must return one finite tensor scalar')
                delta[b] = value
                has_offset[b] = True
        dynamic_anchors &= has_offset[:, None, None, None]
        reference = ground_median + delta[:, None, None, None]
        # Invalid/negative reference depths never enter the data term.
        dynamic_anchors &= torch.isfinite(reference) & (reference > cfg.eps)
        contact_anchors = dynamic_anchors.clone()
        reference, dynamic_anchors = self._propagate(reference, contact_anchors, dynamic, generator)
        return dict(reference=reference, ground_median=ground_median, contact_anchors=contact_anchors,
                    source_anchors=source_anchors, dynamic_anchors=dynamic_anchors,
                    delta=delta, has_offset=has_offset)

    @torch.no_grad()
    def _propagate(self, reference, anchors, dynamic, generator):
        """Scatter local reference depths only inside the anchor's connected object.

        Include the contact anchor itself; random offsets supplement it. Multiple
        references reaching one pixel are averaged without crossing image boundaries.
        """
        from scipy import ndimage
        cfg = self.config
        radius = cfg.propagation_radius
        batch, _, height, width = reference.shape
        result = torch.zeros_like(reference)
        selected = torch.zeros_like(dynamic)
        for b in range(batch):
            coords = anchors[b, 0].nonzero(as_tuple=False)
            if not len(coords): continue
            labels, _ = ndimage.label(dynamic[b, 0].cpu().numpy())
            labels = torch.as_tensor(labels, device=dynamic.device)
            sums = reference.new_zeros(height*width)
            counts = reference.new_zeros(height*width)
            for chunk in coords.split(cfg.chunk_size):
                if cfg.sampling == 'all':
                    yy, xx = torch.meshgrid(torch.arange(-radius, radius+1, device=reference.device),
                                             torch.arange(-radius, radius+1, device=reference.device), indexing='ij')
                    offsets = torch.stack([yy.flatten(), xx.flatten()], 1)[None].expand(len(chunk), -1, -1)
                else:
                    offsets = torch.randint(-radius, radius+1, (len(chunk), cfg.samples_per_anchor, 2),
                                            device=reference.device, generator=generator)
                    offsets = torch.cat([torch.zeros_like(offsets[:, :1]), offsets], 1)
                target = chunk[:, None] + offsets
                rows, cols = target.unbind(-1)
                inside = (rows>=0)&(rows<height)&(cols>=0)&(cols<width)
                safe_rows, safe_cols = rows.clamp(0,height-1), cols.clamp(0,width-1)
                component = labels[chunk[:,0],chunk[:,1]][:,None]
                valid = inside & (labels[safe_rows,safe_cols] == component) & (component > 0)
                values = reference[b,0,chunk[:,0],chunk[:,1]][:,None].expand_as(rows)
                indices = (safe_rows*width+safe_cols)[valid]
                sums.scatter_add_(0, indices, values[valid])
                counts.scatter_add_(0, indices, torch.ones_like(values[valid]))
            result[b,0] = (sums/counts.clamp_min(1)).view(height,width)
            selected[b,0] = (counts>0).view(height,width)
        return result, selected

    def _reduce(self, values, mask):
        masked = values * mask.to(values.dtype)
        if self.config.reduction == 'sum': return masked.sum()
        if self.config.reduction == 'valid_mean':
            return masked.sum() / mask.sum().clamp_min(1).to(values.dtype)
        return masked.mean()

    def forward(self, depth, image, ground_mask, uncertain_mask, dynamic_mask, generator=None):
        ground, uncertain, dynamic = self._check(depth, image, ground_mask, uncertain_mask, dynamic_mask)
        # Use float32 accumulation for AMP/half precision, retaining depth gradients.
        work = depth.float() if depth.dtype in {torch.float16, torch.bfloat16} else depth
        image = image.to(work.dtype)
        result = self.build_reference(work, ground, uncertain, dynamic, generator)
        data = self._reduce((work-result['reference']).abs(), result['dynamic_anchors'])
        inv = work.reciprocal()
        normalized_inv = inv / inv.mean((2, 3), keepdim=True).clamp_min(self.config.eps)
        gradient = (normalized_inv[:, :, 1:] - normalized_inv[:, :, :-1]).abs()
        edge = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(1, keepdim=True)
        # Restrict edges to the same dynamic region at both endpoints.
        smooth_mask = dynamic[:, :, 1:] & dynamic[:, :, :-1]
        smooth = self._reduce(gradient * (-edge).exp(), smooth_mask)
        result.update(loss=data+self.config.smoothness_weight*smooth,
                      data_loss=data, smoothness_loss=smooth,
                      source_count=result['source_anchors'].sum((1,2,3)),
                      dynamic_count=result['dynamic_anchors'].sum((1,2,3)))
        return result
