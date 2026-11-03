"""Paper-equation primitives. These do not establish a complete reproduction.

End-to-end Eq.15–17 reference construction is in ground_adherent.py.
The low-level helpers here remain available for isolated equation checks.
"""
import torch
from layers import SSIM, get_smooth_loss


def photo(pred, target):
    return 0.15 * (pred-target).abs().mean(1, keepdim=True) + 0.85 * SSIM().to(pred.device)(pred, target).mean(1, keepdim=True)


def masked_mean_over_image(value, mask):
    if value.shape != mask.shape:
        raise ValueError('Expected equal Bx1xHxW shapes; broadcasting across samples is forbidden')
    return (value * mask.detach()).mean()


def motion_robust(raw_error, pseudo_error, dynamic_mask):
    """Eq. 13 after temporal-source reduction; full-image mean reduction."""
    return masked_mean_over_image(torch.minimum(raw_error, pseudo_error), 1-dynamic_mask)


def residual_sparsity(flow, rigid_photo_error):
    """Eq. 7, per-image threshold, L1 over flow components."""
    omega = rigid_photo_error.detach() < rigid_photo_error.detach().mean((2, 3), keepdim=True)
    return masked_mean_over_image(flow.abs().sum(1, keepdim=True), omega)


def ground_adherent(depth, reference, dynamic_mask, anchor_mask, image):
    """Eq. 17 given validated reference depths and anchors; image-mean convention.

    References are detached to prevent target collapse. This is an explicit
    reconstruction choice, not recovered proof of the historical implementation.
    """
    anchor = dynamic_mask * anchor_mask
    l1 = masked_mean_over_image((depth-reference.detach()).abs(), anchor)
    inv = depth.reciprocal()
    inv = inv / inv.mean((2, 3), keepdim=True).clamp_min(1e-7)
    dy = (inv[:, :, 1:] - inv[:, :, :-1]).abs()
    edge = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(1, keepdim=True)
    return l1 + (dy * (-edge).exp() * dynamic_mask[:, :, :-1].detach()).mean()


def refinement_objective(mrr, masked_reprojection, gac, smoothness):
    """Eq. 18; do not call with a fabricated/zero ground-adherent constraint."""
    if gac is None:
        raise NotImplementedError('Supply the computed GroundAdherentLoss result; gac=None cannot define the objective')
    return 0.85*mrr + 0.15*masked_reprojection + 0.05*gac + 0.001*smoothness
