# L_gac: implementation and configurable choices

Implemented in `adadepth/ground_adherent.py`, integrated in the refinement branch of `adadepth/training_forward.py`. The configuration below defines the released implementation defaults.

## Pipeline

1. Accept positive finite **depth** D (not normalized sigmoid disparity), RGB, and disjoint ground / grounded-uncertain / high-confidence-dynamic masks. No ground-truth depth or semantic labels enter this module.
2. Compute Sobel gradients of the binary ground mask with replicate padding. Keep the upper ground boundary with positive vertical gradient and a near-vertical image-plane normal. Dilate by the configured contact radius.
3. At uncertain and dynamic candidate pixels in this band, gather a local square patch. Retain ground pixels only, excluding padding. Require the minimum valid sample count, then compute the median ground depth.
4. In each image separately, aggregate `D(uncertain_anchor) - ground_median(uncertain_anchor)` into offset δ. Default: median; optional: mean or injected callback. Never pool examples across a batch.
5. For dynamic contact anchors, form `ground_median + δ`. Reject invalid or nonpositive references. If no uncertain source anchor exists, do not fabricate δ=0 targets: disable the reference term for that image.
6. Always supervise each valid contact anchor itself. Additionally sample nearby pixels and transfer its reference depth there. A target must be in the **same 4-connected dynamic component** as the anchor. Discard out-of-bounds samples; do not clamp them into border pixels. Average references when multiple anchors reach the same pixel.
7. Penalize absolute depth deviation at selected targets. Add edge-aware vertical smoothness of mean-normalized inverse depth on edges whose **both endpoints** belong to the dynamic mask. This avoids smoothing an object into the ground.

References, region masks, local medians, and offsets are detached. Gradients flow into the final predicted depth through the reference penalty and inverse-depth smoothness, not into pseudo-target construction. Reference values use the current depth map detached, matching the units of the optimized prediction.

## Defaults

| Setting | Default | Reason / alternative |
|---|---:|---|
| `window_size` | 9 | Local 9×9 ground context; positive odd size ≥3 |
| `contact_radius` | 2 | Narrow band near ground contact |
| `normal_y_threshold` | 0.8 | Reject primarily lateral boundaries |
| `min_ground_pixels` | 3 | Avoid one-pixel ground references |
| `offset_aggregation` | median | Robust to isolated uncertain-region outliers; mean optional |
| `propagation_radius` | 2 | Local 5×5 neighborhood; zero disables propagation |
| `samples_per_anchor` | 4 | Four random offsets plus the anchor itself, with replacement |
| `sampling` | random | Use `all` to supervise all local offsets deterministically |
| `reduction` | image_mean | Masked sum / full tensor size, consistent with surrounding masked image losses; `valid_mean` and `sum` available |
| `smoothness_weight` | 1 | Internal relative weight; zero isolates reference term |
| `gac_weight` | 0.05 | Outer multiplier in the refinement loss |
| `chunk_size` | 2048 | Bound patch-gather memory; avoids materializing B×H×W×window² |

PyTorch median uses the lower middle element for even counts. Random sampling follows the Torch RNG; the optional `generator` argument allows reproducible isolated calls and must be created for the same device as depth. Changing chunk size preserves deterministic `sampling=all` results; stochastic draws need not stay identical across different chunking/device settings. Connected-component membership is computed on CPU with SciPy; patches, references and losses run on the depth device. GPU timings have not been benchmarked.

No ground / no uncertain anchors / no dynamic anchors yield finite zero reference loss, with per-image counts logged. Dynamic smoothness can still be nonzero when references are unavailable. Small images, border contacts, mixed precision accumulation, and empty regions are handled explicitly. Half/bfloat inputs accumulate in float32.

## Configuration

Dataset JSON presets include all defaults. For deterministic inspection:

```json
{
  "gac_weight": 0.05,
  "gac": {
    "window_size": 9,
    "contact_radius": 2,
    "min_ground_pixels": 3,
    "normal_y_threshold": 0.8,
    "offset_aggregation": "median",
    "propagation_radius": 2,
    "samples_per_anchor": 4,
    "sampling": "all",
    "reduction": "image_mean",
    "smoothness_weight": 1.0,
    "chunk_size": 2048,
    "eps": 1e-7
  }
}
```

Use `gac_weight=0` for an ablation of its contribution. Actual settings are written into saved config.json. Training logs contain `gac/loss`, `gac/data`, `gac/smoothness`, and source/target anchor counts; inspect those counts before a long run.

## Pluggable components

```python
from adadepth.ground_adherent import GroundAdherentLoss
criterion = GroundAdherentLoss(
    {"window_size": 9, "sampling": "all"},
    contact_provider=my_contact_band,  # bool B×1×H×W, same device as input mask
    offset_reducer=lambda offsets: offsets.median(),  # one finite tensor scalar
)
result = criterion(depth, rgb, ground, uncertain, dynamic)
loss = result["loss"]
```

A contact provider takes a ground mask and returns the candidate band. The offset callback receives one image's valid offset samples. Both operate within detached target construction. For alternative local statistics or propagation strategies, subclass `_ground_medians` or `_propagate`, retaining their shape/validity contracts. Custom Python callbacks must have their identity/settings recorded by the caller; arbitrary callback code is not serialized in config.json.

## Training integration

At refinement epochs, the trainer maps its ground, grounded-uncertain, and dynamic regions into disjoint masks; converts disparity to depth; and adds the weighted GAC result to its objective. The output dictionary exposes reference depths and offsets, and training logs record source/target anchor counts.
