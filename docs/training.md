# Training settings

The release provides configuration-driven, single-process training. The supplied weights are archived experiment checkpoints; the current training implementation includes configurable GAC reconstruction choices described in GROUND_ADHERENT.md.

| Stage | Epochs | Trainable components |
|---|---|---|
| Structural prior | 0–9 | prior depth, prior pose, residual flow |
| Pseudo-rigid flow | 10–14 | pseudo-rigid flow |
| Depth refinement | 15–34 | final depth and final pose |

The default learning rate is 8e-5 and drops by a factor of ten at epoch 30. Adjust batch size and workers in the dataset JSON for your hardware. Each configuration includes GAC window/sampling settings and its outer loss multiplier. A zero GAC weight is available for ablation.

`--max-steps 1` runs a short data/gradient/checkpoint check. `--dry-run` prints the effective input configuration. `--initialize-from DIR` loads matching component files for initialization; exact optimizer resume and multi-GPU training are not included.

Outputs contain per-network weights, a training-state snapshot, config.json, and metrics.jsonl. Logs include GAC reference and smoothness losses and valid anchor counts.

Run the implementation checks with `bash scripts/check.sh`.
