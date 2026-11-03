# Data layout

Training loads image triplets and camera intrinsics. Evaluation loads the held-out split and its annotations.

## KITTI

Use KITTI raw sequences with the original date/drive directory structure. Split entries are `sequence frame_index l/r`. Eigen training and test image lists are supplied. Export or download the matching `gt_depths.npz` separately for evaluation; it is not stored in this repository.

## Cityscapes

Training uses a vertically stacked triplet file and a comma-separated `*_cam.txt` intrinsics file for each split entry. Each frame has its bottom 25% removed before stacking. The loader splits the stack into previous/target/next frames.

Evaluation expects the original `leftImg8bit/test/<city>/` and `camera/test/<city>/` structure. Depth files are ordered as `000_depth.npy`, `001_depth.npy`, … in the GT directory, aligned with the supplied test list. The evaluation crop is applied in evaluate_depth.py.

Optional object-class masks use `<city>_<sequence>_<integer_frame>.npy`; supply their directory with `--mask-path`. These are evaluation-only masks.

## Waymo Open

Use the preprocessed sequence layout:

```text
<root>/<split>/<sequence>/FRONT/
├── rgb/
│   ├── cam.json
│   └── original/000001.jpg
├── depth/000001.npy
└── mask/
    ├── 000001.npz
    └── 000001.pickle
```

`cam.json` contains `intrinsic_mat`. Sparse depth arrays are N×3 and follow the existing loader's x/y/depth ordering. Region evaluation uses semantic/motion files; moving objects are label 1, static objects label 2, and unlabeled objects label 3. Missing evaluation annotations raise an error.
