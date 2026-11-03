from pathlib import Path
import datasets

ROOT = Path(__file__).resolve().parents[1]


def make_dataset(config, train=True, split_file=None, load_mask=False):
    name = config['dataset']
    split = Path(split_file or config['train_split'])
    if not split.is_absolute():
        split = ROOT / split
    if not split.is_file():
        raise FileNotFoundError('Missing split file: ' + str(split))
    filenames = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    common = dict(data_path=config['data_path'], filenames=filenames,
                  height=config['height'], width=config['width'],
                  frame_idxs=[0, -1, 1] if train else [0], num_scales=1,
                  is_train=train, img_ext=config.get('img_ext', '.png'),
                  load_depth=False, load_mask=load_mask)
    if name == 'kitti':
        dataset = datasets.KITTIRAWDataset(**common)
    elif name == 'cityscapes':
        dataset = datasets.CityscapesDataset(**common)
    elif name == 'waymo':
        common['load_depth'] = not train
        dataset = datasets.WaymoDataset(cam_name='FRONT', img_type='original', **common)
    else:
        raise ValueError(name)
    dataset.mask_path = config.get('mask_path')
    # This flag is required even if a dataset implementation changes its defaults.
    if train and (dataset.load_depth or dataset.load_mask):
        raise RuntimeError('Training must not load depth or segmentation labels')
    return dataset
