from importlib import import_module
_EXPORTS = {"KITTIRAWDataset": "kitti_dataset", "CityscapesDataset": "cityscapes_dataset", "WaymoDataset": "waymo_dataset"}
def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    return getattr(import_module("." + _EXPORTS[name], __name__), name)
