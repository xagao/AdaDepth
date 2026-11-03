"""Load only the requested backbone; optional backbones have no import side effects."""
from importlib import import_module
_EXPORTS = {
 "ResnetEncoder": ("resnet_encoder", "ResnetEncoder"),
 "DepthDecoder": ("depth_decoder", "DepthDecoder"),
 "PoseDecoder": ("pose_decoder", "PoseDecoder"),
 "FlowNet": ("flownet", "FlowNet"),
 "LiteMono": ("litemono_depth_encoder", "LiteMono"),
 "LiteMonoDepthDecoder": ("litemono_depth_decoder", "LiteMonoDepthDecoder"),
 "HRDepthDecoder": ("HR_Depth_Decoder", "HRDepthDecoder"),
 "hrnet18": ("test_hr_encoder", "hrnet18"),
}
def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    module, attr = _EXPORTS[name]
    return getattr(import_module("." + module, __name__), attr)
