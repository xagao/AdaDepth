"""Architecture names and checkpoint loading shared by training and evaluation."""
from pathlib import Path
import torch
import networks


def depth_pair(backbone, height, width, model='lite-mono-8m', pretrained=False):
    if backbone == 'litemono':
        encoder = networks.LiteMono(model=model, width=width, height=height, drop_path_rate=0.2, pretrained=pretrained)
        decoder = networks.LiteMonoDepthDecoder(encoder.num_ch_enc, [0])
    elif backbone == 'monodepth2':
        encoder = networks.ResnetEncoder(18, pretrained)
        decoder = networks.DepthDecoder(encoder.num_ch_enc, [0])
    elif backbone == 'diffnet':
        encoder = networks.hrnet18(pretrained)
        encoder.num_ch_enc = [64, 18, 36, 72, 144]
        decoder = networks.HRDepthDecoder(encoder.num_ch_enc, [0])
    else:
        raise ValueError('Unsupported backbone: ' + backbone)
    return encoder, decoder


def load_component(model, path):
    """Require every model tensor; allow only known encoder metadata as extras."""
    state = torch.load(Path(path), map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k not in {'height', 'width', 'use_stereo'}}
    model.load_state_dict(state, strict=True)


def load_depth_pair(folder, backbone, height, width, model, branch='refined', device='cpu'):
    encoder, decoder = depth_pair(backbone, height, width, model, pretrained=False)
    suffix = '_ref' if branch == 'refined' else ''
    load_component(encoder, Path(folder) / ('encoder' + suffix + '.pth'))
    load_component(decoder, Path(folder) / ('depth' + suffix + '.pth'))
    return encoder.to(device).eval(), decoder.to(device).eval()
