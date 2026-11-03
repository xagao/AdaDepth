"""Depth metrics with an explicit log base and no implicit rescaling."""
import torch
NAMES = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']


def depth_metrics(gt, pred, log_base='e'):
    if gt.numel() == 0:
        raise ValueError('No valid ground-truth samples')
    if not ((gt > 0).all() and (pred > 0).all() and torch.isfinite(gt).all() and torch.isfinite(pred).all()):
        raise ValueError('Depth values must be positive and finite')
    threshold = torch.maximum(gt/pred, pred/gt)
    log = torch.log if log_base == 'e' else torch.log10
    return torch.stack([(gt-pred).abs().div(gt).mean(), ((gt-pred)**2/gt).mean(),
                        ((gt-pred)**2).mean().sqrt(), ((log(gt)-log(pred))**2).mean().sqrt(),
                        (threshold < 1.25).float().mean(), (threshold < 1.25**2).float().mean(),
                        (threshold < 1.25**3).float().mean()])


def scale_prediction(gt, pred, minimum, maximum, mode='median'):
    ratio = gt.median()/pred.median() if mode == 'median' else pred.new_tensor(1.)
    if not torch.isfinite(ratio) or ratio <= 0:
        raise ValueError('Invalid scale ratio')
    return (pred*ratio).clamp(minimum, maximum), ratio
