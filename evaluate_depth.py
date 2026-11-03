"""Streaming KITTI / Cityscapes / Waymo evaluator. No optimizer or training imports."""
import argparse
import json
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['kitti','cityscapes','waymo'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--backbone', choices=['litemono','monodepth2','diffnet'], default='litemono')
    p.add_argument('--model', default='lite-mono-8m')
    p.add_argument('--branch', choices=['prior','refined'], default='refined')
    p.add_argument('--split-file', required=True)
    p.add_argument('--mask-path', help='Cityscapes evaluation-only object-mask directory')
    p.add_argument('--gt-path', help='KITTI gt_depths.npz or Cityscapes gt_depths directory')
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--img-ext', default='.png')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda', choices=['cpu','cuda'])
    p.add_argument('--scaling', default='median', choices=['median','none'])
    p.add_argument('--log-base', default='e', choices=['e','10'])
    p.add_argument('--regions', action='store_true', help='Cityscapes object masks or Waymo motion masks; evaluation only')
    p.add_argument('--region-scaling', default='image', choices=['image','region'], help='Use one image scale, or explicitly reproduce separate region rescaling')
    p.add_argument('--output', default='evaluation.json')
    p.add_argument('--max-images', type=int, help='Debug subset; output is labeled partial')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main(args):
    if args.dry_run:
        print(json.dumps(vars(args), indent=2)); return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from adadepth.models import load_depth_pair
    from adadepth.data import make_dataset
    from adadepth.metrics import NAMES, depth_metrics, scale_prediction
    from layers import disp_to_depth
    if args.dataset != 'waymo' and not args.gt_path:
        raise ValueError('--gt-path is required for KITTI / Cityscapes')
    if args.regions and args.dataset == 'kitti':
        raise ValueError('KITTI region evaluation requires a separate region protocol')
    config = dict(dataset=args.dataset, data_path=args.data_path, height=args.height,
                  width=args.width, img_ext=args.img_ext, train_split=args.split_file, mask_path=args.mask_path)
    dataset = make_dataset(config, train=False, split_file=args.split_file, load_mask=args.regions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    encoder, decoder = load_depth_pair(args.weights, args.backbone, args.height, args.width,
                                        args.model, args.branch, args.device)
    gt_depths = None
    if args.dataset == 'kitti':
        gt_depths = np.load(args.gt_path, allow_pickle=True, encoding='latin1')['data']
        if len(gt_depths) != len(dataset):
            raise ValueError('GT and split lengths differ')
    errors, ratios = [], []
    region_errors = {'dynamic':[], 'static':[]}
    processed, skipped = 0, 0
    maximum = 75. if args.dataset == 'waymo' else 80.
    minimum = 1e-3
    with torch.inference_mode():
        for batch in loader:
            prediction = decoder(encoder(batch['color', 0, 0].to(args.device)))['disp', 0]
            disparity, _ = disp_to_depth(prediction, 0.1, 100.)
            disparity = disparity.cpu()
            for b in range(disparity.shape[0]):
                index = processed
                if args.max_images and index >= args.max_images:
                    break
                regions = {}
                if args.dataset == 'waymo':
                    height, width = map(int, batch['gt_dim'][b])
                    dense = F.interpolate(disparity[b:b+1], (height,width), mode='bilinear', align_corners=False)[0,0].reciprocal()
                    points = batch['depth_gt'][b]
                    valid = batch['depth_valid'][b].bool()
                    valid &= (points[:,0]>=0)&(points[:,0]<height)&(points[:,1]>=0)&(points[:,1]<width)
                    valid &= (points[:,2]>minimum)&(points[:,2]<maximum)
                    points = points[valid]
                    rows, cols = points[:,0].long(), points[:,1].long()
                    gt, pred = points[:,2], dense[rows,cols]
                    if args.regions:
                        motion = batch['mot_mask'][b][rows,cols]
                        regions = {'dynamic':motion==1, 'static':motion==2}
                else:
                    raw = gt_depths[index] if args.dataset == 'kitti' else np.load(Path(args.gt_path)/('%03d_depth.npy'%index))
                    gt = torch.as_tensor(raw, dtype=torch.float32)
                    if args.dataset == 'cityscapes':
                        gt = gt[:int(round(gt.shape[0]*0.75))]
                    pred = F.interpolate(disparity[b:b+1], tuple(gt.shape), mode='bilinear', align_corners=True)[0,0].reciprocal()
                    if args.dataset == 'cityscapes':
                        if args.regions:
                            mask = F.interpolate(batch['doj_mask'][b:b+1].float(), tuple(gt.shape), mode='nearest')[0,0].bool()
                            regions = {'dynamic':mask[256:,192:1856]}
                        gt, pred = gt[256:,192:1856], pred[256:,192:1856]
                    valid = (gt>minimum)&(gt<maximum)
                    if args.dataset == 'kitti':
                        height,width = gt.shape
                        crop = torch.zeros_like(valid)
                        crop[int(.40810811*height):int(.99189189*height),int(.03594771*width):int(.96405229*width)] = True
                        valid &= crop
                    regions = {k:v[valid] for k,v in regions.items()}
                    gt,pred = gt[valid],pred[valid]
                processed += 1
                if not gt.numel():
                    skipped += 1; continue
                scaled, ratio = scale_prediction(gt,pred,minimum,maximum,args.scaling)
                errors.append(depth_metrics(gt,scaled,args.log_base))
                ratios.append(float(ratio))
                for name, mask in regions.items():
                    if not mask.any(): continue
                    rp = scaled[mask]
                    if args.region_scaling == 'region':
                        rp,_ = scale_prediction(gt[mask],pred[mask],minimum,maximum,args.scaling)
                    region_errors[name].append(depth_metrics(gt[mask],rp,args.log_base))
            if args.max_images and processed >= args.max_images: break
    if not errors: raise RuntimeError('No images with valid GT')
    result = dict(protocol=vars(args), processed=processed, skipped=skipped,
                  partial=processed<len(dataset), image_mean_metrics=dict(zip(NAMES,torch.stack(errors).mean(0).tolist())),
                  scale_ratio_median=float(np.median(ratios)),
                  region_metrics={k:dict(images=len(v),metrics=dict(zip(NAMES,torch.stack(v).mean(0).tolist()))) for k,v in region_errors.items() if v})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main(arguments())
