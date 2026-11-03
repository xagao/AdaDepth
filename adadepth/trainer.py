"""Configuration-driven depth training with staged optimization."""
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import BackprojectDepth, Project3D, D_to_cloud_mask, GroundPlane, SSIM
import networks
from .models import depth_pair, load_component
from .data import make_dataset
from .training_forward import TrainingForward
from .ground_adherent import GroundAdherentLoss
from dataclasses import asdict


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class Trainer(TrainingForward):
    def __init__(self, config):
        self.config = config
        defaults = dict(scales=[0], frame_ids=[0, -1, 1], no_cuda=False,
                        pose_model_type='separate_resnet', pose_model_input='pairs',
                        v1_multiscale=False, use_stereo=False, predictive_mask=False,
                        avg_reprojection=False, disable_automasking=False, no_ssim=False,
                        min_depth=0.1, max_depth=100., disparity_smoothness=0.001,
                        freeze_teacher_epoch=10, train_rigid_flow_epoch=10,
                        train_ref_epoch=15, train_cat_dynamic_epoch=15,
                        train_only_dynamic_epoch=1000000000, median_init=0.2)
        defaults.update(config)
        self.opt = SimpleNamespace(**defaults)
        if self.opt.height % 32 or self.opt.width % 32:
            raise ValueError('Resolution must be divisible by 32')
        if self.opt.scales != [0] or self.opt.frame_ids != [0, -1, 1]:
            raise ValueError('Training supports scale 0 and frames [0,-1,1] only')
        self.device = torch.device(config.get('device', 'cuda'))
        seed = config.get('seed', 42)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.models = {}
        self.models['encoder'], self.models['depth'] = depth_pair('litemono', self.opt.height, self.opt.width, 'lite-mono')
        self.models['encoder_ref'], self.models['depth_ref'] = depth_pair(config.get('backbone', 'litemono'), self.opt.height, self.opt.width, config.get('model', 'lite-mono-8m'))
        for suffix in ['', '_ref']:
            encoder = networks.ResnetEncoder(18, False, num_input_images=2)
            self.models['pose_encoder'+suffix] = encoder
            self.models['pose'+suffix] = networks.PoseDecoder(encoder.num_ch_enc, 1, 2)
        self.models['flow'] = networks.FlowNet(12, 0.1)
        self.models['flow_rigid'] = networks.FlowNet(9, 0.1)
        for model in self.models.values():
            model.to(self.device)
        if config.get('initialize_from'):
            folder = Path(config['initialize_from'])
            for name, model in self.models.items():
                load_component(model, folder/(name+'.pth'))
        self.gac = GroundAdherentLoss(config.get('gac', {})).to(self.device)
        self.gac_weight = config.get('gac_weight', 0.05)
        if self.gac_weight < 0:
            raise ValueError('gac_weight must be nonnegative')
        self.config['gac'] = asdict(self.gac.config)
        self.config['gac_weight'] = self.gac_weight
        self.num_scales = 1
        self.num_pose_frames = 2
        self.num_input_frames = 3
        self.use_pose_net = True
        self.min_depth_tracker, self.max_depth_tracker = 0.1, 0.5
        self.median_depth_tracker = self.opt.median_init
        self.ssim = SSIM().to(self.device)
        self.cloud_to_mask = D_to_cloud_mask(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
        self.gplane = GroundPlane(num_points_per_it=5, max_it=100, tol=0.005, g_prior=0.4)
        self.backproject_depth = {0: BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)}
        self.project_3d = {0: Project3D(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)}
        self.optimizer = torch.optim.Adam([p for m in self.models.values() for p in m.parameters()], lr=config.get('learning_rate', 8e-5))
        self.epoch, self.step = 0, 0
        self.output_dir = Path(config.get('output_dir', 'runs/adadepth'))

    def set_stage(self, epoch):
        self.epoch = epoch
        if epoch < 10:
            active = {'encoder', 'depth', 'pose_encoder', 'pose', 'flow'}
            stage = 'prior'
        elif epoch < 15:
            active = {'flow_rigid'}
            stage = 'pseudo_rigid'
        else:
            active = {'encoder_ref', 'depth_ref', 'pose_encoder_ref', 'pose_ref'}
            stage = 'refinement_with_gac'
        for name, model in self.models.items():
            model.train(name in active)
            model.requires_grad_(name in active)
        lr = self.config.get('learning_rate', 8e-5) * (0.1 if epoch >= 30 else 1)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return stage

    def save(self):
        folder = self.output_dir / ('weights_%02d' % self.epoch)
        folder.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            torch.save(model.state_dict(), folder/(name+'.pth'))
        torch.save(dict(epoch=self.epoch, step=self.step, optimizer=self.optimizer.state_dict(),
                        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        median_depth_tracker=self.median_depth_tracker,
                        min_depth_tracker=self.min_depth_tracker, max_depth_tracker=self.max_depth_tracker), folder/'training_state.pth')
        (folder/'config.json').write_text(json.dumps(self.config, indent=2)+'\n')
        return folder

    def train(self, max_steps=None):
        dataset = make_dataset(self.config, train=True)
        generator = torch.Generator().manual_seed(self.config.get('seed', 42))
        loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, drop_last=True,
                            num_workers=self.config.get('num_workers', 4),
                            worker_init_fn=seed_worker, generator=generator)
        if not len(loader):
            raise ValueError('Training split is smaller than batch_size')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir/'config.json').write_text(json.dumps(self.config, indent=2)+'\n')
        with (self.output_dir/'metrics.jsonl').open('a') as log:
            for epoch in range(self.config.get('start_epoch', 0), self.config.get('num_epochs', 35)):
                stage = self.set_stage(epoch)
                for batch in loader:
                    # Assert that annotation-bearing batches can never enter the optimizer.
                    if any(key in batch for key in ['depth_gt', 'doj_mask', 'semantic_mask', 'motion_mask', 'sem_mask', 'mot_mask']):
                        raise RuntimeError('Annotation found in training batch')
                    _, losses = self.process_batch(batch)
                    loss = losses['loss']
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss')
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    self.step += 1
                    record = dict(epoch=epoch, step=self.step, stage=stage,
                                  loss=float(loss.detach()), time=time.time(),
                                  implementation='adadepth-release', gac_weight=self.gac_weight)
                    for key, value in losses.items():
                        if key.startswith('gac/'):
                            record[key] = float(value.detach())
                    log.write(json.dumps(record)+'\n'); log.flush()
                    if self.step % 20 == 0 or self.step == 1:
                        print(record, flush=True)
                    if max_steps and self.step >= max_steps:
                        print('Saved:', self.save()); return
                print('Saved:', self.save())
