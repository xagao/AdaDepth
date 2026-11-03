# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

"""Depth, pose, flow, scene-region and reprojection operations for training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage
from layers import *

class TrainingForward:
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
            """
        for (key, ipt) in inputs.items():
            inputs[key] = ipt.to(self.device)
        if self.opt.pose_model_type == 'shared':
            all_color_aug = torch.cat([inputs['color_aug', i, 0] for i in self.opt.frame_ids])
            all_features = self.models['encoder'](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {}
            for (i, k) in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            outputs = self.models['depth'](features[0])
        else:
            features = self.models['encoder'](inputs['color_aug', 0, 0])
            outputs = self.models['depth'](features)
            features_ref = self.models['encoder_ref'](inputs['color_aug', 0, 0])
            outputs_ref = self.models['depth_ref'](features_ref)
        if self.opt.predictive_mask:
            outputs['predictive_mask'] = self.models['predictive_mask'](features)
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
            outputs.update(self.predict_poses_ref(inputs, features))
        self.generate_images_pred(inputs, outputs)
        for f_i in self.opt.frame_ids[1:]:
            self.estimate_flow(inputs, outputs, f_i)
        self.generate_mask(inputs, outputs)
        with torch.no_grad():
            pretrain_disp = outputs['disp', 0]
            (_, pretrain_dpepth) = disp_to_depth(pretrain_disp, 0.1, 100)
            '------------------------------ see ground mask----------------------- '
            (plane_mask, raw_plane_param, plane_param) = self.compute_ground_mask(pretrain_dpepth, inputs['inv_K', 0], inputs['K', 0], self.opt.height, self.opt.width)
            outputs['ground_mask'] = plane_mask
            outputs['img_groudmask'] = inputs['color', 0, 0] * outputs['ground_mask']
            '------------------------------------------------------------------- '
            y_min_per_batch = calculate_y_min_per_batch(plane_mask, self.opt.height)
            cam_points = self.cloud_to_mask(pretrain_dpepth, inputs['inv_K', 0], inputs['K', 0], threshold=-0.005, T=None)
            mean_ground_points = compute_mean_ground_points(cam_points, outputs['ground_mask'], self.opt.height, self.opt.width)
            motion_points = cam_points[:, :3].reshape(-1, 3, self.opt.height, self.opt.width) * (outputs['mask'] > 0.4)
            adjusted_y_distances = compute_adjusted_y_distances(motion_points, mean_ground_points, plane_param, outputs['mask'], self.opt.height, self.opt.width)
            outputs['motion_distances'] = visualize_distances(adjusted_y_distances)
            self.process_mask_labels(outputs, adjusted_y_distances, inputs, y_min_per_batch)
        for f_i in self.opt.frame_ids[1:]:
            self.estimate_rigid_flow(inputs, outputs, f_i)
        self.generate_images_pred_ref(inputs, outputs, outputs_ref)
        self.generate_images_full_flow(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        self.update_adaptive_depth_bins(outputs)
        return (outputs, losses)

    def generate_images_full_flow(self, inputs, outputs):
        for frame_id in self.opt.frame_ids[1:]:
            rigid_pix = outputs['sample', frame_id, 0].detach()
            res_flow = outputs['resflow', 0, frame_id].permute(0, 2, 3, 1)
            full_pix = rigid_pix + res_flow
            outputs['sample_full', frame_id] = full_pix
            outputs['color_full', frame_id, 0] = F.grid_sample(inputs['color', frame_id, 0], full_pix, padding_mode='border', align_corners=True)
            with torch.no_grad():
                pix_coords_ = full_pix.detach() / 2 + 0.5
                pix_coords_[..., 0] *= self.opt.width - 1
                pix_coords_[..., 1] *= self.opt.height - 1
                full_flow = self.compute_depth_flow(pix_coords_, self.opt.height, self.opt.width)
                full_flow = full_flow.permute(0, 3, 1, 2)
                outputs['full_flow', frame_id, 0] = full_flow
        for f_i in self.opt.frame_ids[1:]:
            with torch.no_grad():
                rigid_error = image_similarity(inputs['color', 0, 0], outputs['color_full', f_i, 0]).detach()
                outputs['rigid_error_flow', f_i] = L2_norm(rigid_error, dim=1)

    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""
        min_depth = outputs['depth', 0, 0].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs['depth', 0, 0].detach().max(-1)[0].max(-1)[0]
        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1
        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        if self.max_depth_tracker < 0.5:
            self.max_depth_tracker = 0.5
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01
        depth = outputs['depth', 0, 0].detach()
        valid_depth = depth[depth <= 80]
        if valid_depth.numel() > 0:
            median_depth = valid_depth.median(dim=-1)[0].median(dim=-1)[0]
            median_depth = median_depth.mean().cpu().item()
            self.median_depth_tracker = self.median_depth_tracker * 0.99 + median_depth * 0.01
        else:
            median_depth = self.median_depth_tracker

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
            """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == 'shared':
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs['color_aug', f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != 's':
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]
                    if self.opt.pose_model_type == 'separate_resnet':
                        pose_inputs = [self.models['pose_encoder'](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == 'posecnn':
                        pose_inputs = torch.cat(pose_inputs, 1)
                    (axisangle, translation) = self.models['pose'](pose_inputs)
                    outputs['axisangle', 0, f_i] = axisangle
                    outputs['translation', 0, f_i] = translation
                    outputs['cam_T_cam', 0, f_i] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=f_i < 0)
        else:
            if self.opt.pose_model_type in ['separate_resnet', 'posecnn']:
                pose_inputs = torch.cat([inputs['color_aug', i, 0] for i in self.opt.frame_ids if i != 's'], 1)
                if self.opt.pose_model_type == 'separate_resnet':
                    pose_inputs = [self.models['pose_encoder'](pose_inputs)]
            elif self.opt.pose_model_type == 'shared':
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != 's']
            (axisangle, translation) = self.models['pose'](pose_inputs)
            for (i, f_i) in enumerate(self.opt.frame_ids[1:]):
                if f_i != 's':
                    outputs['axisangle', 0, f_i] = axisangle
                    outputs['translation', 0, f_i] = translation
                    outputs['cam_T_cam', 0, f_i] = transformation_from_parameters(axisangle[:, i], translation[:, i])
        return outputs

    def predict_poses_ref(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
            """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == 'shared':
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs['color_aug', f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != 's':
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]
                    if self.opt.pose_model_type == 'separate_resnet':
                        pose_inputs = [self.models['pose_encoder_ref'](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == 'posecnn':
                        pose_inputs = torch.cat(pose_inputs, 1)
                    (axisangle, translation) = self.models['pose_ref'](pose_inputs)
                    outputs['axisangle_ref', 0, f_i] = axisangle
                    outputs['translation_ref', 0, f_i] = translation
                    outputs['cam_T_cam_ref', 0, f_i] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=f_i < 0)
        else:
            if self.opt.pose_model_type in ['separate_resnet', 'posecnn']:
                pose_inputs = torch.cat([inputs['color_aug', i, 0] for i in self.opt.frame_ids if i != 's'], 1)
                if self.opt.pose_model_type == 'separate_resnet':
                    pose_inputs = [self.models['pose_encoder'](pose_inputs)]
            elif self.opt.pose_model_type == 'shared':
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != 's']
            (axisangle, translation) = self.models['pose'](pose_inputs)
            for (i, f_i) in enumerate(self.opt.frame_ids[1:]):
                if f_i != 's':
                    outputs['axisangle', 0, f_i] = axisangle
                    outputs['translation', 0, f_i] = translation
                    outputs['cam_T_cam', 0, f_i] = transformation_from_parameters(axisangle[:, i], translation[:, i])
        return outputs

    def compute_depth_flow(self, pix_coords, height, width):
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.from_numpy(id_coords)
        id_coords = id_coords.permute(1, 2, 0).unsqueeze(0).to(self.device)
        depth_flow = pix_coords - id_coords
        return depth_flow

    def generate_images_pred_ref(self, inputs, outputs, outputs_ref):
        """Generate the warped (reprojected) color images for a minibatch.
            Generated images are saved into the `outputs` dictionary.
            """
        for scale in self.opt.scales:
            disp = outputs_ref['disp', scale]
            outputs['disp_ref', scale] = disp
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode='bilinear', align_corners=False)
                source_scale = 0
            (_, depth) = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs_ref['depth_ref', 0, scale] = depth
            for (i, frame_id) in enumerate(self.opt.frame_ids[1:]):
                if frame_id == 's':
                    T = inputs['stereo_T']
                else:
                    T = outputs['cam_T_cam_ref', 0, frame_id]
                if self.opt.pose_model_type == 'posecnn':
                    axisangle = outputs['axisangle', 0, frame_id]
                    translation = outputs['translation', 0, frame_id]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                cam_points = self.backproject_depth[source_scale](depth, inputs['inv_K', source_scale])
                outputs['cam_points_ref', scale] = cam_points
                pix_coords = self.project_3d[source_scale](cam_points, inputs['K', source_scale], T)
                outputs['sample_ref', frame_id, scale] = pix_coords
                outputs['color_ref', frame_id, scale] = F.grid_sample(inputs['color', frame_id, source_scale], outputs['sample_ref', frame_id, scale], padding_mode='border', align_corners=True)
                color_rigid_estimate_detach = outputs['color_rigid_estimate', frame_id, scale].detach()
                outputs['color_ref_maskregion', frame_id, scale] = F.grid_sample(color_rigid_estimate_detach, outputs['sample_ref', frame_id, scale], padding_mode='border', align_corners=True)
                with torch.no_grad():
                    pix_coords_ = pix_coords.detach() / 2 + 0.5
                    pix_coords_[..., 0] *= self.opt.width - 1
                    pix_coords_[..., 1] *= self.opt.height - 1
                    rigid_flow = self.compute_depth_flow(pix_coords_, self.opt.height, self.opt.width)
                    rigid_flow = rigid_flow.permute(0, 3, 1, 2)
                    outputs['rigid_flow_ref', frame_id, scale] = rigid_flow

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
            Generated images are saved into the `outputs` dictionary.
            """
        for scale in self.opt.scales:
            disp = outputs['disp', scale]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode='bilinear', align_corners=False)
                source_scale = 0
            (_, depth) = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs['depth', 0, scale] = depth
            for (i, frame_id) in enumerate(self.opt.frame_ids[1:]):
                if frame_id == 's':
                    T = inputs['stereo_T']
                else:
                    T = outputs['cam_T_cam', 0, frame_id]
                if self.opt.pose_model_type == 'posecnn':
                    axisangle = outputs['axisangle', 0, frame_id]
                    translation = outputs['translation', 0, frame_id]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                cam_points = self.backproject_depth[source_scale](depth, inputs['inv_K', source_scale])
                outputs['cam_points', scale] = cam_points
                pix_coords = self.project_3d[source_scale](cam_points, inputs['K', source_scale], T)
                with torch.no_grad():
                    pix_coords_ = pix_coords.detach() / 2 + 0.5
                    pix_coords_[..., 0] *= self.opt.width - 1
                    pix_coords_[..., 1] *= self.opt.height - 1
                    rigid_flow = self.compute_depth_flow(pix_coords_, self.opt.height, self.opt.width)
                    rigid_flow = rigid_flow.permute(0, 3, 1, 2)
                    outputs['rigid_flow', frame_id, scale] = rigid_flow
                outputs['sample', frame_id, scale] = pix_coords
                outputs['color', frame_id, scale] = F.grid_sample(inputs['color', frame_id, source_scale], outputs['sample', frame_id, scale], padding_mode='border', align_corners=True)
                if frame_id == 1:
                    tensor = outputs['sample', frame_id, scale].detach()
                    combined_mask = (tensor > 1) | (tensor < -1)
                    combined_mask = combined_mask.any(dim=-1)
                    outputs['edge_mask'] = combined_mask.unsqueeze(1)
                rig_flow = torch.abs(rigid_flow)
                if frame_id == 1:
                    rig_flow *= ~outputs['edge_mask']
                disp_mag = torch.sum(rig_flow ** 2, 1)
                batch_means = disp_mag.mean(dim=(1, 2), keepdim=True)
                if not self.opt.disable_automasking:
                    outputs['color_identity', frame_id, scale] = inputs['color', frame_id, source_scale]
                outputs['forward_warp', frame_id, scale] = forward_warp_bilinear(inputs['color', frame_id, source_scale], outputs['rigid_flow', frame_id, scale].detach().permute(0, 2, 3, 1))
                forward_error = image_similarity(inputs['color', frame_id, 0].detach(), outputs['forward_warp', frame_id, scale].detach())
                outputs['rigid_error_forward_warp', frame_id] = L2_norm(forward_error, dim=1)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
            """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_distiil_loss(self, pred, target):
        abs_diff = torch.abs(pred - target)
        l1_loss = abs_diff.mean(1, True)
        l1_loss = l1_loss.mean()
        return l1_loss

    def compute_percentile(self, tensor, percentile):
        k = int(percentile * tensor.size(1))
        return tensor.kthvalue(k, dim=1)[0]

    def estimate_flow(self, inputs, outputs, f_i):
        """Estimate flow for a given frame"""
        with torch.no_grad():
            rigid_error = image_similarity(inputs['color', 0, 0], outputs['color', f_i, 0]).detach()
            warped_tgt_image = outputs['color', f_i, 0].detach()
        rigid_flow = outputs['rigid_flow', f_i, 0].detach()
        flownet_input = torch.cat((inputs['color', 0, 0], inputs['color', f_i, 0], warped_tgt_image.detach(), rigid_flow, L2_norm(rigid_error, dim=1)), dim=1)
        self.resflow = self.models['flow'](flownet_input)
        outputs['rigid_error', f_i] = L2_norm(rigid_error, dim=1)
        for i in range(4):
            outputs['resflow', i, f_i] = self.resflow[i]

    def estimate_rigid_flow(self, inputs, outputs, f_i):
        """Estimate rigid flow for a given frame"""
        depth_tensor = outputs['depth', 0, 0].detach()
        depth_tensor_flat = depth_tensor.view(depth_tensor.size(0), -1)
        (medians, _) = torch.median(depth_tensor_flat, dim=1)
        ratios = 1 / medians
        (axisangle, translation) = (outputs['axisangle', 0, f_i][:, 0].detach().squeeze(1), outputs['translation', 0, f_i][:, 0].detach().squeeze(1))
        (axisangle, translation) = (-axisangle, -translation) if f_i == -1 else (axisangle, translation)
        translation = translation * ratios.unsqueeze(1)
        with torch.no_grad():
            pose_embedding = torch.cat((axisangle, translation), dim=1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.opt.height, self.opt.width)
            pose_embedding = pose_embedding[:, 5, :, :].unsqueeze(1)
            depth_embedding = outputs['depth', 0, 0].detach()
            depth_embedding = depth_embedding * ratios.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            depth_embedding = 1 / depth_embedding
        if f_i == 1:
            flownet_input = torch.cat((outputs['forward_warp', f_i, 0].detach(), inputs['color', 0, 0], outputs['rigid_flow', f_i, 0].detach(), pose_embedding), dim=1)
            flownet_input_2 = torch.cat((outputs['forward_warp', f_i, 0], inputs['color', 1, 0], outputs['rigid_flow', f_i, 0].detach(), -pose_embedding), dim=1)
        elif f_i == -1:
            flownet_input = torch.cat((outputs['forward_warp', f_i, 0].detach(), inputs['color', 0, 0], outputs['rigid_flow', f_i, 0].detach(), pose_embedding), dim=1)
            flownet_input_2 = torch.cat((outputs['forward_warp', f_i, 0], inputs['color', -1, 0], outputs['rigid_flow', f_i, 0].detach(), -pose_embedding), dim=1)
        self.rigid_flowestimate = self.models['flow_rigid'](flownet_input)
        self.rigid_flowestimate_2 = self.models['flow_rigid'](flownet_input_2)
        for i in range(1):
            outputs['rigid_flow_estimate', i, f_i] = self.rigid_flowestimate[i]
            outputs['rigid_flow_estimate_inv', i, f_i] = self.rigid_flowestimate_2[i]
        forward_pix = outputs['sample', f_i, 0].detach()
        outputs['warped_rigid_flow_estimate', f_i] = F.grid_sample(outputs['rigid_flow_estimate', 0, f_i], forward_pix, align_corners=True)
        self.compute_rigid_error_and_sample(inputs, outputs, f_i)

    def compute_rigid_error_and_sample(self, inputs, outputs, f_i):
        """Compute rigid error and warp sample"""
        rigid_flow_pix = outputs['rigid_flow_estimate', 0, f_i].permute(0, 2, 3, 1)
        sample_pix = self.compute_sample_coordinates(rigid_flow_pix)
        outputs['sample_rigid', f_i, 0] = sample_pix
        rigid_flow_pix_inv = outputs['rigid_flow_estimate_inv', 0, f_i].permute(0, 2, 3, 1)
        sample_pix_inv = self.compute_sample_coordinates(rigid_flow_pix_inv)
        outputs['sample_rigid_inv', f_i, 0] = sample_pix_inv
        raw_rigid_color = F.grid_sample(inputs['color', 0, 0], sample_pix, padding_mode='border', align_corners=True)
        motion_mask = outputs['final_mask']
        if f_i == -1:
            tensor = sample_pix.clone()
            combined_mask_2 = (tensor >= 1) | (tensor <= -1)
            combined_mask_2 = combined_mask_2.any(dim=-1)
            combined_mask_2 = combined_mask_2.float()
            outputs['edge_mask_2'] = combined_mask_2.unsqueeze(1)
            combined_mask_2 = combined_mask_2.unsqueeze(1)
            raw_edge_mask = torch.zeros_like(combined_mask_2)
            raw_edge_mask[..., :10, :] = 1
            raw_edge_mask[..., -10:, :] = 1
            raw_edge_mask[..., :, :10] = 1
            raw_edge_mask[..., :, -10:] = 1
            combine_edge_mask = (combined_mask_2 == 1) | (raw_edge_mask == 1)
            final_edge_mask = (combine_edge_mask == 1) & (motion_mask != 1)
            outputs['color_rigid_estimate', f_i, 0] = torch.where(final_edge_mask == 1, inputs['color', f_i, 0], raw_rigid_color)
        if f_i == 1:
            combined_mask_2 = outputs['edge_mask']
            combined_mask_2 = combined_mask_2.float()
            raw_edge_mask = torch.zeros_like(combined_mask_2)
            raw_edge_mask[..., :10, :] = 1
            raw_edge_mask[..., -10:, :] = 1
            raw_edge_mask[..., :, :10] = 1
            raw_edge_mask[..., :, -10:] = 1
            combine_edge_mask = (combined_mask_2 == 1) | (raw_edge_mask == 1)
            final_edge_mask = (combine_edge_mask == 1) & (motion_mask != 1)
            outputs['color_rigid_estimate', f_i, 0] = torch.where(final_edge_mask == 1, inputs['color', f_i, 0], raw_rigid_color)
        rigid_error = image_similarity(inputs['color', f_i, 0], outputs['color_rigid_estimate', f_i, 0])
        outputs['rigid_error_rigid_estimate', f_i] = L2_norm(rigid_error, dim=1)
        outputs['color_rigid_estimate_inv', f_i, 0] = F.grid_sample(inputs['color', f_i, 0], sample_pix_inv, padding_mode='border', align_corners=True)
        rigid_error = image_similarity(inputs['color', 0, 0], outputs['color_rigid_estimate_inv', f_i, 0])
        outputs['rigid_error_rigid_estimate_inv', f_i] = L2_norm(rigid_error, dim=1)

    def compute_sample_coordinates(self, rigid_flow_pix):
        """Compute sample coordinates for flow estimation"""
        meshgrid = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')
        id_coords = torch.from_numpy(np.stack(meshgrid, axis=0).astype(np.float32)).permute(1, 2, 0).unsqueeze(0).to(self.device)
        sample_pix = rigid_flow_pix + id_coords
        sample_pix[..., 0] /= self.opt.width - 1
        sample_pix[..., 1] /= self.opt.height - 1
        return (sample_pix - 0.5) * 2

    def generate_mask(self, inputs, outputs):
        color_input = inputs['color_aug', 0, 0]
        resflow = outputs['resflow', 0, 1].detach()
        rigid_error = outputs['rigid_error', 1].detach()
        resflow_inv = outputs['resflow', 0, -1].detach()
        rigid_error_inv = outputs['rigid_error', -1].detach()
        combined_resflow = torch.abs(resflow) + torch.abs(resflow_inv)
        combined_resflow = torch.sum(combined_resflow, dim=1)
        combined_resflow = combined_resflow.unsqueeze(1)
        mean_value = combined_resflow.mean(dim=(2, 3), keepdim=True)
        mean_value = mean_value
        max_value = combined_resflow.view(combined_resflow.size(0), -1).max(dim=1, keepdim=True)[0]
        max_value = max_value.clamp(min=1e-06)
        max_value = max_value.view(combined_resflow.size(0), 1, 1, 1)
        outputs['mask'] = (combined_resflow > mean_value).float()
        outputs['mask_filter'] = (outputs['mask'] > 0.8).int()

    def compute_ground_mask(self, pretrain_dpepth, inv_K, K, height, width, threshold=-0.005):
        cam_points = self.cloud_to_mask(pretrain_dpepth, inv_K, K, threshold=threshold, T=None)
        (plane_mask, raw_plane_param, plane_param) = self.gplane(cam_points[:, :3].reshape(-1, 3, height, width))
        return (plane_mask, raw_plane_param, plane_param)

    def process_mask_labels(self, outputs, adjusted_y_distances, inputs, y_min_per_batch):
        """
            Processes the mask labels by filtering connected components, applying y-distance thresholds,
            and refining the mask to remove sky objects and high-altitude ground objects.

            Parameters:
            outputs (dict): The dictionary holding output tensors.
            adjusted_y_distances (Tensor): Adjusted y-axis distances [b, 1, h, w].
            inputs (dict): The input tensor dictionary.
            y_min_per_batch (Tensor): The minimum y-coordinates per batch.
            opt (Namespace): Options including batch size.

            Returns:
            dict: Updated outputs dictionary with processed mask labels.
            """
        device = outputs['mask'].device
        adjusted_y_distances = adjusted_y_distances.squeeze(1)
        (mask_connected_label_vis, mask_connected_label, mask_num_features) = ([], [], [])
        min_area_threshold = 250
        for i in range(self.opt.batch_size):
            image = (outputs['mask'] > 0.7)[i].squeeze(0).cpu()
            image_np = image.numpy()
            (label_image, num_features) = ndimage.label(image_np)
            label_areas = ndimage.sum(image_np, label_image, range(num_features + 1))
            remove_small = label_areas < min_area_threshold
            remove_small_mask = remove_small[label_image]
            label_image[remove_small_mask] = 0
            (label_image, num_features) = ndimage.label(label_image > 0)
            cmap = plt.get_cmap('tab20b')
            mask_label = cmap(label_image / max(num_features, 1))
            mask_label = torch.from_numpy(mask_label).permute(2, 0, 1).unsqueeze(0)
            mask_connected_label_vis.append(mask_label.to(device))
            label_image_tensor = torch.from_numpy(label_image).unsqueeze(0).to(device)
            mask_connected_label.append(label_image_tensor)
            num_features_tensor = torch.tensor(num_features).unsqueeze(0).unsqueeze(0).to(device)
            mask_num_features.append(num_features_tensor)
        mask_connected_label_vis = torch.cat(mask_connected_label_vis, dim=0)
        mask_connected_label = torch.cat(mask_connected_label, dim=0)
        mask_num_features = torch.cat(mask_num_features, dim=0)
        outputs['mask_label'] = mask_connected_label_vis
        mask_check = torch.zeros_like(mask_connected_label).to(device)
        mask_check[mask_connected_label > 0] = 1
        y_coords = torch.arange(mask_check.size(1), device=device).view(1, -1, 1).expand_as(mask_check)
        y_min_per_batch = y_min_per_batch.unsqueeze(1).to(device)
        mask_check[y_coords < y_min_per_batch] = 0
        mask_connected_label_vis_2 = mask_connected_label_vis * mask_check.unsqueeze(1)
        outputs['mask_label_2'] = mask_connected_label_vis_2
        mask_connected_label_wosky = mask_connected_label * mask_check
        mask_connected_label_wosky_flat = mask_connected_label_wosky.view(mask_connected_label_wosky.size(0), -1)
        mask_connected_label_wosky_filtered = torch.where(mask_connected_label_wosky_flat != 0, mask_connected_label_wosky_flat, torch.tensor(-1, device=device))
        unique_values = [torch.unique(mask_connected_label_wosky_filtered[i]) for i in range(mask_check.size(0))]
        counts = torch.tensor([len(u) for u in unique_values])
        max_len = counts.max()
        padded_unique_values = torch.full((mask_connected_label_wosky.size(0), max_len), -1).to(device)
        for (i, values) in enumerate(unique_values):
            padded_unique_values[i, :counts[i]] = values
        padded_unique_values_expanded = padded_unique_values.unsqueeze(1).unsqueeze(2)
        comparison = torch.eq(mask_connected_label.unsqueeze(-1), padded_unique_values_expanded)
        mask_wosky = comparison.any(dim=-1).int()
        mask_connected_label_vis_3 = mask_wosky.unsqueeze(1) * inputs['color', 0, 0]
        outputs['mask_label_3'] = mask_connected_label_vis_3
        outputs['mask_label_3_in1'] = mask_wosky.unsqueeze(1) * inputs['color', 1, 0]
        outputs['mask_label_3_inv1'] = mask_wosky.unsqueeze(1) * inputs['color', -1, 0]
        wo_ground_y_dist = adjusted_y_distances * (1 - outputs['ground_mask']).squeeze(1)
        y_dist_special = comparison * wo_ground_y_dist.unsqueeze(-1)
        threshold = 0.25
        y_dist_mean = y_dist_special.mean(dim=(1, 2))
        non_zero_mask = y_dist_special != 0
        non_zero_sum = torch.sum(y_dist_special * non_zero_mask, dim=(1, 2))
        non_zero_count = torch.sum(non_zero_mask, dim=(1, 2))
        if self.median_depth_tracker < 0.4:
            (self.t_max, self.t_min) = (1, 0.1)
        elif self.median_depth_tracker >= 0.4:
            (self.t_max, self.t_min) = (2.5 * self.median_depth_tracker * 60 / 28, 0.3 * self.median_depth_tracker * 60 / 28)
        y_dist_mean_c = non_zero_sum / non_zero_count.clamp(min=0.0001)
        y_dis_mean_mask = (y_dist_mean_c < self.t_max) & (y_dist_mean_c > self.t_min)
        y_dis_mean_mask = comparison * y_dis_mean_mask.unsqueeze(1).unsqueeze(2)
        y_dis_mean_mask = y_dis_mean_mask.any(dim=-1).int()
        outputs['mask_label_4'] = y_dis_mean_mask.unsqueeze(1)
        outputs['mask_label_embedding'] = mask_wosky.unsqueeze(1) - outputs['mask_label_4']
        outputs['mask_wosky'] = mask_wosky.unsqueeze(1)
        outputs['final_mask'] = outputs['mask_label_4'] * (1 - outputs['ground_mask'])
        y_coords = torch.arange(self.opt.height, device=y_min_per_batch.device).unsqueeze(0).unsqueeze(2)
        ground_adject_mask = y_coords >= y_min_per_batch.unsqueeze(1)
        ground_adject_mask = ground_adject_mask.float()
        ground_adject_mask_f = ground_adject_mask.expand(self.opt.batch_size, 1, self.opt.height, self.opt.width)
        outputs['ground_adject_mask'] = ground_adject_mask_f - outputs['ground_mask']
        return outputs

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
            """
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            reprojection_losses_ref = []
            multi_reprojection_losses = []
            reprojection_losses_flow = []
            multi_reprojection_losses_flow = []
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            disp = outputs['disp', scale]
            disp_ref = outputs['disp_ref', scale]
            color = inputs['color', 0, scale]
            target = inputs['color', 0, source_scale]
            target_ref = inputs['color', 0, source_scale]
            target_flow = inputs['color', 0, source_scale]
            edge_mask = outputs['edge_mask'].detach()
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs['color', frame_id, scale]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                pred_flow = outputs['color_full', frame_id, 0]
                if frame_id == 1:
                    target_edge_mask = edge_mask.repeat(1, 3, 1, 1)
                    target_edge = target.clone()
                    target_edge[target_edge_mask] = 0
                    pred_flow[target_edge_mask] = 0
                    reprojection_losses_flow.append(self.compute_reprojection_loss(pred_flow, target_edge))
                else:
                    reprojection_losses_flow.append(self.compute_reprojection_loss(pred_flow, target))
                pred_ref = outputs['color_ref', frame_id, scale]
                reprojection_losses_ref.append(self.compute_reprojection_loss(pred_ref, target_ref))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            reprojection_losses_flow = torch.cat(reprojection_losses_flow, 1)
            reprojection_losses_ref = torch.cat(reprojection_losses_ref, 1)
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs['color', frame_id, source_scale]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
                reprojection_loss_ref = reprojection_losses_ref
            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 1e-05
                identity_reprojection_loss_ref = identity_reprojection_loss.clone()
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                (to_optimise, idxs) = torch.min(combined, dim=1, keepdim=True)
            if not self.opt.disable_automasking:
                outputs['identity_selection/{}'.format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
                static_image_check = torch.sum(outputs['identity_selection/{}'.format(scale)].detach(), dim=(1, 2, 3))
                threshold = self.opt.height * self.opt.width * (2 / 3)
                outputs['static_image_mask'] = static_image_check < threshold
                outputs['static_image_mask'] = outputs['static_image_mask'].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            mask_final = outputs['final_mask']
            ground_mask = outputs['ground_mask']
            ground_mask = ground_mask * (1 - outputs['mask_label_4'])
            (loss_dynamic_ref_cat, loss_dynamic_ref) = self.compute_dynamic_region_loss(inputs, outputs)
            combined_ref = torch.cat((identity_reprojection_loss_ref, reprojection_loss_ref), dim=1)
            if self.epoch >= self.opt.train_cat_dynamic_epoch:
                combined_ref_cat = torch.cat((combined_ref, loss_dynamic_ref_cat), dim=1)
                (to_optimise_ref_cat, idx_ref_1) = torch.min(combined_ref_cat, dim=1, keepdim=True)
                to_optimise_ref_cat = to_optimise_ref_cat * (1 - mask_final)
            (to_optimise_ref_raw, idxs_ref) = torch.min(combined_ref, dim=1, keepdim=True)
            to_optimise_ref = to_optimise_ref_raw * (1 - mask_final)
            to_optimise_ref_ground = to_optimise_ref_raw * ground_mask
            if self.epoch >= self.opt.train_cat_dynamic_epoch:
                loss_depth_ref = 0.9 * to_optimise_ref_cat.mean() + 0.1 * to_optimise_ref.mean()
            else:
                loss_depth_ref = to_optimise_ref_raw.mean()
            losses['depth_ref_loss'] = loss_depth_ref
            if self.epoch >= self.opt.train_ref_epoch:
                mean_disp_ref = disp_ref.mean(2, True).mean(3, True)
                norm_disp_ref = disp_ref / (mean_disp_ref + 1e-07)
                smooth_loss_ref = get_smooth_loss(norm_disp_ref, color)
                loss += self.opt.disparity_smoothness * smooth_loss_ref / 2 ** scale
                _, final_depth = disp_to_depth(disp_ref, self.opt.min_depth, self.opt.max_depth)
                dynamic = outputs['final_mask'].detach().bool()
                ground = outputs['ground_mask'].detach().bool() & ~dynamic
                uncertain = outputs['mask_label_embedding'].detach().bool() & ~ground & ~dynamic
                gac = self.gac(final_depth, color, ground, uncertain, dynamic)
                loss += self.gac_weight * gac['loss']
                losses['gac/loss'] = gac['loss']
                losses['gac/data'] = gac['data_loss']
                losses['gac/smoothness'] = gac['smoothness_loss']
                losses['gac/source_anchors'] = gac['source_count'].sum()
                losses['gac/dynamic_anchors'] = gac['dynamic_count'].sum()
                outputs['gac_reference'] = gac['reference']
                outputs['gac_delta'] = gac['delta']
                loss += loss_depth_ref
            '可能需要优化，从头就不使用整体场景可能会让模型出错？'
            loss_depth = (to_optimise * (1 - outputs['mask_label_4'])).mean()
            if self.epoch <= self.opt.freeze_teacher_epoch:
                loss += loss_depth
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-07)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / 2 ** scale
            losses['depth_loss'] = loss_depth
            loss += reprojection_losses_flow.mean()
            prob_target = torch.zeros(self.opt.batch_size, 1, self.opt.height, self.opt.width).to(self.device)
            edge_mask = edge_mask.repeat(1, 2, 1, 1).detach()
            rigid_error = outputs['rigid_error', -1] + outputs['rigid_error', 1]
            rigid_error = rigid_error / 2
            rigid_error = rigid_error.squeeze(1)
            static_thresh = rigid_error.mean(dim=(1, 2), keepdim=True)
            static_region = (rigid_error < static_thresh).unsqueeze(1)
            flow_rigid_loss = 0
            for frame_id in self.opt.frame_ids[1:]:
                resflow = outputs['resflow', 0, frame_id]
                if frame_id == -1:
                    flow_smooth_loss = get_smooth_loss(resflow, color)
                    flow_smooth_loss = get_smooth_loss(resflow, norm_disp.detach()) * 5
                resflow = torch.abs(resflow)
                resflow = torch.sum(resflow, 1).unsqueeze(1)
                flow_rigid_loss += resflow[static_region].mean() if static_region.any() else resflow.sum() * 0
            loss += flow_rigid_loss * 0.5
            loss += flow_smooth_loss
            losses['loss_flowsmooth'] = flow_smooth_loss
            loss_rigid_flow_estimate = 0
            reprojection_loss_rigid_estimate = 0
            for frame_id in self.opt.frame_ids[1:]:
                (l_rigid_flow_estimate, reprojection_l_rigid_estimate) = self.compute_rigid_estimate_loss(inputs, outputs, frame_id)
                loss_rigid_flow_estimate += l_rigid_flow_estimate * 3
                reprojection_loss_rigid_estimate += reprojection_l_rigid_estimate
            if self.epoch >= self.opt.train_rigid_flow_epoch:
                loss += loss_rigid_flow_estimate
                loss += reprojection_loss_rigid_estimate
                losses['loss_rigid_flow_estimate '] = loss_rigid_flow_estimate
                losses['reprojection_loss_rigid_estimate '] = reprojection_loss_rigid_estimate
            if self.epoch >= self.opt.train_only_dynamic_epoch:
                loss += loss_dynamic_ref
                losses['loss_dynamic_ref'] = loss_dynamic_ref
            (mask_sparsity_loss, mask_sparsity_2_loss, mask_generate_loss, mask_smooth_loss, mask_depth_smooth_loss) = (0, 0, 0, 0, 0)
            loss += mask_sparsity_loss * 0.1 + mask_sparsity_2_loss * 0 + mask_generate_loss + self.opt.disparity_smoothness * mask_smooth_loss + self.opt.disparity_smoothness * mask_depth_smooth_loss
            loss += 0.1 * (get_ysmooth_loss(norm_disp, color) * (outputs['mask_label_4'] * (1 - outputs['ground_mask']))[:, :, :-1, :]).mean()
            total_loss += loss
            losses['loss/{}'.format(scale)] = loss
        total_loss /= self.num_scales
        losses['loss'] = total_loss
        return losses

    def compute_rigid_estimate_loss(self, inputs, outputs, frame_id):
        loss_rigid_flow_estimate = 0
        reprojection_loss_rigid_estimate = 0
        mask_mix = (1 - outputs['mask_label_4']) * outputs['static_image_mask']
        mask_mix_all = (1 - outputs['mask_label_4']) * outputs['static_image_mask'] * ~outputs['edge_mask']
        if frame_id == 1:
            rigid_flow = outputs['rigid_flow', frame_id, 0].detach()
            loss_rigid_flow_estimate += self.compute_distiil_loss(outputs['warped_rigid_flow_estimate', frame_id] * mask_mix.detach(), -rigid_flow * mask_mix.detach())
            target_rigid = inputs['color', 1, 0]
        elif frame_id == -1:
            rigid_flow = outputs['rigid_flow', frame_id, 0].detach()
            loss_rigid_flow_estimate += self.compute_distiil_loss(outputs['warped_rigid_flow_estimate', frame_id] * mask_mix_all.detach(), -rigid_flow * mask_mix_all.detach())
            target_rigid = inputs['color', -1, 0]
        reprojection_loss_rigid = self.compute_reprojection_loss(outputs['color_rigid_estimate', frame_id, 0], target_rigid)
        if frame_id == 1:
            reprojection_loss_rigid = reprojection_loss_rigid * mask_mix
        elif frame_id == -1:
            reprojection_loss_rigid = reprojection_loss_rigid * mask_mix_all
        reprojection_loss_rigid_estimate += reprojection_loss_rigid.mean()
        return (loss_rigid_flow_estimate, reprojection_loss_rigid_estimate)

    def compute_dynamic_region_loss(self, inputs, outputs):
        target = inputs['color', 0, 0]
        loss_dynamic_ref_1 = self.compute_reprojection_loss(outputs['color_ref_maskregion', 1, 0], target)
        loss_dynamic_ref_2 = self.compute_reprojection_loss(outputs['color_ref_maskregion', -1, 0], target)
        loss_dynamic_ref_ = torch.cat((loss_dynamic_ref_1, loss_dynamic_ref_2), dim=1)
        (to_optimise_ref, _) = torch.min(loss_dynamic_ref_, dim=1, keepdim=True)
        loss_dynamic_ref = to_optimise_ref * outputs['mask_label_4'] * outputs['static_image_mask']
        loss_dynamic_ref = torch.mean(loss_dynamic_ref)
        return (loss_dynamic_ref_, loss_dynamic_ref)
