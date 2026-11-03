# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from kornia.geometry.depth import depth_to_3d
# from kornia.geometry.depth import depth_to_3d_v2
import warnings
# import open3d as o3d

import matplotlib.pyplot as plt

# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="Since kornia 0.7.0 the `depth_to_3d` is deprecated*")


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        # R1 = rot_from_axisangle(-axisangle)
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBN(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBN, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nonlin(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, res=None):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        # if res == None:
        #     Z_res = torch.zeros(cam_points.shape[0], cam_points.shape[2])
        #     cam_points[:, 2, :] = cam_points[:, 2, :] + Z_res.cuda()
        if res != None:
            Z_res = res[:, 2, :, :]
            Z_res = Z_res.view(self.batch_size, -1)
            # print(cam_points[:, 2, :].size())
            cam_points[:, 2, :] = cam_points[:, 2, :] + Z_res

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        # if res == None:
        #     xy_res = torch.zeros(cam_points.shape[0], 2, pix_coords.shape[2], pix_coords.shape[3])
        #     xy_res = xy_res.permute(0, 2, 3, 1)
        #     pix_coords[:, :2, :, :] = pix_coords[:, :2, :, :] + xy_res.cuda()
        if res != None:
            xy_res = res[:, :2, :, :]
            xy_res = xy_res.permute(0, 2, 3, 1)
            pix_coords[:, :, :, :2] = pix_coords[:, :, :, :2] + xy_res
            
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_ysmooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_y

def get_ysmooth_loss_2(disp, img):
    """Edge-aware smoothness loss with depth-order constraint.
       Only enforces smoothness when upper pixel has LARGER disparity (SMALLER depth) 
       than lower pixel. This prevents gradients from propagating from large-depth 
       (small disparity) regions to small-depth regions.
    """
    # 获取相邻像素视差差值的绝对值 [B, C, H-1, W]
    disp_upper = disp[:, :, :-1, :]  # 上方像素视差
    disp_lower = disp[:, :, 1:, :].detach()   # 下方像素视差
    grad_disp_y = torch.abs(disp_upper - disp_lower)
    
    # 创建单向约束掩码：仅当上方视差 > 下方视差时计算损失
    mask = (disp_upper > disp_lower).float()  # [B, C, H-1, W]
    grad_disp_y = grad_disp_y * mask  # 深度大的区域不会影响深度小的区域

    # 边缘感知权重（仅在有效区域计算）
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_disp_y *= torch.exp(-grad_img_y)  # 图像边缘处降低权重

    return grad_disp_y.mean()  # 返回平均损失

def get_gapdepth(disp, mask):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    top_k = 5
    # print(disp.shape)
    disp = disp.squeeze(1)
    mask = mask.squeeze(1)

    # 初始化一个新的图像，防止直接修改原图
    new_disp = disp.clone()

    # 获取 batch size 和图像的高度和宽度
    B, H, W = disp.shape

    for b in range(B):
        # 获取当前 batch 图像的 mask 和 image
        curr_mask = mask[b]
        curr_disp = disp[b]
        
        # 对每列进行处理
        for col in range(W):
            # 获取当前列的掩码为1的地面区域
            valid_pixels = curr_disp[:, col][curr_mask[:, col] == 1]

            # 如果有足够的地面区域，计算均值并替换掩码为0的区域
            if valid_pixels.size(0) >= top_k:
                # 取前 top_k 个地面区域的均值
                col_mean = valid_pixels[:top_k].float().mean().item()

                # 将当前列掩码为0的区域替换为均值
                new_disp[b, curr_mask[:, col] == 0, col] = col_mean
            else:
                # 如果没有足够的地面区域，替换为0
                new_disp[b, curr_mask[:, col] == 0, col] = 0
    
    # print(new_disp)
    return new_disp


def get_gsmooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_y

def photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, intrinsics, poses, poses_inv):

    diff_img_list = []
    diff_color_list = []
    diff_depth_list = []
    valid_mask_list = []
    weight_mask_list = []

    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1, weight_mask_tmp1 = compute_pairwise_loss(
            tgt_img, ref_img, tgt_depth,
            ref_depth, pose, intrinsics
        )
        diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2, weight_mask_tmp2 = compute_pairwise_loss(
            ref_img, tgt_img, ref_depth,
            tgt_depth, pose_inv, intrinsics
        )
        diff_img_list += [diff_img_tmp1, diff_img_tmp2]
        diff_color_list += [diff_color_tmp1, diff_color_tmp2]
        diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
        valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]
        weight_mask_list += [weight_mask_tmp1, weight_mask_tmp2]

    diff_img = torch.cat(diff_img_list, dim=1)
    diff_color = torch.cat(diff_color_list, dim=1)
    diff_depth = torch.cat(diff_depth_list, dim=1)
    valid_mask = torch.cat(valid_mask_list, dim=1)

    # using photo loss to select best match in multiple views

    indices = torch.argmin(diff_color, dim=1, keepdim=True)

    diff_img = torch.gather(diff_img, 1, indices)
    diff_depth = torch.gather(diff_depth, 1, indices)
    valid_mask = torch.gather(valid_mask, 1, indices)

    photo_loss = mean_on_mask(diff_img, valid_mask)
    geometry_loss = mean_on_mask(diff_depth, valid_mask)

    # if hparams.model_version == 'v3':
    #     # get dynamic mask for tgt image
    #     dynamic_mask = []
    #     for i in range(0, len(diff_depth_list), 2):
    #         tmp = diff_depth_list[i]
    #         tmp[valid_mask_list[i] < 1] = 0
    #         dynamic_mask += [1-tmp]

    #     dynamic_mask = torch.cat(dynamic_mask, dim=1).mean(dim=1, keepdim=True)

    #     return photo_loss, geometry_loss, dynamic_mask
    # else:
    return photo_loss, geometry_loss, valid_mask, weight_mask_list


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic):

    ref_img_warped, projected_depth, computed_depth = inverse_warp(
        ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode='zeros')

    diff_depth = (computed_depth-projected_depth).abs() / \
        (computed_depth+projected_depth)

    # masking zero values
    valid_mask_ref = (ref_img_warped.abs().mean(
        dim=1, keepdim=True) > 1e-3).float()
    valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask = valid_mask_tgt * valid_mask_ref

    diff_color = (tgt_img-ref_img_warped).abs().mean(dim=1, keepdim=True)
    # if not hparams.no_auto_mask:
    identity_warp_err = (tgt_img-ref_img).abs().mean(dim=1, keepdim=True)
    auto_mask = (diff_color < identity_warp_err).float()
    valid_mask = auto_mask * valid_mask

    diff_img = (tgt_img-ref_img_warped).abs().clamp(0, 1)
    # if not hparams.no_ssim:
    ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
    diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    diff_img = torch.mean(diff_img, dim=1, keepdim=True)

    # reduce photometric loss weight for dynamic regions
    # if not hparams.no_dynamic_mask:
    weight_mask = (1-diff_depth).detach()
    diff_img = diff_img * weight_mask

    return diff_img, diff_color, diff_depth, valid_mask, weight_mask

# compute mean value on a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value

def inverse_warp(img, depth, ref_depth, T, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    B, _, H, W = img.size()
    
    # T = pose_vec2mat(pose)  # [B,3,4]
    P = torch.matmul(intrinsics, T)[:, :3, :]

    world_points = depth_to_3d_v2(depth, intrinsics[0, :3, :3]) # B 3 H W
    world_points = world_points.squeeze(1)
    world_points = world_points.permute(0,3,1,2)
    # print(world_points[2,0,150,500])
    # print(world_points[2,1,150,500])
    # print(world_points[2,2,150,500])
    world_points = torch.cat([world_points, torch.ones(B,1,H,W).type_as(img)], 1)
    cam_points = torch.matmul(P, world_points.view(B, 4, -1))

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    projected_img = F.grid_sample(img, pix_coords, padding_mode=padding_mode, align_corners=False)
    projected_depth = F.grid_sample(ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img, projected_depth, computed_depth


def image_similarity(x, y, alpha=0.85):
    DSSIM = SSIM()
    return alpha * DSSIM(x,y) + (1-alpha) * torch.abs(x-y)

def image_similarity_else(x, y, alpha=0.85):
    DSSIM = SSIM_kernalE()
    return alpha * DSSIM(x,y) + (1-alpha) * torch.abs(x-y)

class SSIM_kernalE(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_kernalE, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(7, 1)
        self.mu_y_pool   = nn.AvgPool2d(7, 1)
        self.sig_x_pool  = nn.AvgPool2d(7, 1)
        self.sig_y_pool  = nn.AvgPool2d(7, 1)
        self.sig_xy_pool = nn.AvgPool2d(7, 1)

        self.refl = nn.ReflectionPad2d(3)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
    return l2_norm

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

compute_ssim_loss = SSIM().to(device)
def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel



def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img

UNKNOWN_FLOW_THRESH = 1e7
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def save_vis_flow_tofile(flow, output_path):
    vis_flow = flow_to_image(flow)
    from PIL import Image
    img = Image.fromarray(vis_flow)
    img.save(output_path)

def flow_tensor_to_image(flow):
    """Used for tensorboard visualization"""
    flow = flow.permute(1, 2, 0)  # [H, W, 2]
    flow = flow.detach().cpu().numpy()
    flow = flow_to_image(flow)  # [H, W, 3]
    flow = np.transpose(flow, (2, 0, 1))  # [3, H, W]

    return flow


class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(ResnetEncoderMatching, self).__init__()

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)

        # self.compute_depth_bins(raw_depth, min_depth_bin, max_depth_bin)
        self.compute_depth_bins(raw_depth)

        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                                        kernel_size=1, stride=1, padding=0),
                                              nn.ReLU(inplace=True)
                                              )

        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[1],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )

    def compute_percentile(self, tensor, percentile):
        k = int(percentile * tensor.size(1))
        return tensor.kthvalue(k, dim=1)[0]

    def compute_depth_bins(self, raw_depth):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        flat_raw_depth = raw_depth.view(raw_depth.size(0), -1)

        min_depth, max_depth = self.compute_percentile(flat_raw_depth, 0.1), self.compute_percentile(flat_raw_depth, 0.45)   

        # min_depth_bin = min_depth
        # max_depth_bin = min_depth + (max_depth - min_depth) 

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth, max_depth, self.num_depth_bins)
        else:
            raise NotImplementedError
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence

        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            world_points = self.backprojector(self.warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                    self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(
                    1) * edge_mask

                # integrate into cost volume
                cost_volume = cost_volume + diffs
                counts = counts + (diffs > 0).float()
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                    cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks

    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward(self, current_image, lookup_images, poses, K, invK,
                min_depth_bin=None, max_depth_bin=None
                ):

        # feature extraction
        self.features = self.feature_extraction(current_image, return_all_feats=True)
        current_feats = self.features[-1]

        # feature extraction on lookup images - disable gradients to save memory
        with torch.no_grad():
            if self.adaptive_bins:
                self.compute_depth_bins(min_depth_bin, max_depth_bin)

            batch_size, num_frames, chns, height, width = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
            lookup_feats = self.feature_extraction(lookup_images,
                                                   return_all_feats=False)
            _, chns, height, width = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

            # warp features to find cost volume
            cost_volume, missing_mask = \
                self.match_features(current_feats, lookup_feats, poses, K, invK)
            confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                           (1 - missing_mask.detach()))

        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)
        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], cost_volume], 1))

        self.features.append(self.layer2(post_matching_feats))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features, lowest_cost, confidence_mask

class D_to_cloud_mask(nn.Module):
    """Layer to transform depth into point cloud
    """
    def __init__(self, batch_size, height, width):
        super(D_to_cloud_mask, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, inv_K, K, threshold=0.02, T=None):
        y_threshold = threshold
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points

        # Convert cam_points to homogeneous coordinates by adding a row of ones
        
        # cam_points = cam_points.permute(0, 2, 1)  # B, L, 3

        # # 提取 z 轴坐标
        # y = cam_points[:, :, 1]

        # # 生成地面掩码 (标记 z 轴小于阈值的点)
        # ground_mask = (y > y_threshold).float()  # B, L

        # # num_ones = torch.sum(ground_mask)
        # # print(num_ones.item())

        # # 将三维点投影回二维平面 (使用 K 矩阵)
        # proj_points = torch.matmul(K[:, :3, :3], cam_points.permute(0, 2, 1))  # B, 3, L
        # proj_points = proj_points[:, :2, :] / proj_points[:, 2:3, :]  # B, 2, L (除以 z 进行归一化)

        # # 生成与原始图像尺寸相同的掩码图
        # ground_mask_2d = ground_mask.view(self.batch_size, self.height, self.width)

        return cam_points

class GroundPlane(nn.Module):
    def __init__(self, num_points_per_it=5, max_it=25, tol=0.005, g_prior=0.4, vertical_axis=1):
        super(GroundPlane, self).__init__()
        self.num_points_per_it = num_points_per_it
        self.max_it = max_it
        self.tol = tol
        self.g_prior = g_prior
        self.y_prior_threshold = 0.01  # 设定 y 坐标的阈值
        self.vertical_axis = vertical_axis
    
    def forward(self, points):
        """ estimates plane parameters and returns each point's distance to it and a ground mask
        :param points     (B, 3, H, W)
        :ret distance    (B, 1, H, W)
        :ret plane_param (B, 3)
        :ret mask        (B, 1, H, W)
        """
        
        B, _, H, W = points.shape 
        ground_points = points[:, :, - int(self.g_prior*H):, :]
        ground_points_inp = ground_points.reshape(B, 3, -1).permute(0,2,1) # (B, N, 3)

        plane_param = self.estimate_ground_plane(ground_points_inp)
        # print(plane_param.size())

        # normal_avg = torch.nn.functional.normalize(plane_param, dim=1)
        # normal_avg = normal_avg.squeeze(2)
        # normal_avg = - normal_avg
        # print(normal_avg)
        # print(plane_param)
        
        all_points = points.reshape(B, 3, H*W).permute(0,2,1)  # (B, H*W, 3)
        dist2plane = self.dist_from_plane(all_points, plane_param).permute(0,2,1).reshape(B,1,H,W)

        # Create a mask based on distance to the plane
        distance_mask = (torch.abs(dist2plane) < self.tol).float()

        # Create a mask based on y coordinate (assuming vertical_axis is the y axis)
        y_coords = all_points[:, :, self.vertical_axis]  # (B, H*W)
        y_mask = (torch.abs(y_coords) > -0.00).float().reshape(B, 1, H, W)

        # Calculate normals and create a mask based on normal deviation from [0, 1, 0]
        # masked_points = points * y_mask

        # # Calculate normals using masked points and create a mask based on normal deviation from [0, 1, 0]
        # normals = self.calculate_normals(masked_points)
        normals = self.calculate_normals(points)    # b 3 h w
        target_normal = torch.tensor([0.0, 1.0, 0.0]).to(points.device).view(1, 3, 1, 1)
        normal_mask = F.cosine_similarity(normals, target_normal, dim=1).abs() > 0.96  # threshold can be adjusted
        normal_mask = normal_mask.unsqueeze(1)

        # Combine distance-based mask, y coordinate mask, and normal mask
        final_mask = distance_mask * y_mask * normal_mask.float()  # b, 1, h, w
        count_ones = torch.sum(final_mask[0] == 1)

        ones_count = torch.sum(final_mask == 1, dim=[1, 2, 3])

        small_mask_indices = torch.where(ones_count < 20000)[0]
        # print(small_mask_indices)

        if small_mask_indices.numel() > 0:
            other_mask = torch.zeros_like(final_mask)
            other_mask[:,:,int(0.65*H):,:] = 1
            other_mask = normal_mask * other_mask * y_mask

            final_mask[small_mask_indices] = other_mask[small_mask_indices]

            count_ones = torch.sum(normal_mask[small_mask_indices] == 1)
            # print(count_ones, 'y_mask')

        ground_points = points * y_mask
        ground_points = ground_points.reshape(B, 3, H*W).permute(0,2,1)

        normal_vector = self.estimate_plane_normal_batch(ground_points)

        normal_avg = torch.nn.functional.normalize(normal_vector, dim=1)

        normal_avg =  torch.abs(normal_avg)
        # normal_avg_indices = torch.nonzero(final_mask.squeeze(1), as_tuple=False)   # 148778, 3
        
        b, _, h, w = final_mask.shape
        # normal_avg = []

        return final_mask, plane_param, normal_avg
    
    def calculate_normals(self, points):
        """Calculate normals using central difference method"""
        B, _, H, W = points.shape

        # Calculate x and y gradients
        dzdx = (points[:, :, :, 2:] - points[:, :, :, :-2])  # x-derivatives
        dzdy = (points[:, :, 2:, :] - points[:, :, :-2, :])  # y-derivatives

        # Ensure the gradients have the same size by cropping the middle part of the original image
        dzdx = dzdx[:, :, 1:-1, :]  # Crop along the y dimension to match dzdy
        dzdy = dzdy[:, :, :, 1:-1]  # Crop along the x dimension to match dzdx

        # Cross product to get normal
        normals = torch.cross(dzdx, dzdy, dim=1)
        normals = F.normalize(normals, dim=1)  # Normalize the normal vectors

        # Add padding to match original shape
        normals = F.pad(normals, (1, 1, 1, 1), mode='replicate')

        return normals
    
    def dist_from_plane(self, points, param):
        """ get vertical distance of each point from plane specified by param
        :param points   (B, 3) or (SB, B, 3)
        :param param    (3, 1) or (SB, 3, 1)
        :ret distance   (B, 1) or (SB, B, 1)
        """
        
        A, B = self.get_AB(points)
        return A @ param - B

    def estimate_ground_plane(self, points):
        """
        :param points           (B, N, 3)            
        :ret plane parameter    (B, 3) (B)
        """
        
        B, N, _ = points.shape
        T = self.num_points_per_it * self.max_it

        rand_points = []

        for b in range(B):
            rand_ind = np.random.choice(np.arange(N), T, replace=True)
            rand_points.append(points[b][rand_ind])
        rand_points = torch.stack(rand_points)  # (B, T, 3)
        
        ws = self.calc_param(rand_points).reshape(-1, 3, 1)    # (B*self.max_it, 3, 1)
        ps = points.repeat_interleave(self.max_it, dim=0)                  # (B*self.max_it, N, 3)
        
        abs_dist = torch.abs(self.dist_from_plane(ps, ws)).reshape(B, self.max_it, N)

        param_fit = (abs_dist < self.tol).float().mean(2)

        best_fit = param_fit.argmax(1)

        best_w = ws.reshape(B, self.max_it, 3, 1)[np.arange(B), best_fit]

        return best_w
        
    def calc_param(self, points):
        """
        :param points           (B, self.max_it, self.num_points_per_it, 3)            
        :ret plane parameter    (B, self.max_it, 3)
        """
            
        batched_points = points.reshape(-1, self.num_points_per_it, 3)
        
        A, B = self.get_AB(batched_points)
        At = A.transpose(2,1) # batched transpose
        
        w = (torch.inverse(At @ A + 1e-6 * torch.eye(3, device=A.device, dtype=A.dtype)) @ At @ B).reshape(points.size(0), self.max_it, 3, 1)

        return w
    
    def get_AB(self, points):
        """ get mat A and B associated with points
        :param points   (B, 3) or (SB, B, 3)
        :ret A    (B, 3) or (SB, B, 3)
        :ret B    (B, 1) or (SB, B, 1)
        """
        B = points[..., self.vertical_axis:self.vertical_axis+1]
        A = torch.cat([points[..., i:i+1] for i in range(3) if i != self.vertical_axis] + [torch.ones_like(B)], -1)
        return A, B
    
    def estimate_plane_normal_batch(self, points_batch):
        """
        估计每个批次中点云的法向量
        :param points_batch: (B, N, 3) 代表每个批次的点云数据
        :return: 法向量 (B, 3)
        """
        # 计算质心 (B, 1, 3)，保持批次维度
        centroids = torch.mean(points_batch, dim=1, keepdim=True)  # (B, 1, 3)

        # 中心化点集 (B, N, 3)
        centered_points = points_batch - centroids

        # 计算协方差矩阵 (B, 3, 3)
        cov_matrices = torch.matmul(centered_points.transpose(1, 2), centered_points)  # (B, 3, 3)

        # 使用SVD分解协方差矩阵，取出最小的奇异向量
        # torch.svd() returns U, S, V where V is the matrix we want
        _, _, vh = torch.svd(cov_matrices)

        # 最小奇异值对应的向量是法向量
        normals = vh[:, :, -1]  # 取出最后一列 (B, 3)

        return normals

def depth_to_disp(depth, min_depth, max_depth):
    """Inverse of the previous function
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp

def torch_and(*args):
    """ Accept a list of arugments of torch.Tensor of the same shape, compute element-wise and operation for all of them
        Output tensor has the same shape as the input tensors
    """
    out = args[0]
    for a in args:
        assert out.size() == a.size(), "Sizes must match: [{}]".format(', '.join([str(x.size()) for x in args]))
        out = torch.logical_and(out, a)
    return out

def calculate_y_min_per_batch(plane_mask, height):
    y_min_per_batch = []
    for i in range(plane_mask.size(0)):
        mask = plane_mask[i].squeeze(0)
        y_indices = torch.nonzero(mask, as_tuple=True)[0]
        if y_indices.numel() == 0:
            if not y_min_per_batch:
            #     y_min = y_min_per_batch[i-1]
            # else:
                y_min = torch.tensor(height).unsqueeze(0).unsqueeze(0)
            y_min_per_batch.append(y_min)
            continue
        y_min = y_indices.min().item()
        y_min = torch.tensor(y_min).unsqueeze(0).unsqueeze(0)
        y_min_per_batch.append(y_min)
    return torch.cat(y_min_per_batch, dim=0)  # b, 1

def compute_mean_ground_points(cam_points, ground_mask, height, width):
    ground_points = cam_points[:,:3].reshape(-1, 3, height, width) * ground_mask
    ground_mask_bool = (ground_mask == 1).expand_as(ground_points)
    masked_ground_points = ground_points * ground_mask_bool.float()
    
    sum_ground_points = masked_ground_points.sum(dim=(2, 3))
    count_ground_points = ground_mask_bool.sum(dim=(2, 3)).clamp(min=1e-6).float()
    
    mean_ground_points = (sum_ground_points / count_ground_points).unsqueeze(2).unsqueeze(2)
    return mean_ground_points.repeat(1, 1, height, width)

def compute_adjusted_y_distances(motion_points, ground_points, plane_param, mask, height, width):
    ground_normals = plane_param
    y_axis = torch.tensor([0, 1, 0], device=ground_normals.device).view(1, 3)
    
    cosine_angles = torch.nn.functional.cosine_similarity(ground_normals, y_axis, dim=1)
    angles = torch.acos(cosine_angles).unsqueeze(1).unsqueeze(1)
    angles = angles.repeat(1, height, width)
    
    # changed_ground_points = ground_points[:, 1] * torch.cos(angles)
    # print(ground_points[0, 1])
    # # print(changed_ground_points[1])
    # print(motion_points[0, 1])

    y_distances = torch.abs(motion_points[:, 1] - ground_points[:, 1])
    adjusted_y_distances = y_distances * torch.cos(angles)
    # print(adjusted_y_distances.unsqueeze(1) * (mask > 0.4) * 60)
    return adjusted_y_distances.unsqueeze(1) * (mask > 0.4) * 60

def visualize_distances(adjusted_y_distances):
    """
    Visualizes the distances by applying a color map and normalizing the values.
    
    Parameters:
    adjusted_y_distances (Tensor): Adjusted y-axis distances [b, 1, h, w].

    Returns:
    Tensor: Visualized motion distances as tensors with shape [b, 3, h, w].
    """
    # Remove the single channel dimension [b, h, w]
    adjusted_y_distances = adjusted_y_distances.squeeze(1)
    
    # Normalize distances
    min_val_h = adjusted_y_distances.min(dim=1, keepdim=True)[0]  # [b, 1, w]
    min_val = min_val_h.min(dim=2, keepdim=True)[0]  # [b, 1, 1]
    
    max_val_h = adjusted_y_distances.max(dim=1, keepdim=True)[0]  # [b, 1, w]
    max_val = max_val_h.max(dim=2, keepdim=True)[0]  # [b, 1, 1]

    normalized_distances = (adjusted_y_distances - min_val) / (max_val - min_val + 1e-6)

    # List to store the visualized distances
    dis_vis = []
    
    # Visualize for each batch element
    for i in range(normalized_distances.shape[0]):
        np_img = normalized_distances[i].cpu().numpy()  # Convert to numpy [h, w]

        # Apply magma colormap
        cmap = plt.get_cmap('magma')
        colored_img = cmap(np_img)[:, :, :3]  # Convert to RGB [h, w, 3]
        
        # Change format to [3, h, w]
        colored_img = np.transpose(colored_img, (2, 0, 1))

        # Convert back to tensor
        img_tensor = torch.tensor(colored_img, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        dis_vis.append(img_tensor)
    
    # Concatenate all visualized distances into a single tensor
    return torch.cat(dis_vis, dim=0)


def warp_forward_with_flow(base_images, flow, base_occlusions=None):
    """
    Perform forward warping of the base images using the given flow.
    
    Args:
        base_images: Tensor of shape [B, C, H, W], input images to warp.
        flow: Tensor of shape [B, 2, H, W], forward flow field (x and y).
        base_occlusions: Optional occlusion mask (not implemented in this example).
        
    Returns:
        Tensor of shape [B, C, H, W], warped images.
    """

    device = base_images.device  # 获取 base_images 的设备
    B, C, H, W = base_images.shape
    
    # 创建坐标张量并移动到 GPU
    coords_x = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)         
    coords_y = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

    # 解析光流
    flow_x = flow[:, 0, :, :].clone()                                                   
    flow_y = flow[:, 1, :, :].clone()

    # 调整坐标
    coordinates_x = coords_x + flow_x # 克隆 flow_x
    coordinates_y = coords_y + flow_y  # 克隆 flow_y

    # 插值计算
    indices_x = torch.stack((torch.floor(coordinates_x), torch.floor(coordinates_x) + 1), dim=-1).clamp(0, W-1)         
    indices_y = torch.stack((torch.floor(coordinates_y), torch.floor(coordinates_y) + 1), dim=-1).clamp(0, H-1)

    # 权重计算
    unilinear_weights_x = torch.abs(coordinates_x.unsqueeze(-1) - indices_x.float())                                
    unilinear_weights_y = torch.abs(coordinates_y.unsqueeze(-1) - indices_y.float())


    unilinear_weights_x = unilinear_weights_x / unilinear_weights_x.sum(dim=-1, keepdim=True)                       
    unilinear_weights_y = unilinear_weights_y / unilinear_weights_y.sum(dim=-1, keepdim=True)


    # 处理 occlusions (可选)
    if base_occlusions is not None:
        # 这里可以实现 occlusion 处理逻辑
        pass

    # 合成图像
    match_images = torch.zeros_like(base_images).to(device)  # 确保 match_images 在同一设备上
    for i in range(2):  # x 和 y 方向插值
        idx_x = indices_x[..., i].long()
        idx_y = indices_y[..., i].long()

        # 检查 idx_x 和 idx_y 的维度
        print(f"Base images shape: {base_images.shape}, idx_x shape: {idx_x.shape}, idx_y shape: {idx_y.shape}")
        
        # 确保 warped 的计算在同一设备上
        warped = base_images[torch.arange(B, device=device).unsqueeze(1).unsqueeze(1), :, idx_y, idx_x]
        match_images += warped * unilinear_weights_x.unsqueeze(1) * unilinear_weights_y.unsqueeze(1)

    return match_images


# class Forward_Warp_Python:
#     def forward(im0, flow, interpolation_mode):
#         im1 = torch.zeros_like(im0)
#         B = im0.shape[0]
#         H = im0.shape[2]
#         W = im0.shape[3]
#         if interpolation_mode == 0:
#             for b in range(B):
#                 for h in range(H):
#                     for w in range(W):
#                         x = w + flow[b, h, w, 0]
#                         y = h + flow[b, h, w, 1]
#                         nw = (int(torch.floor(x)), int(torch.floor(y)))
#                         ne = (nw[0]+1, nw[1])
#                         sw = (nw[0], nw[1]+1)
#                         se = (nw[0]+1, nw[1]+1)
#                         p = im0[b, :, h, w]
#                         if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
#                             nw_k = (se[0]-x)*(se[1]-y)
#                             ne_k = (x-sw[0])*(sw[1]-y)
#                             sw_k = (ne[0]-x)*(y-ne[1])
#                             se_k = (x-nw[0])*(y-nw[1])
#                             im1[b, :, nw[1], nw[0]] += nw_k*p
#                             im1[b, :, ne[1], ne[0]] += ne_k*p
#                             im1[b, :, sw[1], sw[0]] += sw_k*p
#                             im1[b, :, se[1], se[0]] += se_k*p
#         else:
#             round_flow = torch.round(flow)
#             for b in range(B):
#                 for h in range(H):
#                     for w in range(W):
#                         x = w + int(round_flow[b, h, w, 0])
#                         y = h + int(round_flow[b, h, w, 1])
#                         if x >= 0 and x < W and y >= 0 and y < H:
#                             im1[b, :, y, x] = im0[b, :, h, w]
#         return im1


def Forward_Warp_Python(im0, flow, interpolation_mode):
    im1 = torch.zeros_like(im0)
    B = im0.shape[0]
    H = im0.shape[2]
    W = im0.shape[3]
    if interpolation_mode == 0:
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    x = w + flow[b, h, w, 0]
                    y = h + flow[b, h, w, 1]
                    nw = (int(torch.floor(x)), int(torch.floor(y)))
                    ne = (nw[0]+1, nw[1])
                    sw = (nw[0], nw[1]+1)
                    se = (nw[0]+1, nw[1]+1)
                    p = im0[b, :, h, w]
                    if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
                        nw_k = (se[0]-x)*(se[1]-y)
                        ne_k = (x-sw[0])*(sw[1]-y)
                        sw_k = (ne[0]-x)*(y-ne[1])
                        se_k = (x-nw[0])*(y-nw[1])
                        im1[b, :, nw[1], nw[0]] += nw_k*p
                        im1[b, :, ne[1], ne[0]] += ne_k*p
                        im1[b, :, sw[1], sw[0]] += sw_k*p
                        im1[b, :, se[1], se[0]] += se_k*p
    else:
        round_flow = torch.round(flow)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    x = w + int(round_flow[b, h, w, 0])
                    y = h + int(round_flow[b, h, w, 1])
                    if x >= 0 and x < W and y >= 0 and y < H:
                        im1[b, :, y, x] = im0[b, :, h, w]
    return im1


import torch

def forward_warp_vectorized(im0, flow, interpolation_mode):
    B, C, H, W = im0.shape
    im1 = torch.zeros_like(im0)  # Initialize output image

    # Get the meshgrid for pixel coordinates
    w_grid, h_grid = torch.meshgrid(torch.arange(W, device=flow.device), torch.arange(H, device=flow.device))
    w_grid, h_grid = w_grid.T, h_grid.T  # Ensure (H, W) shape

    if interpolation_mode == 0:
        # Coordinates after applying flow
        x = w_grid.unsqueeze(0) + flow[:, :, :, 0]  # (B, H, W)
        y = h_grid.unsqueeze(0) + flow[:, :, :, 1]  # (B, H, W)
        
        # Get neighboring pixel coordinates
        nw_x = torch.floor(x).long()
        nw_y = torch.floor(y).long()
        ne_x = nw_x + 1
        ne_y = nw_y
        sw_x = nw_x
        sw_y = nw_y + 1
        se_x = nw_x + 1
        se_y = nw_y + 1

        # Clip coordinates to image boundaries
        nw_x = torch.clamp(nw_x, 0, W - 1)
        nw_y = torch.clamp(nw_y, 0, H - 1)
        ne_x = torch.clamp(ne_x, 0, W - 1)
        ne_y = torch.clamp(ne_y, 0, H - 1)
        sw_x = torch.clamp(sw_x, 0, W - 1)
        sw_y = torch.clamp(sw_y, 0, H - 1)
        se_x = torch.clamp(se_x, 0, W - 1)
        se_y = torch.clamp(se_y, 0, H - 1)

        # Calculate interpolation weights
        nw_weight = (se_x.float() - x) * (se_y.float() - y)
        ne_weight = (x - sw_x.float()) * (sw_y.float() - y)
        sw_weight = (ne_x.float() - x) * (y - ne_y.float())
        se_weight = (x - nw_x.float()) * (y - nw_y.float())

        # Accumulate interpolated values using scatter_add_
        for c in range(C):  # Batch process across all channels at once
            # Flatten coordinates and weights for easier accumulation
            flat_nw_idx = (nw_y * W + nw_x).flatten()
            flat_ne_idx = (ne_y * W + ne_x).flatten()
            flat_sw_idx = (sw_y * W + sw_x).flatten()
            flat_se_idx = (se_y * W + se_x).flatten()

            # Flatten image slice for current channel
            flat_im0 = im0[:, c].flatten(start_dim=1)

            # Using scatter_add_ to accumulate at correct positions
            im1[:, c].view(B, -1).scatter_add_(1, flat_nw_idx.unsqueeze(0).expand(B, -1), (nw_weight * flat_im0).flatten(start_dim=1))
            im1[:, c].view(B, -1).scatter_add_(1, flat_ne_idx.unsqueeze(0).expand(B, -1), (ne_weight * flat_im0).flatten(start_dim=1))
            im1[:, c].view(B, -1).scatter_add_(1, flat_sw_idx.unsqueeze(0).expand(B, -1), (sw_weight * flat_im0).flatten(start_dim=1))
            im1[:, c].view(B, -1).scatter_add_(1, flat_se_idx.unsqueeze(0).expand(B, -1), (se_weight * flat_im0).flatten(start_dim=1))

    else:
        # Rounding flow for non-interpolated case
        round_flow = torch.round(flow).long()
        new_w = torch.clamp(w_grid.unsqueeze(0) + round_flow[:, :, :, 0], 0, W - 1)
        new_h = torch.clamp(h_grid.unsqueeze(0) + round_flow[:, :, :, 1], 0, H - 1)

        # Direct assignment for non-interpolated pixels
        for b in range(B):
            im1[b].scatter_(1, (new_h[b] * W + new_w[b]).flatten().unsqueeze(0), im0[b].view(C, -1))

    return im1

def forward_warp_nearest(im0, flow):
    B, C, H, W = im0.shape

    # Create an output tensor filled with zeros
    im1 = torch.zeros_like(im0)

    # Generate pixel coordinate grid
    h_grid, w_grid = torch.meshgrid(
        torch.arange(H, device=flow.device),
        torch.arange(W, device=flow.device)
    )
    
    # Stack into (H, W, 2) shape for coordinate grid
    base_grid = torch.stack((w_grid, h_grid), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Compute the target coordinates by adding flow to base grid
    target_coords = base_grid + flow

    # Round the target coordinates to nearest integers
    target_coords_rounded = target_coords.round().long()

    # Clip to ensure the coordinates are within image bounds
    target_coords_clipped = torch.clamp(target_coords_rounded, min=0, max=max(H, W) - 1)

    # Assign pixels from im0 to im1 based on flow displacement
    for b in range(B):
        for c in range(C):
            im1[b, c, target_coords_clipped[b, :, :, 1], target_coords_clipped[b, :, :, 0]] = im0[b, c, :, :]

    return im1

import torch

# def forward_warp_bilinear(im0, flow):
#     B, C, H, W = im0.shape
    
#     # Create an output tensor filled with zeros
#     im1 = torch.zeros_like(im0)

#     # Generate pixel coordinate grid
#     h_grid, w_grid = torch.meshgrid(
#         torch.arange(H, device=flow.device),
#         torch.arange(W, device=flow.device)
#     )
    
#     # Stack into (H, W, 2) shape for coordinate grid
#     base_grid = torch.stack((w_grid, h_grid), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

#     # Compute the target coordinates by adding flow to base grid
#     target_coords = base_grid + flow
    
#     # Extract fractional and integer parts of the target coordinates
#     target_x, target_y = target_coords[:, :, :, 0], target_coords[:, :, :, 1]
#     x0 = torch.floor(target_x).long()  # Left
#     x1 = x0 + 1  # Right
#     y0 = torch.floor(target_y).long()  # Top
#     y1 = y0 + 1  # Bottom

#     # Clip coordinates to ensure they're within image bounds
#     x0 = torch.clamp(x0, 0, W - 1)
#     x1 = torch.clamp(x1, 0, W - 1)
#     y0 = torch.clamp(y0, 0, H - 1)
#     y1 = torch.clamp(y1, 0, H - 1)

#     # Compute the interpolation weights
#     wa = (x1.float() - target_x) * (y1.float() - target_y)  # Top-left (NW)
#     wb = (target_x - x0.float()) * (y1.float() - target_y)  # Top-right (NE)
#     wc = (x1.float() - target_x) * (target_y - y0.float())  # Bottom-left (SW)
#     wd = (target_x - x0.float()) * (target_y - y0.float())  # Bottom-right (SE)

#     # Perform the forward warp by distributing pixel values with weights
#     for b in range(B):
#         for c in range(C):
#             im1[b, c, y0[b], x0[b]] += wa[b] * im0[b, c, :, :]
#             im1[b, c, y0[b], x1[b]] += wb[b] * im0[b, c, :, :]
#             im1[b, c, y1[b], x0[b]] += wc[b] * im0[b, c, :, :]
#             im1[b, c, y1[b], x1[b]] += wd[b] * im0[b, c, :, :]

#     return im1

import torch

def forward_warp_bilinear(im0, flow):
    B, C, H, W = im0.shape
    
    # Create an output tensor filled with zeros
    im1 = torch.zeros_like(im0)

    # Generate pixel coordinate grid
    h_grid, w_grid = torch.meshgrid(
        torch.arange(H, device=flow.device),
        torch.arange(W, device=flow.device)
    )
    
    # Stack into (H, W, 2) shape for coordinate grid
    base_grid = torch.stack((w_grid, h_grid), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Compute the target coordinates by adding flow to base grid
    target_coords = base_grid + flow
    
    # Extract fractional and integer parts of the target coordinates
    target_x, target_y = target_coords[:, :, :, 0], target_coords[:, :, :, 1]
    x0 = torch.floor(target_x).long()  # Left
    x1 = x0 + 1  # Right
    y0 = torch.floor(target_y).long()  # Top
    y1 = y0 + 1  # Bottom

    # Clip coordinates to ensure they're within image bounds
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # Compute the interpolation weights
    wa = (x1.float() - target_x) * (y1.float() - target_y)  # Top-left (NW)
    wb = (target_x - x0.float()) * (y1.float() - target_y)  # Top-right (NE)
    wc = (x1.float() - target_x) * (target_y - y0.float())  # Bottom-left (SW)
    wd = (target_x - x0.float()) * (target_y - y0.float())  # Bottom-right (SE)

    # Reshape the image into flat form for easier indexing and accumulation
    flat_im0 = im0.view(B, C, -1)  # (B, C, H*W)
    
    # Flatten the coordinate grid for easier scatter_add
    flat_idx_tl = (y0 * W + x0).view(B, -1)  # Top-left
    flat_idx_tr = (y0 * W + x1).view(B, -1)  # Top-right
    flat_idx_bl = (y1 * W + x0).view(B, -1)  # Bottom-left
    flat_idx_br = (y1 * W + x1).view(B, -1)  # Bottom-right

    # Reshape the weights
    wa = wa.view(B, -1)  # Top-left weights
    wb = wb.view(B, -1)  # Top-right weights
    wc = wc.view(B, -1)  # Bottom-left weights
    wd = wd.view(B, -1)  # Bottom-right weights

    # Perform scatter_add_ to accumulate weighted pixel values
    for b in range(B):
        for c in range(C):
            im1[b, c].view(-1).scatter_add_(0, flat_idx_tl[b], (flat_im0[b, c] * wa[b]).view(-1))
            im1[b, c].view(-1).scatter_add_(0, flat_idx_tr[b], (flat_im0[b, c] * wb[b]).view(-1))
            im1[b, c].view(-1).scatter_add_(0, flat_idx_bl[b], (flat_im0[b, c] * wc[b]).view(-1))
            im1[b, c].view(-1).scatter_add_(0, flat_idx_br[b], (flat_im0[b, c] * wd[b]).view(-1))

    return im1

# def depth_consistency_loss(depth, mask_ground, mask_object):
#     B, C, H, W = depth.shape

#     mask_depth = mask_ground * depth

#     N_1 = 3
#     N_2 = 5
#     conv_kernel_1 = torch.ones(1, 1, N_1, 1) / N_1
#     conv_kernel_2 = torch.ones(1, 1, N_2, 1) / N_2

#     mean_below_1 = F.conv2d(depth, conv_kernel_1, padding=(N_1 - 1, 0))
#     mean_below_2 = F.conv2d(depth, conv_kernel_2, padding=(N_2 - 1, 0))

#     shift_mask_depth_1 = torch.roll(mask_depth, 1, dims=2)
#     shift_mask_depth_2 = torch.roll(mask_depth, 1, dims=2)

#     shift_mask_1 = torch.roll(mask_ground, 1, dims=2)
#     shift_mask_2 = torch.roll(mask_ground, 2, dims=2)

#     mask_1 = shift_mask_1 * mask_object
#     mask_2 = shift_mask_2 * mask_object

#     loss_1 = mask_1 * (shift_mask_depth_1 - mean_below_1)
#     loss_1 = loss_1.mean()

#     loss_2 = (mask_2 - mask_1) * (shift_mask_depth_2 - mean_below_2)
#     loss_2 = loss_2.mean()

def depth_consistency_loss(depth, mask_ground, mask_object):
    B, C, H, W = depth.shape
    
    # 定义卷积核（保持输入输出尺寸一致）
    def create_conv(N):
        conv = nn.Conv2d(1, 1, kernel_size=(N,1), padding=(N-1,0), bias=False)
        conv.weight.data = torch.ones(1,1,N,1)/N
        conv.requires_grad_(False)
        return conv.to(depth.device)
    
    conv3 = create_conv(3)  # 下方3像素均值
    conv5 = create_conv(5)  # 下方5像素均值
    
    # 计算下方均值（输出尺寸 H+2 和 H+4）
    mean3 = conv3(depth.detach())[:, :, :-2, :]  # 裁剪最后2行保持H尺寸
    mean5 = conv5(depth.detach())[:, :, :-4, :]  # 裁剪最后4行
    
    # 非循环位移（向上1像素）
    def shift_up(x, k):
        return F.pad(x, (0,0,0,k))[:, :, k:, :]  # 顶部填充0，取H行
    
    # 获取位移后的深度和掩码
    depth_shift1 = shift_up(depth, 1)[:, :, :H-1, :]  # 新尺寸 (B,1,H-1,W)
    mask_shift1 = shift_up(mask_ground, 1)[:, :, :H-1, :]
    mask_shift2 = shift_up(mask_ground, 2)[:, :, :H-2, :]
    
    # 对齐所有张量到 (B,1,H-2,W)
    target_H = H-2
    depth_shift1 = depth_shift1[:, :, :target_H, :]
    mask_shift1 = mask_shift1[:, :, :target_H, :]
    mean3 = mean3[:, :, :target_H, :]
    mean5 = mean5[:, :, :target_H, :]
    mask_object = mask_object[:, :, :target_H, :]
    
    # 计算有效区域
    valid_mask1 = (mask_shift1 * mask_object).float()
    valid_mask2 = (mask_shift2[:, :, :target_H, :] * mask_object).float()
    
    # 计算损失（MSE）
    loss1 = (valid_mask1 * (depth_shift1 - mean3)**2).mean()
    loss2 = ((valid_mask2 - valid_mask1) * (depth_shift1 - mean5)**2).mean()
    
    return loss1, valid_mask1, valid_mask2, mean3


def detect_ground_edges(mask_ground, y_threshold=0.8, noise_threshold=0.1):
    """
    检测法线方向接近 y 轴方向的地面边缘，并筛除噪音点。
    
    参数:
        mask_ground (torch.Tensor): 地面掩码，形状为 (B, 1, H, W)。
        y_threshold (float): 法线方向接近 y 轴的阈值（默认为 0.9）。
        noise_threshold (float): 噪音点过滤的阈值（默认为 0.1）。
    
    返回:
        edges (torch.Tensor): 检测到的地面边缘，形状为 (B, 1, H, W)。
    """
    B, C, H, W = mask_ground.shape
    
    # 1. 对地面掩码进行形态学操作（膨胀和腐蚀）以去除噪音点
    kernel = torch.ones(1, 1, 3, 3, device=mask_ground.device)  # 3x3 卷积核
    mask_ground = F.max_pool2d(mask_ground, kernel_size=3, stride=1, padding=1)  # 膨胀
    mask_ground = F.avg_pool2d(mask_ground, kernel_size=3, stride=1, padding=1)  # 腐蚀
    
    # 2. 使用 Sobel 算子检测边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask_ground.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask_ground.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(mask_ground, sobel_x, padding=1)
    grad_y = F.conv2d(mask_ground, sobel_y, padding=1)
    
    # 计算梯度幅值和方向
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_dir = grad_y / (grad_mag + 1e-6)  # 法线方向的 y 分量
    
    # 3. 筛选法线方向接近 y 轴的边缘
    edges = (grad_dir > y_threshold) & (grad_mag > noise_threshold)
    
    # 4. 确保边缘线下方为地面区域，上方为非地面区域
    mask_ground_shifted = F.pad(mask_ground[:, :, 1:, :], (0, 0, 0, 1), mode='constant', value=0)  # 下移 1 像素
    edges = edges & (mask_ground_shifted > 0) & (mask_ground == 0)
    
    return edges.float()