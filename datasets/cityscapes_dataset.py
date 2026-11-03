# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)

        if self.is_train:
            self.RAW_WIDTH = 1024
            self.RAW_HEIGHT = 384
        else:
            self.RAW_WIDTH = 2048
            self.RAW_HEIGHT = 1024       

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        city, frame_name = self.filenames[index].split()
        side = None
        return city, frame_name, side

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner
        if self.is_train:
            camera_file = os.path.join(self.data_path, city, "{}_cam.txt".format(frame_name))
            camera = np.loadtxt(camera_file, delimiter=",")
            fx = camera[0]
            fy = camera[4]
            u0 = camera[2]
            v0 = camera[5]
            intrinsics = np.array([[fx, 0, u0, 0],
                                [0, fy, v0, 0],
                                [0,  0,  1, 0],
                                [0,  0,  0, 1]]).astype(np.float32)

            intrinsics[0, :] /= self.RAW_WIDTH
            intrinsics[1, :] /= self.RAW_HEIGHT
        else:
            split = "test"  # if self.is_train else "val"
            camera_file = os.path.join(self.data_path, 'camera',
                                    split, city, frame_name + '_camera.json')
            with open(camera_file, 'r') as f:
                camera = json.load(f)
            fx = camera['intrinsic']['fx']
            fy = camera['intrinsic']['fy']
            u0 = camera['intrinsic']['u0']
            v0 = camera['intrinsic']['v0']
            intrinsics = np.array([[fx, 0, u0, 0],
                                [0, fy, v0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).astype(np.float32)
            intrinsics[0, :] /= self.RAW_WIDTH
            intrinsics[1, :] /= self.RAW_HEIGHT * 0.75            
        return intrinsics

    def get_color(self, city, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        inputs = {}
        if self.is_train:
            color = self.loader(self.get_image_path(city, frame_name))
            color = np.array(color)

            h = color.shape[0] // 3
            inputs[("color", -1, -1)] = pil.fromarray(color[:h, :])
            inputs[("color", 0, -1)] = pil.fromarray(color[h:2*h, :])
            inputs[("color", 1, -1)] = pil.fromarray(color[2*h:, :])

            if do_flip:
                for key in inputs:
                    inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)
        else:
            # color = self.loader(self.get_image_path(city, frame_name))
            # w, h = color.size
            # crop_h = h * 3 // 4
            # color = color.crop((0, 0, w, crop_h))
            # inputs[("color", 0, -1)] = color
            for f_i in self.frame_idxs:
                if f_i == 0:
                    color = self.loader(self.get_image_path(city, frame_name))
                    w, h = color.size
                    crop_h = h * 3 // 4
                    color = color.crop((0, 0, w, crop_h))
                else:
                    new_frame_name =  self.modify_frame_name(frame_name, f_i==1)
                    color = self.loader(self.get_image_path(city, new_frame_name))
                    w, h = color.size
                    crop_h = h * 3 // 4
                    color = color.crop((0, 0, w, crop_h))
                inputs[("color", f_i, -1)] = color

        return inputs

    def get_image_path(self, city, frame_name):
        if self.is_train:
            return os.path.join(self.data_path, city, "{}.png".format(frame_name))
        else:
            folder = "leftImg8bit_sequence"
            split = "test"
            image_path = os.path.join(
                self.data_path, folder, split, city, frame_name + '_leftImg8bit.png')
            return image_path      
    
    def modify_frame_name(self, frame_name, increment=True):
        parts = frame_name.split('_')
        if len(parts) != 3:
            raise ValueError(f"Unexpected frame_name format: {frame_name}")
        
        # 提取部分
        city = parts[0]
        num1 = parts[1]
        num2 = int(parts[2])
        
        # 根据参数增加或减少帧号
        if increment:
            num2 += 1
        else:
            num2 -= 1
        
        # 格式化新的帧号以确保其具有相同的位数
        new_num2 = f"{num2:06d}"
        
        # 组合部分形成新的 frame_name
        new_frame_name = f"{city}_{num1}_{new_num2}"
        return new_frame_name
    
    def get_doj_mask(self, city, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        if not getattr(self, 'mask_path', None):
            raise ValueError('Cityscapes region evaluation requires --mask-path')
        city, seq, frame = frame_name.split('_')
        frame = int(frame)
        inputs = {}
        for offset in self.frame_idxs:
            suffix = '' if offset == 0 else ('-1' if offset == -1 else '+1')
            filename = '{}_{}_{}{}.npy'.format(city, seq, frame, suffix)
            mask = np.load(os.path.join(self.mask_path, filename))
            inputs['doj_mask' + suffix] = pil.fromarray(mask)

        if do_flip:
            for key in inputs:
                inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)

        # 将 PIL 图像转换为 PyTorch 张量
        # for key in inputs:
        #     inputs[key] = to_tensor(inputs[key])
            
        return inputs
    # # 示例用法
    # frame_name = "leverkusen_000039_000019"

    # # 获取前一帧的 frame_name
    # previous_frame_name = modify_frame_name(frame_name, increment=False)
    # print(previous_frame_name)  # 输出 "leverkusen_000039_000018"

    # # 获取后一帧的 frame_name
    # next_frame_name = modify_frame_name(frame_name, increment=True)
    # print(next_frame_name)  # 输出 "leverkusen_000039_000020"

