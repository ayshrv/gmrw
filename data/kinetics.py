from __future__ import print_function, absolute_import

import math
import time

import os
import os.path as osp
import random
from glob import glob
from math import ceil
import mediapy as media

import numpy as np
import torch
import torch.utils.data as data

from data.augs import get_color_aug_transform, get_single_image_aug_transform
from data.flow_augmentations import (CenterCrop, RandomResizedCrop, Resize,
                                     ToTensor)

class KineticsDataset(data.Dataset):
    def __init__(
        self, 
        use_frame_transform=True,
        use_color_transform=False,
        img_size1=None,
        img_size2=None,
        split="train",
        root="datasets/kinetics700-2020",
        no_of_frames=2,
        frame_len_diff=20,
        random_frame_skip=False,
        random_seed=786234,
        aug_setting="setting1",
        training=True,
    ):

        assert isinstance(img_size1, tuple) and len(img_size1) == 2
        assert isinstance(img_size2, tuple) and len(img_size2) == 2
        
        assert (
            use_color_transform is False
        )  ## Not using color transforms, change color_augment to handle batches if True

        self.use_frame_transform = use_frame_transform
        self.use_color_transform = use_color_transform
        self.frame_len_diff = frame_len_diff
        self.no_of_frames = no_of_frames
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        self.random_frame_skip = random_frame_skip

        self.random_frame_skip_len = 20

        self.video_list = []
        self.data_root = root
        self.split = split

        self.video_list = sorted(glob(osp.join(self.data_root, self.split, "*/*.mp4")))

        self.ignore_id_list = [
            "datasets/kinetics700-2020/train/adjusting glasses/5d9mIpws4cg_000130_000140.mp4",
            "datasets/kinetics700-2020/train/changing gear in car/A-FCzUzEd4U_000000_000010.mp4",
            "datasets/kinetics700-2020/train/cleaning shoes/y7cYaYX4gdw_000047_000057.mp4", 
            "datasets/kinetics700-2020/train/cracking back/SYTMgaqGhfg_000010_000020.mp4",
            "datasets/kinetics700-2020/train/faceplanting/BSN_nDiTwBo_000004_000014.mp4",
            "datasets/kinetics700-2020/train/flipping pancake/zLD_q2djrYs_000030_000040.mp4", 
            "datasets/kinetics700-2020/train/gospel singing in church/NNazT7dDWxA_000130_000140.mp4",
            "datasets/kinetics700-2020/train/making sushi/_dbw-EJqoMY_001023_001033.mp4",
            "datasets/kinetics700-2020/train/punching bag/ixQrfusr6k8_000001_000011.mp4",
            "datasets/kinetics700-2020/train/roller skating/FAqHwAPZfeE_000018_000028.mp4"
        ]

        for video_id in self.ignore_id_list:
            if video_id in self.video_list:
                self.video_list.remove(video_id)

        video_id = self.video_list[0]

        frames = media.read_video(video_id)
        
        H, W = frames[0].shape[:2]

        self.img_size0 = (H, W)

        ## Transforms related inits
        ## Non-breaking shortcuts Transforms
        if self.use_frame_transform:
            aspect_ratio = img_size1[0] / img_size1[1]
            self.to_tensor = ToTensor(convert_to_float=True)
            self.random_crop = RandomResizedCrop(
                size=img_size1,
                scale=(1, 1),
                ratio=(aspect_ratio, aspect_ratio),
                seed=random_seed,
            )
            if training is False:
                self.center_crop_1 = CenterCrop(size=img_size1)

                if img_size1[0] > self.img_size0[0] or img_size1[1] > self.img_size0[1]:
                    scale_H = img_size1[0] / self.img_size0[0]
                    scale_W = img_size1[1] / self.img_size0[1]
                    max_scale = max(scale_H, scale_W)
                    new_img_size0 = (
                        ceil(max_scale * self.img_size0[0]),
                        ceil(max_scale * self.img_size0[1]),
                    )
                    self.resize_crop_1 = Resize(size=new_img_size0)
                else:
                    self.resize_crop_1 = None

            else:
                self.center_crop_1 = None
                self.resize_crop_1 = None

        if use_color_transform:
            self.color_aug = get_single_image_aug_transform()
        else:
            self.color_aug = None

        ## Breaking shortcuts Transforms
        self.aspect_ratio = img_size2[0] / img_size2[1]
        # Args for RandomResizedCrop
        if aug_setting == "setting1":
            self.aug_scale = [0.08, 1.0]
            self.aug_ratio = [self.aspect_ratio * 0.7, self.aspect_ratio * 1.3]
        elif aug_setting == "setting2":
            self.aug_scale = [0.3, 1.0]
            self.aug_ratio = [self.aspect_ratio * 0.7, self.aspect_ratio * 1.3]
        elif aug_setting == "setting3":
            self.aug_scale = [0.5, 1.0]
            self.aug_ratio = [self.aspect_ratio * 0.7, self.aspect_ratio * 1.3]
        elif aug_setting == "setting0":
            self.aug_scale = [1.0, 1.0]
            self.aug_ratio = [self.aspect_ratio, self.aspect_ratio]

        self.random_resized_crop_transform = RandomResizedCrop(
            size=img_size2,
            scale=self.aug_scale,
            ratio=self.aug_ratio,
            seed=random_seed,
        )

        if training is False:
            self.center_crop_2 = CenterCrop(size=img_size2)

            if img_size2[0] > img_size1[0] or img_size2[1] > img_size1[1]:
                scale_H = img_size2[0] / img_size1[0]
                scale_W = img_size2[1] / img_size1[1]
                max_scale = max(scale_H, scale_W)
                new_img_size1 = (
                    ceil(max_scale * img_size1[0]),
                    ceil(max_scale * img_size1[1]),
                )
                self.resize_crop_2 = Resize(size=new_img_size1)
            else:
                self.resize_crop_2 = None
        else:
            self.center_crop_2 = None
            self.resize_crop_2 = None

        if use_color_transform:
            self.color_aug_tuple = get_color_aug_transform()
        else:
            self.color_aug_tuple = None


    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]

        try:
            frames = media.read_video(video_id)
        except:

            print(f"Error in video id {video_id}")
            # fmt: off
            import ipdb; ipdb.set_trace() 
            # fmt: on
            
        frame_length = len(frames)
        
        if frame_length - self.frame_len_diff < 1:
            frame1_idx = 0
            frame2_idx = frame_length - 1
        else:
            frame1_idx = np.random.randint(0, frame_length - self.frame_len_diff)
            frame2_idx = frame1_idx + self.frame_len_diff

        frame1 = frames[frame1_idx]
        frame2 = frames[frame2_idx]

        frames = [frame1, frame2]
        
        # frames: (T, C, H, W), float, 0-255
        ##########################

        frames, _, _ = self.to_tensor(frames)

        T, H, W, _ = frames.shape

        if self.use_frame_transform:
            if self.center_crop_1 is not None:
                if self.resize_crop_1 is not None:
                    frames, _, _ = self.resize_crop_1(frames)
                frames, _, _ = self.center_crop_1(frames)
            frames, _, _, _ = self.random_crop(frames)

        if self.color_aug:
            frames_aug = []
            for t in range(frames.shape[0]):
                frames_aug.append(self.color_aug(frames[t]))
            frames = torch.stack(frames_aug, dim=0)
        else:
            frames = frames / 255.0

        # frames: float, 0-1

        # frames: (T, C, H, W)
        H, W = frames.shape[2:]

        frames = frames.reshape(T, 3, H, W)

        # color augmentation
        # (T, 3, H, W)
        frames_forward, frames_backward = self.color_augment(frames)
        
        if self.center_crop_2 is not None:
            if self.resize_crop_2 is not None:
                frames_forward, _, _ = self.resize_crop_2(
                    frames_forward
                )
            frames_forward, _, _ = self.center_crop_2(frames_forward)

        (
            frames_forward,
            _,
            _,
            affine_mat_forward,
        ) = self.random_resized_crop_transform(frames_forward)

        if self.center_crop_2 is not None:
            if self.resize_crop_2 is not None:
                frames_backward, _, _ = self.resize_crop_2(
                    frames_backward
                )
            frames_backward, _, _ = self.center_crop_2(frames_backward)
        (
            frames_backward,
            _,
            _,
            affine_mat_backward,
        ) = self.random_resized_crop_transform(frames_backward)

        H, W = frames_forward.shape[2:]

        # frames: 2*(T-1), 3, H, W
        frames = torch.cat((frames_forward, frames_backward), dim=0)

        affine_mat_backward_inv = torch.inverse(affine_mat_backward[0])
        affine_mat_f2b = torch.matmul(affine_mat_forward[0], affine_mat_backward_inv)
        affine_mat_f2b = affine_mat_f2b[:2]

        affine_mat_forward_inv = torch.inverse(affine_mat_forward[0])
        affine_mat_b2f = torch.matmul(affine_mat_backward[0], affine_mat_forward_inv)
        affine_mat_b2f = affine_mat_b2f[:2]

        # H, W: img_size2
        # frames: (2*T, 3, H, W), float, 0-1

        return (
            frames,
            affine_mat_b2f,
        )


    def color_augment(self, frames):
        frames_aug = []
        frames_aug_back = []
        if isinstance(self.color_aug, tuple):
            for t in range(frames.size(0)):
                frames_aug.append(self.color_aug[0](frames[t]))
            for t in reversed(range(frames.size(0))):
                frames_aug_back.append(self.color_aug[1](frames[t]))
        elif self.color_aug is None:
            return frames, frames.flip(0)
        else:
            for t in range(frames.size(0)):
                frames_aug.append(self.color_aug(frames[t]))
            for t in reversed(range(frames.size(0))):
                frames_aug_back.append(self.color_aug(frames[t]))
        return torch.stack(frames_aug, dim=0), torch.stack(frames_aug_back, dim=0)
