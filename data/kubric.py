import functools
import logging
import os
import os.path as osp
import random
from math import ceil

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.utils.data as data
import torch.nn.functional as F

from data.augs import get_color_aug_transform, get_single_image_aug_transform
from data.flow_augmentations import CenterCrop, RandomResizedCrop, Resize, ToTensor

logger = logging.getLogger(__name__)


class KubricDataloader:
    def __init__(
        self,
        use_frame_transform=True,
        use_color_transform=False,
        img_size1=None,
        img_size2=None,
        split="train",
        root="datasets/movi/",
        no_of_frames=2,
        frame_len_diff=1,
        random_frame_skip=False,
        aug_setting="setting1",
        random_seed=786234,
        batch_size=1,
        shuffle=False,
        worker_id=-1,
        num_workers=-1,
        num_parallel_point_extraction_calls=16,
        training=True,
        use_data_aug_curriculum=False,
    ):
        assert isinstance(img_size1, tuple) and len(img_size1) == 2
        assert isinstance(img_size2, tuple) and len(img_size2) == 2

        assert (
            use_color_transform is False
        )  ## Not using color transforms, change color_augment to handle batches if True

        self.use_frame_transform = use_frame_transform
        self.use_color_transform = use_color_transform
        self.split = split
        self.frame_len_diff = frame_len_diff
        self.no_of_frames = no_of_frames
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        self.random_seed = random_seed
        self.random_frame_skip = random_frame_skip
        self.batch_size = batch_size

        self.total_frame_len = 24
        self.random_frame_skip_len = 20

        assert (self.no_of_frames - 1) * self.frame_len_diff < self.total_frame_len

        dataset = tfds.load(
            "movi_e/256x256",
            data_dir=root,
            shuffle_files=shuffle,
        )

        dataset = dataset[split]

        if split == "validation":
            dataset = dataset.take(248)

        frame = next(iter(dataset))["video"]
        _, H, W, _ = frame.shape
        self.img_size0 = (H, W)

        self.len = dataset.cardinality().numpy()

        self.num_workers = num_workers
        self.worker_id = worker_id

        self.sharded_data_len = self.len
        if self.num_workers > 0:
            dataset = dataset.shard(self.num_workers, self.worker_id)
            self.sharded_data_len = dataset.cardinality().numpy()

        if shuffle:
            dataset = dataset.shuffle(2 * batch_size)

        dataset = dataset.map(
            functools.partial(
                self.filter_data_items,
            ),
            num_parallel_calls=num_parallel_point_extraction_calls,
        )

        dataset = dataset.batch(batch_size, drop_remainder=True)

        self.dataset = tfds.as_numpy(dataset)

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
        elif aug_setting == "setting4":
            self.aug_scale = [0.7, 0.7]
            self.aug_ratio = [self.aspect_ratio, self.aspect_ratio]
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

        self.affine_th = 0.5

        if use_color_transform:
            self.color_aug_tuple = get_color_aug_transform()
        else:
            self.color_aug_tuple = None

        logger.info(f"Loaded Kubric dataset with {self.sharded_data_len} videos.")


    def filter_data_items(self, data):
        frames = data["video"]  # (T, H, W, 3)
        forward_flow_range = data["metadata"]["forward_flow_range"]
        forward_flow = data["forward_flow"]

        backward_flow_range = data["metadata"]["backward_flow_range"]
        backward_flow = data["backward_flow"]

        return {
            "frames": frames,
            "forward_flow": forward_flow,
            "backward_flow": backward_flow,
            "forward_flow_range": forward_flow_range,
            "backward_flow_range": backward_flow_range,
        }

    def color_augment(self, frames):
        frames_aug = []
        frames_aug_back = []
        if isinstance(self.color_aug_tuple, tuple):
            assert NotImplementedError
        elif self.color_aug_tuple is None:
            return frames, frames.flip(1)
        else:
            assert NotImplementedError
            # for t in range(frames.size(0)):
            #     frames_aug.append(self.color_aug_tuple(frames[t]))
            # for t in reversed(range(frames.size(0))):
            #     frames_aug_back.append(self.color_aug_tuple(frames[t]))
        return torch.stack(frames_aug, dim=0), torch.stack(frames_aug_back, dim=0)

    def preprocess_data(self, data):
        frames = data["frames"]  # (B, T, H, W, 3)
        forward_flow = data["forward_flow"]  # (B, T, H, W, 2)
        backward_flow = data["backward_flow"]  # (B, T, H, W, 2)

        forward_flow_range = data["forward_flow_range"]  # (B, 2)
        minv = forward_flow_range[:, 0][:, None, None, None, None]  # (B, 1, 1, 1, 1)
        maxv = forward_flow_range[:, 1][:, None, None, None, None]  # (B, 1, 1, 1, 1)
        forward_flow = forward_flow / 65535 * (maxv - minv) + minv

        backward_flow_range = data["backward_flow_range"]  # (B, 2)
        minv = backward_flow_range[:, 0][:, None, None, None, None]  # (B, 1, 1, 1, 1)
        maxv = backward_flow_range[:, 1][:, None, None, None, None]  # (B, 1, 1, 1, 1)
        backward_flow = backward_flow / 65535 * (maxv - minv) + minv
        backward_flow = -backward_flow

        forward_flow = np.flip(forward_flow, 4)
        backward_flow = np.flip(backward_flow, 4)

        frames = torch.tensor(frames)  # (B, T, H, W, 3)
        forward_flow = torch.tensor(forward_flow.copy())  # (B, T, H, W, 2)
        backward_flow = torch.tensor(backward_flow.copy())  # (B, T, H, W, 2)

        forward_flow = forward_flow[:, :-1]  # (B, T-1, H, W, 2)
        backward_flow = backward_flow[:, 1:]  # (B, T-1, H, W, 2)

        B, T, H, W, _ = frames.shape

        if self.random_frame_skip:
            # random skip frames
            frame_len_diff = random.randint(1, self.random_frame_skip_len)
        else:
            frame_len_diff = self.frame_len_diff

        random_inital_t = random.randint(
            0,
            self.total_frame_len - (self.no_of_frames - 1) * (frame_len_diff) - 1,
        )

        i = random_inital_t
        l = frame_len_diff
        n = self.no_of_frames
        frames = frames[:, i : i + n * l : l]
        forward_flow = forward_flow[:, i : i + (n - 1) * l : l]
        backward_flow = backward_flow[:, i + l - 1 : i + (n - 1) * l : l]

        T = self.no_of_frames

        flows_f_b = torch.cat(
            [forward_flow, backward_flow], dim=1
        )  # (B, 2(T-1), H, W, 2)

        frames = frames.reshape(B * T, H, W, 3)
        flows_f_b = flows_f_b.reshape(2 * B * (T - 1), H, W, 2)

        frames, flows_f_b, _ = self.to_tensor(frames, flows_f_b)

        if self.use_frame_transform:
            if self.center_crop_1 is not None:
                if self.resize_crop_1 is not None:
                    frames, flows_f_b, _ = self.resize_crop_1(frames, flows_f_b)
                frames, flows_f_b, _ = self.center_crop_1(frames, flows_f_b)
            frames, flows_f_b, _, _ = self.random_crop(frames, flows_f_b)

        if self.color_aug:
            frames_aug = []
            for t in range(frames.shape[0]):
                frames_aug.append(self.color_aug(frames[t]))
            frames = torch.stack(frames_aug, dim=0)
        else:
            frames = frames / 255.0

        # frames: float, 0-1

        # frames: (B*T, C, H, W)
        # flows_f_b: (B*2(T-1), 2, H, W)

        H, W = frames.shape[2:]

        frames = frames.reshape(B, T, 3, H, W)
        flows_f_b_reshaped = flows_f_b.reshape(B, 2 * (T - 1), 2, H, W)
        flows_b = flows_f_b_reshaped[:, T - 1 :]  # (B, T-1, 2, H, W)

        # color augmentation
        # (B, T, 3, H, W)
        frames_forward, frames_backward = self.color_augment(frames)
        flows_b = torch.flip(flows_b, dims=[1])  # (B, T-1, 2, H, W)
        flows_b = flows_b.reshape(B * (T - 1), 2, H, W)  # (B*T-1, 2, H, W)

        last_frame_occluded = False
        if random.randint(0, 1) == 1:
            last_frame_occluded = True

        frames_forward = frames_forward.reshape(B * T, 3, H, W)
        frames_backward = frames_backward.reshape(B * T, 3, H, W)

        extra_tensors_forward = None
        
        if self.center_crop_2 is not None:
            if self.resize_crop_2 is not None:
                frames_forward, flows_f_b, extra_tensors_forward = self.resize_crop_2(
                frames_forward, flows_f_b, extra_tensors_forward
                )
            frames_forward, flows_f_b, extra_tensors_forward = self.center_crop_2(frames_forward, flows_f_b, extra_tensors_forward)

        (
            frames_forward,
            flows_f_b,
            extra_tensors_forward,
            affine_mat_forward,
        ) = self.random_resized_crop_transform(frames_forward, flows_f_b, extra_tensors_forward)

        extra_tensors_backward = None

        if self.center_crop_2 is not None:
            if self.resize_crop_2 is not None:
                frames_backward, flows_b, extra_tensors_backward = self.resize_crop_2(
                    frames_backward, flows_b, extra_tensors_backward
                )
            frames_backward, flows_b, extra_tensors_backward = self.center_crop_2(frames_backward, flows_b, extra_tensors_backward)
        (
            frames_backward,
            flows_b,
            extra_tensors_backward,
            affine_mat_backward,
        ) = self.random_resized_crop_transform(frames_backward, flows_b, extra_tensors_backward)

        H, W = frames_forward.shape[2:]

        frames_forward = frames_forward.reshape(B, T, 3, H, W)
        frames_backward = frames_backward.reshape(B, T, 3, H, W)
        flows_f_b = flows_f_b.reshape(B, 2 * (T - 1), 2, H, W)
        flows_b = flows_b.reshape(B, T - 1, 2, H, W)

        flows_f = flows_f_b[:, : T - 1]
        forward_flows_b = flows_f_b[:, T - 1 :]

        # frames: B, 2*(T-1), 3, H, W
        frames = torch.cat((frames_forward, frames_backward), dim=1)
              
        frames = frames.reshape(B, 2 * T, 3, H, W)

        affine_mat_backward_inv = torch.inverse(affine_mat_backward[0])
        affine_mat_f2b = torch.matmul(affine_mat_forward[0], affine_mat_backward_inv)
        affine_mat_f2b = affine_mat_f2b[:2]

        affine_mat_forward_inv = torch.inverse(affine_mat_forward[0])
        affine_mat_b2f = torch.matmul(affine_mat_backward[0], affine_mat_forward_inv)
        affine_mat_b2f = affine_mat_b2f[:2]

        affine_mat_f2b = affine_mat_f2b[None].repeat(B, 1, 1)
        affine_mat_b2f = affine_mat_b2f[None].repeat(B, 1, 1)

        # H, W: img_size2
        # frames: (B, 2*T, 3, H, W), float, 0-1
        # flows_f: (B, T-1, 2, H, W), float
        # flows_b: (B, T-1, 2, H, W), float
        
        return (
            frames,
            affine_mat_f2b,
            affine_mat_b2f,
            flows_f,
            flows_b,
            forward_flows_b,
        )

    def __iter__(self):
        for i, item in enumerate(self.dataset):
            (
                frames,
                affine_mat_f2b,
                affine_mat_b2f,
                forward_flow,
                backward_flow,
                forward_flows_b,
            ) = self.preprocess_data(item)
            new_item = (
                frames,
                affine_mat_f2b,
                affine_mat_b2f,
                forward_flow,
                backward_flow,
                forward_flows_b,
            )
            yield new_item

    def __len__(self):
        return self.sharded_data_len // self.batch_size
