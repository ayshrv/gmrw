from abc import abstractmethod

import kornia
import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F


class BaseTransform(torch.nn.Module):
    @abstractmethod
    def __call__(self, images, flows, extra_tensors=None):
        pass


class ToTensor(BaseTransform):
    """Converts a 4D numpy.ndarray or a list of 3D numpy.ndarrays into a
    4D torch.Tensor. Unlike the torchvision implementation, no normalization is
    applied to the values of the inputs.
    Args:
        images_order: str: optional, default 'HWC'
            Must be one of {'CHW', 'HWC'}. Indicates whether the input images
            have the channels first or last.
        flows_order: str: optional, default 'HWC'
            Must be one of {'CHW', 'HWC'}. Indicates whether the input flows
            have the channels first or last.
        fp16: bool: optional, default False
            If True, the tensors use have-precision floating point.
        device: str or torch.device: optional, default 'cpu'
            Name of the torch device where the tensors will be put in.
    """

    def __init__(
        self,
        images_order="HWC",
        flows_order="HWC",
        extra_tensors_order="HWC",
        convert_to_float=False,
        device="cpu",
    ):
        self.images_order = images_order.upper()
        self.flows_order = flows_order.upper()
        self.extra_tensors_order = extra_tensors_order.upper()
        self.convert_to_float = convert_to_float
        self.device = device

    def __call__(
        self,
        images,
        flows=None,
        extra_tensors=None,
    ):
        if isinstance(images, list) or isinstance(images, tuple):
            images = np.stack(images, axis=0)
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)
        if self.images_order == "HWC":
            images = images.permute(0, 3, 1, 2)
        if self.convert_to_float:
            images = images.float()
        images = images.to(device=self.device)

        if flows is not None:
            if isinstance(flows, list) or isinstance(flows, tuple):
                flows = np.stack(flows, axis=0)
            if not isinstance(flows, torch.Tensor):
                flows = torch.from_numpy(flows)
            if self.flows_order == "HWC":
                flows = flows.permute(0, 3, 1, 2)
            if self.convert_to_float:
                flows = flows.float()
            flows = flows.to(device=self.device)

        if extra_tensors:
            assert isinstance(extra_tensors, list)

            new_extra_tensors = []
            for ten in extra_tensors:
                if isinstance(ten, list) or isinstance(ten, tuple):
                    ten = np.stack(ten, axis=0)
                if not isinstance(ten, torch.Tensor):
                    ten = torch.from_numpy(ten)
                if self.extra_tensors_order == "HWC":
                    ten = ten.permute(0, 3, 1, 2)
                if self.convert_to_float:
                    ten = ten.float()
                ten = ten.to(device=self.device)
                new_extra_tensors.append(ten)
        else:
            new_extra_tensors = None

        return images, flows, new_extra_tensors


class Compose(BaseTransform):
    """Similar to torchvision Compose. Applies a series of transforms from
    the input list in sequence.
    Args:
        transforms_list: Sequence[BaseTransform]:
            A sequence of transforms to be applied.
    """

    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, images, flows=None, extra_tensors=None):
        for t in self.transforms_list:
            images, flows, extra_tensors = t(images, flows, extra_tensors)
        return images, flows, extra_tensors


class RandomResizedCrop(BaseTransform):
    """
    args format same as kornia.augmentation.RandomResizedCrop
    """

    def __init__(
        self,
        size,
        scale,
        ratio,
        seed=786234,
    ):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        rng = torch.manual_seed(seed)
        self.k_aug_transform = K.RandomResizedCrop(
            size=self.size,
            scale=self.scale,
            ratio=self.ratio,
            same_on_batch=True,
            return_transform=True,
        )
        self.aug_params = None

    def _edit_aug_params(self, old_aug_params, new_shape):
        new_B = new_shape[0]
        # assert old_aug_params["src"].shape[0] >= new_B

        new_aug_params = {}
        new_aug_params["forward_input_shape"] = new_shape
        new_aug_params["src"] = old_aug_params["src"][:1].repeat(new_B, 1, 1)
        new_aug_params["dst"] = old_aug_params["dst"][:1].repeat(new_B, 1, 1)
        new_aug_params["input_size"] = old_aug_params["input_size"][:1].repeat(new_B, 1)
        new_aug_params["batch_prob"] = old_aug_params["batch_prob"][:1].repeat(new_B)
        return new_aug_params

    def __call__(
        self,
        images,
        flows=None,
        extra_tensors=None,
    ):

        B, C, H, W = images.shape

        images, transmat = self.k_aug_transform(images)
        self.aug_params = self.k_aug_transform._params

        if flows is not None:
            aug_params = self._edit_aug_params(self.aug_params, flows.shape)
            flows, _ = self.k_aug_transform(flows, params=aug_params)

            top_left_coord = aug_params["src"][:, 0]  # (B, 2)
            top_right_coord = aug_params["src"][:, 1]  # (B, 2)
            bottom_right_coord = aug_params["src"][:, 2]  # (B, 2)
            bottom_left_coord = aug_params["src"][:, 3]  # (B, 2)
            scale_w = self.size[0] / (
                top_right_coord[:, 0] - top_left_coord[:, 0]
            )  # (B, ) self.size[0]
            scale_h = self.size[1] / (
                bottom_right_coord[:, 1] - top_right_coord[:, 1]
            )  # (B, ) self.size[1]
            scale_w = scale_w[:, None, None].repeat(1, flows.shape[2], flows.shape[3])
            scale_h = scale_h[:, None, None].repeat(1, flows.shape[2], flows.shape[3])

            flows[:, 0] = flows[:, 0] * scale_w
            flows[:, 1] = flows[:, 1] * scale_h
        else:
            flows = None

        if extra_tensors is not None:
            assert isinstance(extra_tensors, list)
            new_extra_tensors = []
            for ten in extra_tensors:
                aug_params = self._edit_aug_params(self.aug_params, ten.shape)
                ten, _ = self.k_aug_transform(ten, params=aug_params)
                new_extra_tensors.append(ten)
        else:
            new_extra_tensors = None

        transmat = kornia.geometry.conversions.normalize_homography(
            transmat, (H, W), self.size
        )
        return images, flows, new_extra_tensors, transmat


class CenterCrop(BaseTransform):
    def __init__(
        self,
        size,
    ):
        super().__init__()
        self.size = size
        self.k_aug_transform = K.CenterCrop(
            size=self.size, align_corners=True, return_transform=False
        )

    def __call__(self, images, flows=None, extra_tensors=None):
        images = self.k_aug_transform(images)
        if flows is not None:
            flows = self.k_aug_transform(flows)
        else:
            flows = None

        if extra_tensors is not None:
            assert isinstance(extra_tensors, list)
            new_extra_tensors = []
            for ten in extra_tensors:
                ten = self.k_aug_transform(ten)
                new_extra_tensors.append(ten)
        else:
            new_extra_tensors = None

        return images, flows, new_extra_tensors


class Resize(BaseTransform):
    def __init__(
        self,
        size,
    ):
        super().__init__()
        self.size = size
        self.k_aug_transform = K.Resize(size=self.size, return_transform=False)

    def __call__(self, images, flows=None, extra_tensors=None):
        _, _, old_H, old_W = images.shape
        images = self.k_aug_transform(images)
        _, _, new_H, new_W = images.shape
        if flows is not None:
            flows = self.k_aug_transform(flows)
            scale_w = new_W / old_W
            scale_h = new_H / old_H
            flows[:, 0] = flows[:, 0] * scale_w
            flows[:, 1] = flows[:, 1] * scale_h
        else:
            flows = None

        if extra_tensors is not None:
            assert isinstance(extra_tensors, list)
            new_extra_tensors = []
            for ten in extra_tensors:
                ten = self.k_aug_transform(ten)
                new_extra_tensors.append(ten)
        else:
            new_extra_tensors = None

        return images, flows, new_extra_tensors
