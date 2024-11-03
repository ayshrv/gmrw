import logging

import kornia
import kornia.augmentation as K
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

logger = logging.getLogger(__name__)

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])

        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])


def get_frame_transforms(frame_transform_str, img_size):
    transforms_list = []

    size_transform_only = None

    if "random_resized_crop" in frame_transform_str:
        size_transform_only = torchvision.transforms.RandomResizedCrop(
            img_size, scale=(1, 1), ratio=(1, 1), interpolation=2
        )
        transforms_list.append(size_transform_only)
    else:
        size_transform_only = torchvision.transforms.Resize((img_size, img_size))
        transforms_list.append(size_transform_only)

    return transforms_list, size_transform_only


def get_train_transforms(args, norm=True):

    cropped_image_size = args.img_size1 + 2 * args.ghost_nodes_padding

    # RandomResizedCrop
    frame_transform, size_transform_only = get_frame_transforms(
        "random_resized_crop", cropped_image_size
    )

    # if args.img_size1 != args.img_size2:
    #     frame_transform += [torchvision.transforms.CenterCrop(args.img_size2)]

    if norm:
        frame_transform += NORM
    else:
        frame_transform += [transforms.ToTensor()]

    logger.info(f"Train Transforms: {frame_transform}")

    train_transform = MapTransform(torchvision.transforms.Compose(frame_transform))

    size_transform_only = MapTransform(
        torchvision.transforms.Compose(size_transform_only)
    )

    return train_transform, size_transform_only


class RandomResizedCropHFilp(torch.nn.Module):
    def __init__(self, size, aug_scale, aug_ratio, same_on_batch=True, debug=False):
        super().__init__()
        self.size = size

        if debug:
            self.rrcrop = K.CenterCrop(
                size=(self.size, self.size), return_transform=True, keepdim=True
            )
        else:
            self.rrcrop = K.RandomResizedCrop(
                size=(self.size, self.size),
                scale=aug_scale,
                ratio=aug_ratio,
                same_on_batch=same_on_batch,
                return_transform=True,
            )

    def forward(self, x):
        B, C, H, W = x.shape
        x, transmat = self.rrcrop(x)
        transmat = kornia.geometry.conversions.normalize_homography(
            transmat, (H, W), (self.size, self.size)
        )
        return x, transmat


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_color_aug_transform():
    transform_1 = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
        ]
    )
    transform_2 = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
        ]
    )
    transform = (transform_1, transform_2)

    return transform


def get_single_image_aug_transform():
    transform_1 = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
        ]
    )
    return transform_1


