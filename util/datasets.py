# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os
import warnings

from torch.utils.data.dataset import ConcatDataset, Dataset
warnings.simplefilter("ignore", UserWarning)
import PIL
import torch
import random
import math
import numpy as np
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from PIL import ImageFilter, ImageOps
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
import albumentations as A


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        cutout_size (int): The length (in pixels) of each square patch.
    """
    def __init__(self, cutout_num, cutout_size):
        self.cutout_num = cutout_num
        self.cutout_size = cutout_size

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with cutout_num of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = torch.ones(h, w).to(torch.float32)
        
        for n in range(self.cutout_num):
            # y = np.random.randint(h)
            # x = np.random.randint(w)

            y = torch.randint(0, h, size=(1,)).item()
            x = torch.randint(0, w, size=(1,)).item()

            y1 = np.clip(y - self.cutout_size // 2, 0, h)
            y2 = np.clip(y + self.cutout_size // 2, 0, h)
            x1 = np.clip(x - self.cutout_size // 2, 0, w)
            x2 = np.clip(x + self.cutout_size // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=1.0, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'validate') #'val'
    
    dataset = datasets.ImageFolder(root, transform=transform)
    
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',   # 'bicubic'
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        # crop_pct = 224 / 256
        crop_pct = 0.95
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        # transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_params(img, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    height, width = img.size
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def check_xy_in_region(xy, region):
    x, y = xy
    region_x = region[:, 0].numpy()
    region_y = region[:, 1].numpy()
    x_index = np.where(region_x==x)[0].tolist()
    y_index = np.where(region_y==y)[0].tolist()
    flag = False
    for idx in list(set(x_index) & set(y_index)):
        x_, y_ = region[idx]
        if x == x_ and y == y_:
            flag = True
    return flag


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')




class BaseCrossModalIMDataset(datasets.ImageFolder):
    """
    Dataset for Cross-Modal Image Modeling
    Args:
        data_path (str): Path to the dataset
        data_path_ (str): Path to the dataset
        img_size (int): Size of the image
        transform (dict): Transformations for image
    """
    def __init__(
        self,
        data_path,
        data_path_,
        img_size=224,
        transform=None,
    ):
        super(BaseCrossModalIMDataset, self).__init__(data_path, data_path_)
        self.data_path = data_path
        self.data_path_ = data_path_
        self.img_size = img_size
        self.transform = transform # should be enabled for dual image inputs (Albumentations)
        
        classes, class_to_idx = self.find_classes(self.data_path)
        extensions = ["jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif"]
        self.samples = self.make_dataset(self.data_path, class_to_idx, extensions)
        self.samples_ = self.make_dataset(self.data_path_, class_to_idx, extensions)
    
    def __getitem__(self, index):
        path, _ = self.samples[index]
        path_, _ = self.samples_[index]
        images, images_ = self.loader(path), self.loader(path_)
        if self.transform is not None:
            images, images_ = self.transform(images=images, images_=images_)
        return images, images_

    def __len__(self):
        return len(self.samples)

class CrossModalMIMDataset_(BaseCrossModalIMDataset):
    """
    Dataset for Cross-Modal Correlation Image Modeling
    Args:
        data_path (str): Path to the dataset
        data_path_ (str): Path to the dataset
        img_size (int): Size of the image
        transform (dict): Transformations for image
    """
    def __init__(
        self,
        data_path,
        data_path_,
        *args,
        **kwargs
    ):
        super(CrossModalMIMDataset_, self).__init__(data_path, data_path_, *args, **kwargs)
    
    def __getitem__(self, index):
        path, _ = self.samples[index]
        path_, _ = self.samples_[index]
        images, images_ = self.loader(path), self.loader(path_)
        if self.transform is not None:
            images, images_ = self.transform(images=images, images_=images_)
        return images, images_
    
    def __len__(self):
        return len(self.samples)
    


