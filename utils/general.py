import os
import random
import logging
import numpy as np
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torchvision.transforms import functional as F

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class LossReduction(Enum):
    """Alias for loss reduction"""
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: LossReduction = "mean",
) -> torch.Tensor:
    """Apply element-wise weight and reduce loss.
    Args:
        loss: element-wise loss
        weight: element-wise weight
        reduction: reduction mode
    Returns:
        torch.Tensor
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if reduction == LossReduction.MEAN:
        loss = torch.mean(loss)
    elif reduction == LossReduction.SUM:
        loss = torch.sum(loss)
    elif reduction == LossReduction.NONE:
        return loss

    return loss


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_weight_decay(model, weight_decay=1e-5):
    """Applying weight decay to only weights, not biases"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or isinstance(param, nn.BatchNorm2d) or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.},
            {"params": decay, "weight_decay": weight_decay}]


def strip_optimizers(f: str, save_f: str = None):
    """Strip optimizer from checkpoint 'f' to finalize training and save to 'save_f'"""
    checkpoint = torch.load(f, map_location="cpu")
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']  # remove optimizer
    if 'epoch' in checkpoint:
        del checkpoint['epoch']  # remove epoch info
    if 'lr_scheduler' in checkpoint:
        del checkpoint['lr_scheduler']  # remove lr_scheduler info

    # checkpoint['model'] = checkpoint['model'].half()  # convert model to half precision
    # Convert all tensors in the state dictionary to half precision
    checkpoint['model'] = {k: v.half() for k, v in checkpoint['model'].items()}
    # for p in checkpoint['model'].parameters():
    #     p.requires_grad = False  # set requires_grad to False

    save_f = save_f or f.replace('.pt', '_stripped.pt')  # save as stripped version
    torch.save(checkpoint['model'], save_f)
    file_size_mb = os.path.getsize(save_f) / 1e6  # get file size in MB
    logging.info(f"Optimizer stripped from {f}, saved as {save_f} ({file_size_mb:.1f}MB)")


class Augmentation:
    """Standard Augmentation"""

    def __init__(self, hflip_prop: float = 0.5) -> None:
        transforms = []
        if hflip_prop > 0:
            transforms.append(RandomHorizontalFlip(hflip_prop))
        transforms.extend([
            PILToTensor(),
            ToDtype(dtype=torch.float, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.transforms = Compose(transforms)

    def __call__(self, image, target):
        return self.transforms(image, target)


class PILToTensor:
    """Convert PIL image to torch tensor"""

    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToDtype:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target):
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Compose:
    """Composing all transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
