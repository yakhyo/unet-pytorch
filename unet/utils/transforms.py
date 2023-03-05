import random
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as F


class Augmentation:
    """Standard Augmentation"""

    def __init__(
            self,
            hflip_prob: float = 0.5,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        transforms = []
        if hflip_prob > 0:
            transforms.append(RandomHorizontalFlip(hflip_prob))
        transforms.extend(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


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


class PILToTensor:
    """Convert PIL image to torch tensor"""

    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    """Convert Image dtype"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    """Normalize the input"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
