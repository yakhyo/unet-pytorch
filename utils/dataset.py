import os
import random
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils import data

from utils.general import Augmentation


class Carvana(data.Dataset):
    def __init__(self, root: str, scale: float = 0.5, transforms: Augmentation = Augmentation()) -> None:
        self.root = root
        self.scale = scale

        self.images_dir = os.path.join(self.root, "train_images")
        self.labels_dir = os.path.join(self.root, "train_masks")
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(self.images_dir)]

        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        image_path = os.path.join(self.images_dir, f"{filename}.jpg")
        mask_path = os.path.join(self.labels_dir, f"{filename}.png")

        # image load
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # resize
        image, mask = self.preprocess(image, mask, scale=self.scale)

        assert (image.size == mask.size), f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    @staticmethod
    def preprocess(image, mask, scale):
        w, h = image.size
        newW, newH = int(scale * w), int(scale * h)

        # Resizing the image using BICUBIC interpolation for smooth results
        image = image.resize((newW, newH), Image.BICUBIC)
        # Resizing the mask using NEAREST interpolation to maintain label integrity
        mask = mask.resize((newW, newH), Image.NEAREST)

        return image, mask
