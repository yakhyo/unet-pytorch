import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, root, image_size=512, transforms=None, mask_suffix="_mask"):
        self.root = root
        self.image_size = image_size
        self.transforms = transforms
        self.mask_suffix = mask_suffix
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.root, "images"))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        image_path = os.path.join(self.root, f"images{os.sep}{filename}.jpg")
        mask_path = os.path.join(self.root, f"masks{os.sep}{filename + self.mask_suffix}.gif")

        # image load
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        assert image.size == mask.size, f'Image and mask {filename} should be the same size, but are {image.size} and {mask.size}'

        # resize
        image, mask = self.resize_pil(image, mask, image_size=self.image_size)

        # preprocess
        image = self.preprocess(image, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        # to tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return {'image': image, 'mask': mask}

    @staticmethod
    def resize_pil(image, mask, image_size):
        w, h = image.size
        scale = min(image_size / w, image_size / h)

        # resize image
        image = image.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
        mask = mask.resize((int(w * scale), int(h * scale)), resample=Image.NEAREST)

        # pad size
        delta_w = image_size - int(w * scale)
        delta_h = image_size - int(h * scale)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # pad image
        image = ImageOps.expand(image, (left, top, right, bottom))
        mask = ImageOps.expand(mask, (left, top, right, bottom))

        return image, mask

    @staticmethod
    def resize_cv2(image, mask, image_size):
        h, w = image.shape[:2]
        scale = min(image_size / w, image_size / h)

        # resize image
        image = cv2.resize(image, (int(w * scale), int(h * scale)), Image.BILINEAR)
        mask = cv2.resize(mask, (int(w * scale), int(h * scale)), Image.BILINEAR)

        # pad size
        delta_w = image_size - int(w * scale)
        delta_h = image_size - int(h * scale)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # pad image
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return image, mask

    @staticmethod
    def preprocess(image, is_mask):
        img_ndarray = np.asarray(image)
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray


if __name__ == '__main__':
    test = Dataset("./data")
    iterable = iter(test)
    while True:
        pict = next(iterable)
        pict = pict['image'].numpy()
        pict = pict.transpose((1, 2, 0))

        cv2.imshow('Frame', pict)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
