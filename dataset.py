import os
import numpy as np
from PIL import Image

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

        # preprocess
        image, mask = preprocess(image, mask)


def preprocess(image, mask):
    w, h = image.size
    print(w, h)
    return image, mask


if __name__ == '__main__':
    test = Dataset("./data")
    _ = next(iter(test))
