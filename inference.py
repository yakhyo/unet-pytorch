# TODO: Needs to be fixed
import argparse

import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from unet.models import UNet

from unet.utils.dataset import Carvana
from unet.utils.general import plot_img_and_mask


def resize(image, image_size=512):
    w, h = image.size
    scale = min(image_size / w, image_size / h)

    # resize image
    image = image.resize((int(w * scale), int(h * scale)))

    # pad size
    delta_w = image_size - int(w * scale)
    delta_h = image_size - int(h * scale)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # pad image
    image = ImageOps.expand(image, (left, top, right, bottom))

    return image


def predict_img(model, full_img, device, out_threshold=0.5):
    model.eval()
    image = resize(full_img)
    image = torch.from_numpy(Carvana.preprocess(image, is_mask=False))
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image)

        if model.out_channels > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        full_mask = probs.cpu().squeeze()

    if model.out_channels == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), model.out_channels).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument("--weights", default="weights/best.pt", help="Model path")
    parser.add_argument("--input", required=True, help="Filenames of input images")
    parser.add_argument("--output", default="output.jpg", help="Filenames of output images")
    parser.add_argument("--viz", action="store_true", help="Visualize the images as they are processed")
    parser.add_argument("--no-save", action="store_true", help="Do not save the output masks")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold value")

    return parser.parse_args()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=2)
    ckpt = torch.load(args.weights, map_location=device)["model"].float().state_dict()
    model.load_state_dict(ckpt)
    model.to(device=device)

    for i, filename in enumerate([args.input]):
        image = Image.open(filename)
        mask = predict_img(model=model.float(), full_img=image, out_threshold=args.mask_threshold, device=device)

        result = mask_to_image(mask)
        result.save(args.output)

        if args.viz:
            image = resize(image)
            plot_img_and_mask(image, mask)
