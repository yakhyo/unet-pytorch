import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet.utils import Dataset
from unet.models import UNet
from unet.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    image, _ = Dataset.resize_pil(full_img, full_img, image_size=512)
    image = torch.from_numpy(Dataset.preprocess(image, is_mask=False))
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(image)

        if net.out_channels > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = transform(probs.cpu()).squeeze()

    if net.out_channels == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.out_channels).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='weights/checkpoint_epoch5.pth', help='Model path')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(in_channels=3, out_channels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    for i, filename in enumerate(in_files):
        image = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=image,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
        image, _ = Dataset.resize_pil(image, image, image_size=512)
        if args.viz:
            plot_img_and_mask(image, mask)
