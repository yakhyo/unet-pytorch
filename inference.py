import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torchvision.transforms.functional as F

from models.unet import UNet


class PILToTensor:
    """Convert PIL image to torch tensor"""

    def __call__(self, image):
        image = F.pil_to_tensor(image)
        return image


class ToDtype:
    def __init__(self, dtype, scale=True):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image):
        if self.scale:
            image = F.convert_image_dtype(image, self.dtype)  # Scale the image to [0, 1]
        else:
            image = image.to(dtype=self.dtype)
        return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


class Compose:
    """Composing all transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class InferenceAugmentation:
    """Inference Augmentation"""

    def __init__(self, scale) -> None:
        self.scale = scale
        self.transforms = Compose([
            PILToTensor(),
            ToDtype(dtype=torch.float, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, image):
        image = self.resize(image)
        return self.transforms(image)

    def resize(self, image):
        w, h = image.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        image = image.resize((newW, newH), Image.BICUBIC)
        return image


def resize(image, scale):
    w, h = image.size
    newW, newH = int(scale * w), int(scale * h)
    image = image.resize((newW, newH), Image.BICUBIC)
    return image


def inference(model, device, params):
    # initialize inference augmentation
    preprocess = InferenceAugmentation(scale=params.scale)
    # read image
    input_image = Image.open(params.image_path).convert("RGB")
    # preprocess
    input_tensor = preprocess(input_image)
    # add batch
    input_batch = input_tensor.unsqueeze(0)
    # move to device
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)[0]

    output_predictions = output.argmax(0).cpu().numpy()

    return output_predictions


color_palette = [
    (0, 0, 0),       # Color for Background
    (255, 0, 0),     # Color for Car
]


def visualize_segmentation_map(image, segmentation_mask):
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # Create an RGB image with the same height and width as the segmentation
    h, w = segmentation_mask.shape
    colored_segmentation = np.zeros((h, w, 3), dtype=np.uint8)

    num_classes = np.max(segmentation_mask)

    # Map each class to its respective color
    for class_id, color in enumerate(color_palette):
        colored_segmentation[segmentation_mask == class_id] = color

    # Convert image to BGR format for blending
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend the image with the segmentation mask
    blended_image = cv2.addWeighted(bgr_image, 0.6, colored_segmentation, 0.4, 0)

    return blended_image, colored_segmentation


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Image Segmentation Inference")
    parser.add_argument("--model-path", type=str, default="./weights/last.pt", help="Path to the model weights")
    parser.add_argument("--image-path", type=str, default="assets/image.jpg", help="Path to the input image")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for resizing the image")
    parser.add_argument("--save-overlay", action="store_true", help="Save the overlay image if this flag is set")

    args = parser.parse_args()

    return args


def load_model(params, device):
    # Initialize the model
    model = UNet(in_channels=3, num_classes=2)

    # Load weights and convert to float32 because weights stored in f16
    state_dict = torch.load(params.model_path, map_location=device)
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].float()

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(params, device)

    # inference
    segmentation_map = inference(model, device, params=params)

    filename = os.path.basename(params.image_path)
    dirname = os.path.dirname(params.image_path)

    # save segmentation mask
    segmentation_image = Image.fromarray((segmentation_map * 255).astype(np.uint8))
    segmentation_image.save(f"./assets/{filename[:-4]}_mask.png")

    # save overlay mask on input image and
    if params.save_overlay:
        print("Saving the overlay image.")
        image = Image.open(params.image_path).convert("RGB")
        image = resize(image, params.scale)

        # returns overlayed image and colored mask
        overlayed_image, colored_class_map = visualize_segmentation_map(image, segmentation_map)

        cv2.imwrite(f"./assets/{filename[:-4]}_mask_color.png", colored_class_map)
        cv2.imwrite(f"./assets/{filename[:-4]}_overlay.png", overlayed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    args = parse_args()
    main(params=args)
