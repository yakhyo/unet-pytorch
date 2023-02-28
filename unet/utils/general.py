import os
from typing import Optional

import matplotlib.pyplot as plt
import torch


def dice_coeff(
        input: torch.Tensor,
        target: torch.Tensor,
        reduce_batch_first: Optional[bool] = False,
        epsilon: Optional[float] = 1e-6,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size(), f"`input`: {input.size()} and `target`: {target.size()} has different size"
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
        input: torch.Tensor,
        target: torch.Tensor,
        reduce_batch_first: Optional[bool] = False,
        epsilon: Optional[float] = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: Optional[bool] = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f"Output mask (class {i + 1})")
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f"Output mask")
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def strip_optimizers(f: str = "weights/last.ckpt"):
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "epoch":
        x[k] = None

    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, f)
    mb = os.path.getsize(f) / 1e6  # filesize
    print(f"Optimizer stripped from {f}, saved as {f} {mb:.1f}MB")
