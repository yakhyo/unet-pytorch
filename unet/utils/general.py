import os
import matplotlib.pyplot as plt
import torch


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


def strip_optimizers(f: str):
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "epoch":
        x[k] = None

    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, f)
    mb = os.path.getsize(f) / 1e6  # filesize
    print(f"Optimizer stripped from {f}, saved as {f} {mb:.1f}MB")
