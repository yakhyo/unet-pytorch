import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm

from models.unet import UNet
from utils.dataset import Carvana
from utils.general import strip_optimizers, random_seed, add_weight_decay
from utils.loss import DiceCELoss, DiceLoss


# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="UNet training arguments")

    # Data parameters
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Directory containing the dataset (default: './data')"
    )
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for input image size (default: 0.5)")

    # Model parameters
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes (default: 2)")
    parser.add_argument("--weights", type=str, default="", help="Path to pretrained model weights (default: '')")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: 8)"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-8, help="Weight decay (default: 1e-8)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (default: 0.9)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Frequency of printing training progress (default: 10)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume training from (default: '')"
    )

    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only."
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="weights",
        help="Directory to save model weights (default: 'weights')"
    )

    args = parser.parse_args()
    return args


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    batch_loss = []

    for batch_idx, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image = image.to(device)
        target = target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_loss.append(loss.item())

        # Print and log progress at specified frequency
        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Train: [{epoch:>3d}][{batch_idx + 1:>4d}/{len(data_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Time: {(time.time() - start_time):.3f}s "
                f"LR: {lr:.7f}"
            )
    logging.info(f"Avg batch loss: {np.mean(batch_loss):.7f}")


@torch.inference_mode()
def evaluate(model, data_loader, device, conf_threshold=0.5):
    model.eval()
    dice_score = 0
    criterion = DiceLoss()

    for image, target in tqdm(data_loader, total=len(data_loader)):
        image, target = image.to(device), target.to(device)
        output = model(image)

        if model.num_classes == 1:
            output = F.sigmoid(output) > conf_threshold

        dice_loss = criterion(output, target)
        dice_score += 1 - dice_loss.item()  # Ensure dice_loss is a scalar

    average_dice_score = dice_score / len(data_loader)

    return average_dice_score, dice_loss.item()


def main(params):
    random_seed()

    # Create folder to save weights
    os.makedirs(params.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: [{device}]")

    if params.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Dataset
    dataset = Carvana(root=params.data, scale=params.scale)

    # Split train and validation (test)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_data, test_data = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        drop_last=True,
        pin_memory=True
    )

    model = UNet(in_channels=3, num_classes=params.num_classes)
    model.to(device)

    # Optimizers & LR Scheduler & Mixed Precision & Loss
    parameters = add_weight_decay(model)
    optimizer = torch.optim.RMSprop(
        parameters,
        lr=params.lr,
        weight_decay=params.weight_decay,
        momentum=params.momentum,
        foreach=True
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    scaler = torch.cuda.amp.GradScaler() if params.amp else None
    criterion = DiceCELoss()

    start_epoch = 0
    if params.resume:
        checkpoint = torch.load(f"{params.resume}", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    logging.info(
        f"Network: [UNet]:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.num_classes} output channels (number of classes)"
    )
    summary(model, (3, int(1928*params.scale), int(1280*params.scale)))

    for epoch in range(start_epoch, params.epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            lr_scheduler,
            device,
            epoch,
            print_freq=params.print_freq,
            scaler=scaler
        )
        dice_score, dice_loss = evaluate(model, test_loader, device)
        lr_scheduler.step(dice_score)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        logging.info(f"Dice Score: {dice_score:.7f} | Dice Loss: {dice_loss:.7f}")
        if params.amp:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, f"{params.save_dir}/checkpoint.pth")

    # Strip optimizers & save weights
    strip_optimizers(f"{params.save_dir}/checkpoint.pth", save_f=f"{params.save_dir}/last.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
