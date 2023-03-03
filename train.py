import argparse
import logging
import os
from copy import deepcopy

import torch
from evaluate import evaluate
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet.models import UNet
from unet.utils.dataset import Carvana
from unet.utils.loss import DiceCELoss


# from unet.utils.general import strip_optimizers


def strip_optimizers(f: str):
    """Strip optimizer from 'f' to finalize training"""
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "best_score":
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, f)
    mb = os.path.getsize(f) / 1e6  # get file size
    logging.info(f"Optimizer stripped from {f}, saved as {f} {mb:.1f}MB")


def train(opt, model, device):
    best_score, start_epoch = 0, 0
    os.makedirs(opt.save_dir, exist_ok=True)
    best, last = f"{opt.save_dir}/best.pt", f"{opt.save_dir}/last.pt"

    # Check pretrained weights
    pretrained = opt.weights.endswith(".pt")
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict())
        logging.info(f"Model ckpt loaded from {opt.weights}")
    model.to(device)

    # Optimizers & LR Scheduler & Mixed Precision & Loss
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = DiceCELoss()

    # Resume
    if pretrained:
        if ckpt["optimizer"] is not None:
            start_epoch = ckpt["epoch"] + 1
            best_score = ckpt["best_score"]
            optimizer.load_state_dict(ckpt["optimizer"])
            logging.info(f"Optimizer loaded from {opt.weights}")
            if start_epoch > opt.epochs:
                logging.info(
                    f"{opt.weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {opt.epochs} epochs"
                )
                opt.epochs += start_epoch
        del ckpt

    # Dataset
    dataset = Carvana(root="./data", image_size=opt.image_size)

    # Split
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_data, test_data = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=8, drop_last=True, pin_memory=True)

    # Training
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        epoch_loss = 0
        logging.info(("\n" + "%12s" * 5) % ("Epoch", "GPU Mem", "CE Loss", "Dice Loss", "Total Loss"))
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for batch in progress_bar:
            images = batch["image"]
            true_masks = batch["mask"]

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=opt.amp):
                masks_pred = model(images)
                loss, losses = criterion(masks_pred, true_masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(
                ("%12s" * 2 + "%12.4g" * 3) % (f"{epoch + 1}/{opt.epochs}", mem, losses["ce"], losses["dl"], loss)
            )

        val_score = evaluate(model, test_loader, device)
        print("Dice score:", val_score)
        scheduler.step(epoch)
        ckpt = {
            "epoch": epoch,
            "best_score": best_score,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, last)
        if best_score < val_score:
            best_score = max(best_score, val_score)
            torch.save(ckpt, best)
    for f in best, last:
        strip_optimizers(f)


def parse_opt():
    parser = argparse.ArgumentParser(description="UNet training arguments")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size, default: 512")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs, default: 5")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size, default: 12")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default: 1e-5")
    parser.add_argument("--weights", type=str, default="", help="Pretrained model, default: None")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")

    return parser.parse_args()


def main(opt):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = UNet(in_channels=3, out_channels=opt.num_classes)

    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )

    train(opt, model, device)


if __name__ == "__main__":
    params = parse_opt()
    main(params)
