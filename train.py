import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from unet.models import UNet
from unet.utils.dataset import Carvana
from unet.utils.general import strip_optimizers, plot_img_and_mask
from unet.utils.loss import DiceLoss, Loss


def train(args):
    image_size, epochs, batch_size, weights = args.image_size, args.epochs, args.batch_size, args.weights
    best, last = f"{args.save_dir}/best.pt", f"{args.save_dir}/last.pt"
    os.makedirs("weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=args.classes)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # criterion = nn.CrossEntropyLoss()
    # criterion = DiceLoss(model.out_channels)
    criterion = Loss()
    start_epoch = 0

    if weights.endswith(".pt"):
        print(f"[INFO] Loading weights from {weights}...")
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt["model"].float().state_dict())
        # TODO: load the optimizer to GPU is issue?
        # optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt["epoch"]:
            start_epoch = ckpt["epoch"] + 1
        print("[INFO] Loaded successfully")

    model.to(device=device)

    # 1. Create dataset
    dataset = Carvana(root="./data", image_size=args.image_size)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_data, test_data = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=8, drop_last=True, pin_memory=True)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    # 5. Begin training
    best_score = 0
    losses = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        print(('\n' + '%20s' * 5) % ('EPOCH', 'Cross Entropy Loss', 'Dice Loss', 'Total Loss', 'GPU'))
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, batch in progress_bar:
            images = batch["image"]
            true_masks = batch["mask"]

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=args.amp):
                masks_pred = model(images)
                loss, losses = criterion(masks_pred, true_masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%20s' + '%20.4g' + '%20.4g' + '%20.4g' + '%20s') % (
                '%g/%g' % (epoch + 1, args.epochs), losses['ce'], losses['dl'], loss, mem)
            progress_bar.set_description(s)
        val_score = evaluate(model, test_loader, device)
        print("Dice score:", val_score)
        scheduler.step(epoch)
        ckpt = {"epoch": epoch, "optimizer": optimizer.state_dict(), "model": deepcopy(model).half()}
        torch.save(ckpt, last)
        if best_score < val_score:
            best_score = max(best_score, val_score)
            torch.save(ckpt, best)
    for f in best, last:
        strip_optimizers(f)


def evaluate(model, dataloader, device):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for idx, batch in enumerate(dataloader):
        image, mask_true = batch["image"], batch["mask"]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, model.out_channels).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = model(image)

            # convert to one-hot format
            if model.out_channels == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.out_channels).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False
                )

        if idx % 10 == 0:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
            print(f"Evaluating [{idx:>4d}/{len(dataloader)}] Mem: {mem}")

    model.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    dice_score /= num_val_batches
    return dice_score.item()


def get_args():
    parser = argparse.ArgumentParser(description="UNet training code")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size, default: 512")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs, default: 5")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size, default: 12")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default: 1e-5")
    parser.add_argument("--weights", type=str, default="", help="Pretrained model, default: None")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes")

    return parser.parse_args()


if __name__ == "__main__":
    params = get_args()

    train(params)
