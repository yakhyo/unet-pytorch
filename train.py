import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from unet.models import UNet
from unet.optim import RMSprop
from unet.loss import CrossEntropyLoss
from unet.scheduler import PlateauLRScheduler
from unet.utils import Dataset, dice_loss, multiclass_dice_coeff, dice_coeff


def train(args):
    os.makedirs("weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=args.classes)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))

    model.to(device=device)

    # 1. Create dataset
    dataset = Dataset(root="./data", image_size=512)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_data, test_data = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=8, drop_last=True, pin_memory=True)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = PlateauLRScheduler(optimizer, mode='max', patience_t=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = CrossEntropyLoss()

    # 5. Begin training
    best_score = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for idx, batch in enumerate(train_loader):
            images = batch["image"]
            true_masks = batch["mask"]

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=args.amp):
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(F.softmax(masks_pred, dim=1).float(),
                                  F.one_hot(true_masks, model.out_channels).permute(0, 3, 1, 2).float(),
                                  multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            if idx % 10 == 0:
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                print(
                    f"Epoch: {epoch}/{args.epochs} [{idx:>4d}/{len(train_loader)}] Loss: {loss.item():>4f} Mem: {mem}")

        val_score = evaluate(model, test_loader, device)
        print("Dice score:", val_score)
        scheduler.step(epoch)
        torch.save(model.half().state_dict(), f"weights/last.pth")
        if best_score < val_score:
            best_score = max(best_score, val_score)
            torch.save(model.half().state_dict(), f"weights/best.pth")


def evaluate(model, dataloader, device):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for idx, batch in enumerate(dataloader):
        image, mask_true = batch['image'], batch['mask']
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
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

        if idx % 10 == 0:
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            print(f'Evaluating [{idx:>4d}/{len(dataloader)}] Mem: {mem}')

    model.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    dice_score /= num_val_batches
    return dice_score.item()


def get_args():
    parser = argparse.ArgumentParser(description="UNet training code")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--load", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes")

    return parser.parse_args()


if __name__ == '__main__':
    params = get_args()

    train(params)
