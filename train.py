import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import Dataset
from utils import dice_loss
from evaluate import evaluate
from unet import UNet


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    dataset = Dataset(root='./data', image_size=512)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_data, test_data = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=8,
                              shuffle=True,
                              pin_memory=True,
                              )
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             num_workers=8,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=True,
                             )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm(enumerate(train_loader), total=len(train_data))
        for idx, batch in progress_bar:
            images = batch['image']
            true_masks = batch['mask']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks)
                dice = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                 F.one_hot(true_masks, net.out_channels).permute(0, 3, 1, 2).float(),
                                 multiclass=True)
                loss += dice

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, epochs), loss.item(), mem)
            progress_bar.set_description(s)

        val_score = evaluate(net, test_loader, device)
        scheduler.step(val_score)

        if save_checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(net.state_dict(), f'checkpoints/checkpoint_epoch{epoch}.pth')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(in_channels=3, out_channels=args.classes)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
