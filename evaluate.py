import torch
import torch.nn.functional as F

from unet.utils.loss import DiceLoss


@torch.inference_mode()
def evaluate(model, dataloader, device, conf_threshold=0.5):
    model.eval()
    dice_score = 0
    # Calculate Dice Loss without background
    dice_loss = DiceLoss(include_background=False)
    # iterate over the validation set
    for idx, batch in enumerate(dataloader):
        image, mask = batch["image"], batch["mask"]
        image = image.to(device=device, dtype=torch.float32)
        gt_mask = mask.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            pred_mask = model(image)

            # INFO: Dice Score = 1 - Dice Loss
            if model.out_channels == 1:
                pred_mask = F.sigmoid(pred_mask) > conf_threshold

            dice_score += 1 - dice_loss(pred_mask, gt_mask)

        if idx % 10 == 0:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
            print(f"Evaluating [{idx:>4d}/{len(dataloader)}] Mem: {mem}")

    model.train()

    # Fixes a potential division by zero error
    if len(dataloader) == 0:
        return dice_score
    dice_score /= len(dataloader)
    return dice_score.item()
