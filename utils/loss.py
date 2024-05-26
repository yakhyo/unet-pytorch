from typing import Optional


import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils.general import weight_reduce_loss, LossReduction

__all__ = ["DiceLoss", "DiceCELoss"]


def dice_loss(
    inputs: Tensor,
    targets: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "none",
    eps: float = 1e-5
) -> Tensor:
    """Dice loss calculation function
    Args:
        inputs: input tensor
        targets: target tensor
        weight: loss weight
        reduction: reduction mode
        eps: epsilon to avoid zero division
    Returns:
        torch.Tensor
    """
    inputs = F.softmax(inputs, dim=1)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    if inputs.shape != targets.shape:
        raise AssertionError(
            f"Mismatch in shapes: expected inputs of shape {targets.shape}, but got {inputs.shape}."
        )

    # flatten prediction and label tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = torch.sum(inputs * targets)
    denominator = torch.sum(inputs) + torch.sum(targets)

    # calculate the dice loss
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1 - dice_score

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(inputs)
    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, reduction: LossReduction = "mean", loss_weight: Optional[float] = 1.0, eps: float = 1e-5) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, inputs: Tensor, targets: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        loss = self.loss_weight * dice_loss(
            inputs,
            targets,
            weight=weight,
            reduction=self.reduction,
            eps=self.eps,
        )

        return loss


class DiceCELoss(nn.Module):
    """Dice and Cross Entropy Loss"""

    def __init__(
        self,
        reduction: LossReduction = "mean",
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, inputs: Tensor, targets: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        # calculate dice loss
        dice_loss_ = dice_loss(
            inputs,
            targets,
            weight=weight,
            reduction=self.reduction,
            eps=self.eps,
        )
        # calculate cross entropy loss
        cross_entropy = F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)

        # accumulate loss according to given weights
        loss = self.dice_weight * dice_loss_ + cross_entropy * self.ce_weight

        return loss
