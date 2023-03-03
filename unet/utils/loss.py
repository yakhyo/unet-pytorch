from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class ActivationFunction:
    """Alias for activation function"""

    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


class LossReduction:
    """Alias for loss reduction"""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class DiceLoss(nn.Module):
    def __init__(
            self,
            include_background: Optional[bool] = True,
            epsilon: Optional[float] = 1e-5,
            activation: Union[ActivationFunction, str] = ActivationFunction.SOFTMAX,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.epsilon = epsilon
        self.activation = activation
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        if self.activation == ActivationFunction.SOFTMAX:
            inputs = torch.softmax(inputs, dim=1)
        if self.activation == ActivationFunction.SIGMOID:
            inputs = torch.softmax(inputs, dim=1)

        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

        if not self.include_background:
            if inputs.shape[1] == 1:
                raise Warning("Single channel prediction, `include_background=False` ignored")
            else:
                # if skipping background, removing first channel
                targets = targets[:, 1:]
                inputs = inputs[:, 1:]

        if targets.shape != inputs.shape:
            raise AssertionError(f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})")

        # flatten prediction and label tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = torch.sum(inputs * targets)
        denominator = torch.sum(inputs) + torch.sum(targets)

        # calculate the dice loss
        loss = 1.0 - (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        elif self.reduction == LossReduction.NONE:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(loss.shape[0:2]) + [1] * (len(inputs.shape) - 2)
            loss = loss.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return loss


class Loss(nn.Module):
    """Cross Entropy and Dice Loss"""

    def __init__(
            self,
            include_background: Optional[bool] = True,
            epsilon: Optional[float] = 1e-5,
            activation: Union[ActivationFunction, str] = ActivationFunction.SOFTMAX,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(include_background, epsilon, activation, reduction)

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return ce_loss + dice_loss, {"ce": ce_loss, "dl": dice_loss}


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
