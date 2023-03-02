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
            epsilon: Optional[float] = 1e-5,
            activation: Union[ActivationFunction, str] = ActivationFunction.SOFTMAX,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.activation = activation
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        if self.activation == ActivationFunction.SOFTMAX:
            inputs = torch.softmax(inputs, dim=1)
        if self.activation == ActivationFunction.SIGMOID:
            inputs = torch.softmax(inputs, dim=1)

        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)
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
        if self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)

        return loss


class Loss(nn.Module):
    """Cross Entropy and Dice Loss"""

    def __init__(
            self,
            epsilon: Optional[float] = 1e-5,
            activation: Union[ActivationFunction, str] = ActivationFunction.SOFTMAX,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(epsilon, activation, reduction)

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return ce_loss + dice_loss, {"ce": ce_loss, "dl": dice_loss}