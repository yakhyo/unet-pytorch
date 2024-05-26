from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Standard Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 1,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            inplace: bool = True,
            bias: bool = False,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    """Double Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.conv2 = ConvBNReLU(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Down(nn.Module):
    """Feature Downscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor=2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    """Feature Upscale"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=out_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2
            )
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """UNet Segmentation Model"""

    def __init__(self, in_channels: int, num_classes: int, bilinear: bool = False) -> None:
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.input_conv = DoubleConv(in_channels, out_channels=64)

        factor = 2 if bilinear else 1

        # Downscale ⬇️
        self.down1 = Down(in_channels=64, out_channels=128, scale_factor=2)  # P/2
        self.down2 = Down(in_channels=128, out_channels=256, scale_factor=2)  # P/4
        self.down3 = Down(in_channels=256, out_channels=512, scale_factor=2)  # P/8
        self.down4 = Down(in_channels=512, out_channels=1024 // factor, scale_factor=2)  # P/16

        # Upscale ⬆️
        self.up1 = Up(in_channels=1024, out_channels=512 // factor, bilinear=bilinear)
        self.up2 = Up(in_channels=512, out_channels=256 // factor, bilinear=bilinear)
        self.up3 = Up(in_channels=256, out_channels=128 // factor, bilinear=bilinear)
        self.up4 = Up(in_channels=128, out_channels=64, bilinear=bilinear)

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.input_conv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x_ = self.up1(x4, x3)
        x_ = self.up2(x_, x2)
        x_ = self.up3(x_, x1)
        x_ = self.up4(x_, x0)

        x_ = self.output_conv(x_)

        return x_
