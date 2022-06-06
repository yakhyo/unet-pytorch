import torch
import torch.nn as nn


def _pad(kernel_size, dilation):
    """ Padding mode = `same` """
    padding = (kernel_size - 1) // 2 * dilation
    return padding


class DConv(nn.Module):
    """ Double Convolutional Layer """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 ) -> None:
        super(DConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=_pad(kernel_size, dilation) if padding is None else padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias,
                      ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=_pad(kernel_size, dilation) if padding is None else padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias,
                      ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UNet(nn.Module):
    """ UNet Segmentation Model """

    def __init__(self, in_channels: int, out_channels: int, init_weight: bool = True) -> None:
        super(UNet, self).__init__()

        self.input_conv = DConv(in_channels, out_channels=64)

        # Down-scaling
        self.down1 = nn.MaxPool2d(kernel_size=2)  # P/2
        self.down1_conv = DConv(in_channels=64, out_channels=128)

        self.down2 = nn.MaxPool2d(kernel_size=2)  # P/4
        self.down2_conv = DConv(in_channels=128, out_channels=256)

        self.down3 = nn.MaxPool2d(kernel_size=2)  # P/8
        self.down3_conv = DConv(in_channels=256, out_channels=512)

        self.down4 = nn.MaxPool2d(kernel_size=2)  # P/16
        self.down4_conv = DConv(in_channels=512, out_channels=1024)

        # Up-scaling
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up1_conv = DConv(in_channels=1024, out_channels=512)

        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up2_conv = DConv(in_channels=512, out_channels=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up3_conv = DConv(in_channels=256, out_channels=128)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up4_conv = DConv(in_channels=128, out_channels=64)

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.input_conv(x)

        x1 = self.down1(x0)
        x1 = self.down1_conv(x1)

        x2 = self.down2(x1)
        x2 = self.down2_conv(x2)

        x3 = self.down3(x2)
        x3 = self.down3_conv(x3)

        x4 = self.down4(x3)
        x4 = self.down4_conv(x4)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1_conv(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, x0], dim=1)
        x = self.up4_conv(x)

        x = self.output_conv(x)

        return x


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=2)
    dummy = torch.randn(1, 3, 256, 256)
    print(model(dummy).shape)
    params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params(model))
