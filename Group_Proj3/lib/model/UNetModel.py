import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.down1 = DoubleConv(in_channels, c)
        self.down2 = DoubleConv(c, c * 2)
        self.down3 = DoubleConv(c * 2, c * 4)
        self.down4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.conv4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.conv3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.conv2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.conv1 = DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, out_channels, 1)

    @staticmethod
    def _merge(skip, upsampled):
        if skip.shape[-2:] != upsampled.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        return torch.cat((skip, upsampled), dim=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b = self.bottleneck(self.pool(d4))
        x = self.conv4(self._merge(d4, self.up4(b)))
        x = self.conv3(self._merge(d3, self.up3(x)))
        x = self.conv2(self._merge(d2, self.up2(x)))
        x = self.conv1(self._merge(d1, self.up1(x)))
        return self.output(x)


UNet2 = UNet
