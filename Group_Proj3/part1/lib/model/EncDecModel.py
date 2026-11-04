import torch
import torch.nn as nn
import torch.nn.functional as F

class EncDec(nn.Module):
    """Map [B, 3, H, W] to binary logits [B, 1, H, W], with H and W >= 16."""

    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # Restore the recorded spatial sizes. Only shapes are reused here;
        # encoder features are not concatenated into the decoder.
        d0 = F.interpolate(b, size=e2.shape[-2:], mode="nearest")
        d0 = F.relu(self.dec_conv0(d0))
        d1 = F.interpolate(d0, size=e1.shape[-2:], mode="nearest")
        d1 = F.relu(self.dec_conv1(d1))
        d2 = F.interpolate(d1, size=e0.shape[-2:], mode="nearest")
        d2 = F.relu(self.dec_conv2(d2))
        d3 = F.interpolate(d2, size=x.shape[-2:], mode="nearest")
        return self.dec_conv3(d3)  # logits; sigmoid is applied outside the model
