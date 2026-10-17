"""Video classification models.

All 2D models share an ImageNet-pretrained ResNet-18 backbone so the comparison
isolates *where* temporal information is fused. Inputs are [B, T, C, H, W].
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18

NUM_CLASSES = 10
N_FRAMES = 10


def _resnet18(pretrained=True):
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)


def _head(in_features, dropout=0.5):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES))


def _widen_first_conv(net, in_channels):
    """Replace conv1 to take `in_channels` inputs (cross-modality init, Wang et al. 2016):
    average the pretrained RGB filters and replicate them over the new channels."""
    w = net.conv1.weight.data                                   # [64, 3, 7, 7]
    conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv.weight.data = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * 3 / in_channels
    net.conv1 = conv
    return net


class PerFrame(nn.Module):
    """Single-frame classifier. Trained on individual frames; at test time the
    softmax scores of all frames of a video are averaged (aggregation)."""
    kind = 'frame'

    def __init__(self):
        super().__init__()
        self.net = _resnet18()
        self.net.fc = _head(512)

    def forward(self, x):                      # x: [B, 3, H, W] or [B, T, 3, H, W]
        if x.dim() == 4:
            return self.net(x)
        B, T = x.shape[:2]
        return self.net(x.flatten(0, 1)).view(B, T, -1)   # per-frame logits


class LateFusion(nn.Module):
    """Shared 2D CNN applied to every frame; per-frame features are concatenated
    (keeping temporal order) and fused only in the final classifier."""
    kind = 'video'

    def __init__(self):
        super().__init__()
        self.net = _resnet18()
        self.net.fc = nn.Identity()
        self.fc = _head(512 * N_FRAMES)

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.net(x.flatten(0, 1)).view(B, T * 512)
        return self.fc(f)


class EarlyFusion(nn.Module):
    """All frames stacked along the channel axis (3T channels), so temporal
    information is merged already in the first convolution."""
    kind = 'video'

    def __init__(self, in_per_frame=3, n_frames=N_FRAMES):
        super().__init__()
        self.net = _widen_first_conv(_resnet18(), in_per_frame * n_frames)
        self.net.fc = _head(512)

    def forward(self, x):                      # [B, T, C, H, W] -> [B, T*C, H, W]
        return self.net(x.flatten(1, 2))


class CNN3D(nn.Module):
    """R3D-18 (3x3x3 convolutions throughout). `inflate=True` initialises it from the
    ImageNet ResNet-18 by inflating each 2D kernel over time (I3D, Carreira & Zisserman)."""
    kind = 'video'

    def __init__(self, inflate=False):
        super().__init__()
        self.net = r3d_18(weights=None)
        if inflate:
            self._inflate_from_2d()
        self.net.fc = _head(512)

    def _inflate_from_2d(self):
        sd2 = _resnet18().state_dict()
        sd3 = self.net.state_dict()
        k2 = [k for k in sd2 if not k.startswith('fc') and 'num_batches' not in k]
        k3 = [k for k in sd3 if not k.startswith('fc') and 'num_batches' not in k]
        assert len(k2) == len(k3)
        new = {}
        for a, b in zip(k2, k3):
            w2, w3 = sd2[a], sd3[b]
            if w3.dim() == 5:                              # conv: [O, I, t, h, w]
                t = w3.shape[2]
                assert w2.shape == w3.shape[:2] + w3.shape[3:], (a, b)
                new[b] = w2.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
            else:
                assert w2.shape == w3.shape, (a, b)
                new[b] = w2
        self.net.load_state_dict(new, strict=False)

    def forward(self, x):                      # [B, T, C, H, W] -> [B, C, T, H, W]
        return self.net(x.transpose(1, 2))


class FlowStream(EarlyFusion):
    """Temporal stream of the two-stream network: 9 stacked flow fields
    (2 x 9 = 18 channels) into a 2D ResNet-18."""
    def __init__(self):
        super().__init__(in_per_frame=2, n_frames=N_FRAMES - 1)


MODELS = {
    'perframe': PerFrame,
    'late': LateFusion,
    'early': EarlyFusion,
    '3d_scratch': lambda: CNN3D(inflate=False),
    '3d_inflated': lambda: CNN3D(inflate=True),
    'flow': FlowStream,
}
