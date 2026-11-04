import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probability = torch.sigmoid(logits)
        probability = probability.flatten(1)
        target = target.flatten(1)
        dice = (2 * (probability * target).sum(1) + self.smooth) / (
            probability.sum(1) + target.sum(1) + self.smooth
        )
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        pt = probability * target + (1 - probability) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, positive_weight=2.0):
        super().__init__()
        self.positive_weight = positive_weight

    def forward(self, logits, target):
        weight = torch.where(
            target > 0.5,
            torch.as_tensor(self.positive_weight, device=logits.device),
            torch.ones_like(target),
        )
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


class PointBCELoss(nn.Module):
    """Mean binary cross-entropy over annotated pixels only; -1 means unknown."""

    def forward(self, logits, points):
        if logits.shape != points.shape:
            raise ValueError("logits and points must have identical shapes")
        if not bool(((points == -1) | (points == 0) | (points == 1)).all()):
            raise ValueError("Point labels must be -1, 0 or 1")
        annotated = points != -1
        selected = logits[annotated]
        if selected.numel() == 0:
            # An empty indexed sum is differentiable zero, even if unrelated
            # unannotated logits contain NaNs. No arbitrary pixel is supervised.
            return selected.sum()
        return F.binary_cross_entropy_with_logits(selected, points[annotated].to(logits.dtype))
