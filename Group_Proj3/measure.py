"""Pixel metrics for binary segmentation, optionally restricted to an FOV."""

import torch


def _counts(logits, target, valid_region=None, threshold=0.5):
    prediction = torch.sigmoid(logits) >= threshold
    target = target >= 0.5
    if valid_region is None:
        valid_region = torch.ones_like(target, dtype=torch.bool)
    else:
        valid_region = valid_region >= 0.5
    prediction = prediction & valid_region
    target = target & valid_region
    return (
        (prediction & target).sum().float(),
        (prediction & ~target & valid_region).sum().float(),
        (~prediction & target & valid_region).sum().float(),
        ((~prediction) & (~target) & valid_region).sum().float(),
    )


def segmentation_metrics(logits, target, valid_region=None, threshold=0.5,
                          smooth=1e-7):
    tp, fp, fn, tn = _counts(logits, target, valid_region, threshold)
    return {
        "dice": (2 * tp + smooth) / (2 * tp + fp + fn + smooth),
        "iou": (tp + smooth) / (tp + fp + fn + smooth),
        "accuracy": (tp + tn + smooth) / (tp + tn + fp + fn + smooth),
        "sensitivity": (tp + smooth) / (tp + fn + smooth),
        "specificity": (tn + smooth) / (tn + fp + smooth),
    }
