"""Pixel metrics for binary segmentation, optionally restricted to an FOV."""

import torch


def confusion_counts(logits, target, valid_region=None, threshold=0.5):
    """Return integer [TP, FP, FN, TN] counts within the valid region."""
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    prediction = torch.sigmoid(logits) >= threshold
    target = target >= 0.5
    if valid_region is None:
        valid_region = torch.ones_like(target, dtype=torch.bool)
    else:
        valid_region = valid_region.expand_as(target) >= 0.5
    return torch.stack((
        (prediction & target & valid_region).sum(),
        (prediction & ~target & valid_region).sum(),
        (~prediction & target & valid_region).sum(),
        (~prediction & ~target & valid_region).sum(),
    ))


def segmentation_metrics(logits, target, valid_region=None, threshold=0.5,
                          smooth=1e-7):
    counts = confusion_counts(logits, target, valid_region, threshold)
    return metrics_from_counts(counts, smooth=smooth)


def metrics_from_counts(counts, smooth=1e-7):
    """Calculate the five metrics from pooled [TP, FP, FN, TN] counts.

    The existing smoothing convention gives 1 for an empty denominator.
    Integer counts are converted to float64 only for the final arithmetic.
    """
    tp, fp, fn, tn = counts.to(dtype=torch.float64).unbind()
    return {
        "dice": (2 * tp + smooth) / (2 * tp + fp + fn + smooth),
        "iou": (tp + smooth) / (tp + fp + fn + smooth),
        "accuracy": (tp + tn + smooth) / (tp + tn + fp + fn + smooth),
        "sensitivity": (tp + smooth) / (tp + fn + smooth),
        "specificity": (tn + smooth) / (tn + fp + smooth),
    }
