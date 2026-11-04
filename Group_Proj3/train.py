"""Train an encoder-decoder or U-Net on PH2 or DRIVE."""

import argparse
import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.dataset.DRIVEDataset import DRIVEDataset
from lib.dataset.PH2Dataset import PH2Dataset
from lib.losses import BCELoss, FocalLoss, WeightedBCELoss
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.splits import drive_splits, ph2_splits


def build_datasets(dataset_name, size, seed):
    split_ids = ph2_splits(seed) if dataset_name == "PH2" else drive_splits(seed)
    dataset_cls = PH2Dataset if dataset_name == "PH2" else DRIVEDataset
    kwargs = {"image_transform": transforms.ToTensor(), "size": size}
    return tuple(
        dataset_cls(split=split, sample_ids=split_ids[split], **kwargs)
        for split in ("train", "val", "test")
    )


def masked_loss(loss_fn, logits, target, valid_region):
    pixel_mask = valid_region.expand_as(target)
    if isinstance(loss_fn, BCELoss):
        values = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
    elif isinstance(loss_fn, WeightedBCELoss):
        weight = torch.where(
            target > 0.5,
            torch.as_tensor(loss_fn.positive_weight, device=logits.device),
            torch.ones_like(target),
        )
        values = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight, reduction="none"
        )
    elif isinstance(loss_fn, FocalLoss):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        probability = torch.sigmoid(logits)
        pt = probability * target + (1 - probability) * (1 - target)
        alpha_t = loss_fn.alpha * target + (1 - loss_fn.alpha) * (1 - target)
        values = alpha_t * (1 - pt).pow(loss_fn.gamma) * bce
    else:
        return loss_fn(logits, target)
    return (values * pixel_mask).sum() / pixel_mask.sum().clamp_min(1.0)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    totals = {"loss": 0.0, "count": 0}
    from measure import segmentation_metrics
    metrics = {name: 0.0 for name in (
        "dice", "iou", "accuracy", "sensitivity", "specificity"
    )}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            valid = batch["valid_region"].to(device)
            logits = model(images)
            totals["loss"] += masked_loss(loss_fn, logits, targets, valid).item()
            result = segmentation_metrics(logits, targets, valid)
            for name in metrics:
                metrics[name] += result[name].item()
            totals["count"] += 1
    count = max(totals["count"], 1)
    return totals["loss"] / count, {name: value / count for name, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PH2", "DRIVE"), default="PH2")
    parser.add_argument("--model", choices=("encdec", "unet"), default="unet")
    parser.add_argument("--loss", choices=("bce", "focal", "weighted_bce"), default="bce")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, valset, testset = build_datasets(args.dataset, args.size, args.split_seed)
    loaders = [
        DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.num_workers),
        DataLoader(valset, args.batch_size, shuffle=False, num_workers=args.num_workers),
        DataLoader(testset, args.batch_size, shuffle=False, num_workers=args.num_workers),
    ]
    model = (EncDec() if args.model == "encdec" else UNet()).to(device)
    loss_fn = {
        "bce": BCELoss(),
        "focal": FocalLoss(),
        "weighted_bce": WeightedBCELoss(),
    }[args.loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = args.checkpoint or (
        f"checkpoints/{args.dataset.lower()}_{args.model}_{args.loss}.pth"
    )
    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        started = time()
        model.train()
        total = 0.0
        for batch in loaders[0]:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            valid = batch["valid_region"].to(device)
            optimizer.zero_grad()
            loss = masked_loss(loss_fn, model(images), targets, valid)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / max(len(loaders[0]), 1)
        val_loss, val_metrics = evaluate(model, loaders[1], loss_fn, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} dice={val_metrics['dice']:.4f} "
            f"time={time() - started:.1f}s"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    for name, loader in zip(("train", "val", "test"), loaders):
        loss, metrics = evaluate(model, loader, loss_fn, device)
        print(name, f"loss={loss:.5f}", " ".join(
            f"{key}={value:.5f}" for key, value in metrics.items()
        ))
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
