"""Train an encoder-decoder or U-Net on PH2 or DRIVE."""

import argparse
import json
import math
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
from measure import confusion_counts, metrics_from_counts


def build_datasets(dataset_name, size, seed, data_root=None):
    split_fn = ph2_splits if dataset_name == "PH2" else drive_splits
    split_ids = split_fn(seed, data_root=data_root)
    dataset_cls = PH2Dataset if dataset_name == "PH2" else DRIVEDataset
    kwargs = {"image_transform": transforms.ToTensor(), "size": size,
              "data_root": data_root}
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
        raise TypeError("FOV masking supports BCELoss, FocalLoss and WeightedBCELoss")
    return (values * pixel_mask).sum() / pixel_mask.sum().clamp_min(1.0)


def evaluate(model, loader, loss_fn, device, return_counts=False):
    """Average loss by valid pixels and compute global split metrics."""
    model.eval()
    totals = {"loss": 0.0, "pixels": 0}
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            valid = batch["valid_region"].to(device)
            logits = model(images)
            n_pixels = int(valid.expand_as(targets).sum().item())
            totals["loss"] += masked_loss(loss_fn, logits, targets, valid).item() * n_pixels
            totals["pixels"] += n_pixels
            counts += confusion_counts(logits, targets, valid)
    if totals["pixels"] == 0:
        raise ValueError("Cannot evaluate a split without valid pixels")
    metrics = metrics_from_counts(counts)
    result = (totals["loss"] / totals["pixels"], {
        name: value.item() for name, value in metrics.items()
    })
    if return_counts:
        return (*result, counts.cpu().tolist())
    return result


def save_run_record(path, record, create=False):
    """Write strict JSON; exclusive creation protects an earlier run's record."""
    contents = json.dumps(record, indent=2, allow_nan=False)
    with open(path, "x" if create else "w", encoding="utf-8") as handle:
        handle.write(contents + "\n")


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
    parser.add_argument("--seed", type=int, default=42,
                        help="Model initialization and training-shuffle seed.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Optional number of PyTorch CPU threads.")
    parser.add_argument("--data-root", default=None,
                        help="Folder containing PH2_Dataset_images and DRIVE; defaults to HPC.")
    parser.add_argument("--checkpoint", default=None,
                        help="New checkpoint path; it and its .json record must be unused.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32:
        parser.error("--epochs and --batch-size must be positive; --size must be at least 32")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.num_workers < 0:
        parser.error("--lr must be finite and positive; --num-workers must be nonnegative")
    if args.num_threads is not None:
        if args.num_threads < 1:
            parser.error("--num-threads must be positive")
        torch.set_num_threads(args.num_threads)

    checkpoint = args.checkpoint or (
        f"checkpoints/{args.dataset.lower()}_{args.model}_{args.loss}.pth"
    )
    if os.path.lexists(checkpoint):
        parser.error(f"Checkpoint path already exists: {checkpoint}. "
                     "Choose a new --checkpoint path.")
    record_path = checkpoint + ".json"
    if os.path.lexists(record_path):
        parser.error(f"Run record path already exists: {record_path}. "
                     "Choose a new --checkpoint path.")

    torch.manual_seed(args.seed)
    train_generator = torch.Generator().manual_seed(args.seed)
    print(f"split_seed={args.split_seed} training_seed={args.seed}")
    print(f"Training dataset={args.dataset} model={args.model} loss={args.loss}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, valset, testset = build_datasets(
        args.dataset, args.size, args.split_seed, args.data_root,
    )
    loaders = [
        DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                   generator=train_generator),
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
    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
    record = {
        "status": "running",
        "config": {**vars(args), "checkpoint": checkpoint},
        "working_directory": os.getcwd(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "loss_parameters": {name: getattr(loss_fn, name)
                            for name in ("alpha", "gamma", "positive_weight")
                            if hasattr(loss_fn, name)},
        "metric_aggregation": "global_valid_pixels",
        "loss_aggregation": "mean_valid_pixels",
        "prediction_threshold": 0.5,
        "metric_smoothing": 1e-7,
        "preprocessing": {"size": args.size, "image_range": [0, 1],
                          "image_resize": "bilinear", "mask_resize": "nearest",
                          "mask_foreground": "nonzero", "augmentation": "none"},
        "splits": {name: list(dataset.sample_ids) for name, dataset in
                   zip(("train", "val", "test"), (trainset, valset, testset))},
        "history": [],
        "best_epoch": None,
        "best_val_loss": None,
        "final_metrics": {},
    }
    save_run_record(record_path, record, create=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        started = time()
        model.train()
        total = 0.0
        total_pixels = 0
        for batch in loaders[0]:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            valid = batch["valid_region"].to(device)
            optimizer.zero_grad()
            loss = masked_loss(loss_fn, model(images), targets, valid)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite; stopping before an optimizer step")
            loss.backward()
            optimizer.step()
            n_pixels = int(valid.expand_as(targets).sum().item())
            total += loss.item() * n_pixels
            total_pixels += n_pixels
        if total_pixels == 0:
            raise ValueError("Cannot report training loss without valid pixels")
        train_loss = total / total_pixels
        val_loss, val_metrics = evaluate(model, loaders[1], loss_fn, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} dice={val_metrics['dice']:.4f} "
            f"time={time() - started:.1f}s"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)
            record["best_epoch"] = epoch + 1
            record["best_val_loss"] = val_loss
        record["history"].append({"epoch": epoch + 1, "train_loss": train_loss,
                                  "val_loss": val_loss, "val_metrics": val_metrics})
        save_run_record(record_path, record)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    for name, loader in zip(("train", "val", "test"), loaders):
        loss, metrics = evaluate(model, loader, loss_fn, device)
        print(name, f"loss={loss:.5f}", " ".join(
            f"{key}={value:.5f}" for key, value in metrics.items()
        ))
        record["final_metrics"][name] = {"loss": loss, **metrics}
    record["status"] = "completed"
    save_run_record(record_path, record)
    print(f"saved checkpoint: {checkpoint}")
    print(f"saved run record: {record_path}")


if __name__ == "__main__":
    main()
