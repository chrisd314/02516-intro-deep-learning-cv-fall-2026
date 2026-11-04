"""Train PH2 with full masks or fixed sparse clicks, using the Part 1 models."""

import argparse
import math
import os
from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.dataset.PH2Dataset import PH2Dataset
from lib.dataset.PointPH2Dataset import PointPH2Dataset
from lib.losses import BCELoss, FocalLoss, PointBCELoss, WeightedBCELoss
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.points import STRATEGIES
from lib.records import file_hash, load_record, model_hash, save_run_record, validate_reference, validate_splits
from lib.splits import ph2_splits
from measure import confusion_counts, metrics_from_counts

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

def add_common_arguments(parser):
    parser.add_argument("--model", choices=("encdec", "unet"), default="unet")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail rather than train on CPU; enabled by the HPC script.")


def validate_arguments(args):
    if args.size < 32 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("Size must be >=32; batch size and epochs must be positive")
    if args.seed < 0 or args.num_workers < 0 or args.num_threads < 1:
        raise ValueError("Seed/workers must be nonnegative and CPU threads positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")


def build_datasets(args, reference=None):
    generated = ph2_splits(args.split_seed, data_root=args.data_root)
    splits = generated if reference is None else reference["splits"]
    validate_splits(splits, [value for ids in generated.values() for value in ids])
    return tuple(PH2Dataset(split, sample_ids=splits[split], size=args.size,
                            image_transform=transforms.ToTensor(), data_root=args.data_root)
                 for split in ("train", "val", "test"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--supervision", choices=("full", "points"), default="points")
    parser.add_argument("--positive-clicks", type=int, default=10)
    parser.add_argument("--negative-clicks", type=int, default=10)
    parser.add_argument("--click-seed", type=int, default=42)
    parser.add_argument("--strategy", choices=STRATEGIES, default="uniform")
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-record", help="Completed fully supervised PH2 BCE .pth.json.")
    args = parser.parse_args()
    try:
        validate_arguments(args)
        if min(args.positive_clicks, args.negative_clicks, args.click_seed) < 0 or args.margin < 1:
            raise ValueError("Click counts/seed must be nonnegative and margin positive")
        if args.supervision == "points" and args.positive_clicks + args.negative_clicks == 0:
            raise ValueError("Point supervision needs at least one requested click")
    except ValueError as error:
        parser.error(str(error))
    args.checkpoint = str(Path(args.checkpoint).resolve())
    record_path = args.checkpoint + ".json"
    annotation_path = args.checkpoint + ".points.json"
    for path in (args.checkpoint, record_path, annotation_path):
        if os.path.lexists(path):
            parser.error(f"Output already exists: {path}; choose a new --checkpoint")
    config = {**vars(args), "dataset": "PH2", "loss": "bce"}
    reference = None
    if args.reference_record:
        reference = load_record(args.reference_record)
        validate_reference(reference, config)

    torch.set_num_threads(args.num_threads)
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the allocated GPU and HPC Python environment")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainbase, valset, testset = build_datasets(args, reference)
    if args.supervision == "points":
        trainset = PointPH2Dataset(trainbase, args.positive_clicks, args.negative_clicks,
                                  args.click_seed, args.strategy, args.margin)
        annotation_count = sum(int((labels != -1).sum()) for labels in trainset.labels)
        if annotation_count == 0:
            raise ValueError("No annotated training pixels are available")
    else:
        trainset = trainbase
    trainloader = DataLoader(trainset, args.batch_size, shuffle=True,
                             num_workers=args.num_workers, generator=generator)
    valloader = DataLoader(valset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = (EncDec() if args.model == "encdec" else UNet()).to(device)
    initial_hash = model_hash(model)
    if reference and reference.get("initial_parameter_sha256", initial_hash) != initial_hash:
        raise ValueError("Initial parameters differ from the fully supervised reference")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dense_loss = BCELoss()
    point_loss = PointBCELoss()
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "running", "config": config, "working_directory": os.getcwd(),
        "device": str(device), "torch_version": str(torch.__version__),
        "initial_parameter_sha256": initial_hash, "loss_parameters": {},
        "metric_aggregation": "global_valid_pixels", "loss_aggregation": "mean_valid_pixels",
        "training_loss_aggregation": "mean_annotated_pixels" if args.supervision == "points" else "mean_valid_pixels",
        "validation_supervision": "full_masks_for_evaluation_only",
        "prediction_threshold": 0.5, "metric_smoothing": 1e-7,
        "preprocessing": {"size": args.size, "image_range": [0, 1], "image_resize": "bilinear",
                          "mask_resize": "nearest", "mask_foreground": "nonzero", "augmentation": "none"},
        "splits": {name: list(dataset.sample_ids) for name, dataset in
                   zip(("train", "val", "test"), (trainbase, valset, testset))},
        "reference_record": str(Path(args.reference_record).resolve()) if reference else None,
        "reference_record_sha256": file_hash(args.reference_record) if reference else None,
        "annotations": annotation_path if args.supervision == "points" else None,
        "history": [], "best_epoch": None, "best_val_loss": None, "final_metrics": {},
    }
    if args.supervision == "points":
        record["annotation_summary"] = {
            name: {"mean_sampled": sum(case[name]["sampled"] for case in trainset.annotations.values()) / len(trainset),
                   "fallback_clicks": sum(case[name]["fallback"] for case in trainset.annotations.values())}
            for name in ("positive", "negative")
        }
    save_run_record(record_path, record, create=True)
    if args.supervision == "points":
        save_run_record(annotation_path, {"labels": {"unknown": -1, "background": 0, "lesion": 1},
                                          "config": config, "cases": trainset.annotations}, create=True)
    print(f"Training PH2 {args.model} supervision={args.supervision} "
          f"strategy={args.strategy} clicks={args.positive_clicks}+{args.negative_clicks} device={device}", flush=True)
    best_val = float("inf")
    for epoch in range(args.epochs):
        started = time()
        model.train()
        total, total_pixels = 0.0, 0
        for batch in trainloader:
            images = batch["image"].to(device)
            if args.supervision == "points":
                points = batch["points"].to(device)
                n_pixels = int((points != -1).sum())
            else:
                targets = batch["target"].to(device)
                valid = batch["valid_region"].to(device)
                n_pixels = int(valid.expand_as(targets).sum())
            if n_pixels == 0:
                continue
            optimizer.zero_grad()
            logits = model(images)
            loss = point_loss(logits, points) if args.supervision == "points" else masked_loss(dense_loss, logits, targets, valid)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Training loss is not finite")
            loss.backward()
            optimizer.step()
            total += loss.item() * n_pixels
            total_pixels += n_pixels
        if total_pixels == 0:
            raise ValueError("No supervised pixels in this training epoch")
        train_loss = total / total_pixels
        val_loss, val_metrics = evaluate(model, valloader, dense_loss, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            record["best_epoch"], record["best_val_loss"] = epoch + 1, val_loss
        record["history"].append({"epoch": epoch + 1, "train_loss": train_loss,
                                  "val_loss": val_loss, "val_metrics": val_metrics})
        save_run_record(record_path, record)
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.5f} "
              f"val_loss={val_loss:.5f} val_dice={val_metrics['dice']:.5f} "
              f"time={time() - started:.1f}s", flush=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    train_eval_loader = DataLoader(trainbase, args.batch_size, shuffle=False, num_workers=args.num_workers)
    for name, loader in zip(("train", "val", "test"), (train_eval_loader, valloader, testloader)):
        loss, metrics = evaluate(model, loader, dense_loss, device)
        record["final_metrics"][name] = {"loss": loss, **metrics}
        print(name, f"dense_evaluation_bce={loss:.6f}", metrics, flush=True)
    record["checkpoint_sha256"] = file_hash(args.checkpoint)
    record["status"] = "completed"
    save_run_record(record_path, record)
    print(f"Saved {record_path}", flush=True)


if __name__ == "__main__":
    main()
