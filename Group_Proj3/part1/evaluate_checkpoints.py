"""Re-evaluate the 12 existing checkpoints without retraining or replacing them."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.losses import BCELoss, FocalLoss, WeightedBCELoss
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from plot_results import save_ablation, save_figure, save_metric_bars, write_summary
from train import build_datasets, evaluate


def save_examples(model, dataset, device, output):
    """Use the first two fixed test IDs, chosen independently of their scores."""
    count = min(2, len(dataset))
    figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    model.eval()
    with torch.no_grad():
        for row in range(count):
            batch = dataset[row]
            probability = torch.sigmoid(model(batch["image"].unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            valid = batch["valid_region"][0].numpy().astype(bool)
            target = batch["target"][0].numpy()
            prediction = probability >= 0.5
            panels = (batch["image"].permute(1, 2, 0).numpy(), target,
                      probability, prediction)
            titles = (batch["image_id"], "Ground truth", "Foreground probability", "Prediction (p >= 0.5)")
            for axis, panel, title in zip(axes[row], panels, titles):
                if panel.ndim == 2:
                    panel = np.ma.masked_where(~valid, panel)
                    axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
                else:
                    axis.imshow(panel)
                axis.set_title(title)
                axis.set_axis_off()
    figure.tight_layout()
    save_figure(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory, created exclusively.")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if args.size < 32 or args.batch_size < 1 or args.num_threads < 1:
        parser.error("Size must be >=32; batch size and CPU threads must be positive")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    paths = [args.checkpoints_dir / f"{dataset.lower()}_{model}_{loss}.pth"
             for dataset in ("PH2", "DRIVE") for model in ("encdec", "unet")
             for loss in ("bce", "focal", "weighted_bce")]
    for path in paths:
        if not path.is_file():
            parser.error(f"Required checkpoint not found: {path}")
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {name: build_datasets(name, args.size, args.split_seed, args.data_root)
                for name in ("PH2", "DRIVE")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    experiments, sources = [], []
    report = {
        "status": "running", "kind": "historical_checkpoint_re_evaluation",
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "device": str(device), "torch_version": str(torch.__version__),
        "metric_aggregation": "global_valid_pixels", "loss_aggregation": "mean_valid_pixels",
        "prediction_threshold": 0.5, "metric_smoothing": 1e-7,
        "splits": {name: {split: list(dataset.sample_ids)
                          for split, dataset in zip(("train", "val", "test"), values)}
                   for name, values in datasets.items()},
        "limitations": [
            "Historical training seeds, exact historical split membership and best-epoch provenance were not recorded.",
            "This evaluates fixed weights on the stated current splits; it does not establish that these splits were held out historically.",
            "Loss parameters use the existing defaults: focal alpha=0.25/gamma=2; positive BCE weight=2.",
            "These results do not replace a fresh reproducible fully supervised reference for Part 2.",
        ],
        "checkpoints": sources, "experiments": experiments,
    }
    for dataset_name in ("PH2", "DRIVE"):
        loaders = [DataLoader(dataset, args.batch_size, shuffle=False)
                   for dataset in datasets[dataset_name]]
        for model_name in ("encdec", "unet"):
            for loss_name, loss_fn in (("bce", BCELoss()), ("focal", FocalLoss()),
                                       ("weighted_bce", WeightedBCELoss())):
                started = time.perf_counter()
                stem = f"{dataset_name.lower()}_{model_name}_{loss_name}"
                path = args.checkpoints_dir / f"{stem}.pth"
                model = (EncDec() if model_name == "encdec" else UNet()).to(device)
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                experiment = {"dataset": dataset_name, "model": model_name,
                              "loss": loss_name, "epochs": [], "results": {}}
                counts_by_split = {}
                for split, loader in zip(("train", "val", "test"), loaders):
                    loss, metrics, counts = evaluate(model, loader, loss_fn, device, return_counts=True)
                    experiment["results"][split] = {"loss": loss, **metrics}
                    counts_by_split[split] = dict(zip(("tp", "fp", "fn", "tn"), counts))
                experiments.append(experiment)
                sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "counts": counts_by_split})
                save_examples(model, datasets[dataset_name][2], device,
                              args.output_dir / f"{stem}_examples.png")
                score = experiment["results"]["test"]["dice"]
                print(f"{stem}: test Dice={score:.6f}; {time.perf_counter() - started:.1f}s", flush=True)
    write_summary(experiments, args.output_dir / "metrics_summary.csv")
    save_metric_bars(experiments, args.output_dir / "test_metrics.png")
    save_ablation(experiments, args.output_dir / "loss_ablation.png")
    report["status"] = "completed"
    with (args.output_dir / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved verified evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
