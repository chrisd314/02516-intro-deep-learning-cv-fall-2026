"""Create report figures from the sequential modelrun LSF output."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("dice", "iou", "accuracy", "sensitivity", "specificity")
SPLITS = ("train", "val", "test")
HEADER = re.compile(
    r"Training dataset=(?P<dataset>\w+) model=(?P<model>\w+) loss=(?P<loss>\w+)"
)
EPOCH = re.compile(
    r"epoch (?P<epoch>\d+)/\d+ train_loss=(?P<train>[0-9.]+) "
    r"val_loss=(?P<val>[0-9.]+) dice=(?P<dice>[0-9.]+)"
)
RESULT = re.compile(
    r"(?P<split>train|val|test) loss=(?P<loss>[0-9.]+) "
    r"dice=(?P<dice>[0-9.]+) iou=(?P<iou>[0-9.]+) "
    r"accuracy=(?P<accuracy>[0-9.]+) "
    r"sensitivity=(?P<sensitivity>[0-9.]+) "
    r"specificity=(?P<specificity>[0-9.]+)"
)


def parse_log(path):
    experiments = []
    current = None
    for line in path.read_text().splitlines():
        header = HEADER.search(line)
        if header:
            current = {
                **header.groupdict(),
                "epochs": [],
                "results": {},
            }
            experiments.append(current)
            continue
        if current is None:
            continue
        epoch = EPOCH.search(line)
        if epoch:
            current["epochs"].append(
                {key: float(value) for key, value in epoch.groupdict().items()}
            )
        result = RESULT.search(line)
        if result:
            values = result.groupdict()
            split = values.pop("split")
            current["results"][split] = {
                key: float(value) for key, value in values.items()
            }
    return [experiment for experiment in experiments if experiment["results"]]


def parse_records(directory):
    """Read completed train.py sidecars, preserving full metric precision."""
    experiments = []
    for path in sorted(directory.glob("*.pth.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "completed":
            print(f"Skipping incomplete run record: {path}")
            continue
        config = record["config"]
        experiments.append({
            "dataset": config["dataset"], "model": config["model"], "loss": config["loss"],
            "epochs": [{"epoch": row["epoch"], "train": row["train_loss"],
                        "val": row["val_loss"], "dice": row["val_metrics"]["dice"]}
                       for row in record["history"]],
            "results": record["final_metrics"],
        })
    return experiments


def write_summary(experiments, output):
    fields = ["dataset", "model", "loss", "split", "loss_value", *METRICS]
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in experiments:
            for split in SPLITS:
                if split not in experiment["results"]:
                    continue
                result = experiment["results"][split]
                writer.writerow({
                    "dataset": experiment["dataset"],
                    "model": experiment["model"],
                    "loss": experiment["loss"],
                    "split": split,
                    "loss_value": result["loss"],
                    **{metric: result[metric] for metric in METRICS},
                })


def save_figure(figure, output):
    """Create a new PNG exclusively and close the figure even if saving fails."""
    try:
        with Path(output).open("xb") as handle:
            figure.savefig(handle, format="png", dpi=200)
    finally:
        plt.close(figure)


def save_metric_bars(experiments, output):
    rows = [
        (experiment, experiment["results"]["test"])
        for experiment in experiments
        if "test" in experiment["results"]
    ]
    if not rows:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.text(0.5, 0.5, "No completed test results available",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.tight_layout()
        save_figure(figure, output)
        return
    labels = [
        f'{experiment["dataset"]}\n{experiment["model"]}\n{experiment["loss"]}'
        for experiment, _ in rows
    ]
    values = np.array([[result[metric] for metric in METRICS] for _, result in rows])
    figure, axis = plt.subplots(figsize=(16, 7))
    x = np.arange(len(labels))
    width = 0.15
    for index, metric in enumerate(METRICS):
        axis.bar(x + (index - 2) * width, values[:, index], width, label=metric)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Test-set segmentation metrics")
    axis.legend(ncol=len(METRICS))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output)


def save_ablation(experiments, output):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, dataset in zip(axes, ("PH2", "DRIVE")):
        selected = [
            experiment for experiment in experiments
            if experiment["dataset"] == dataset
            and experiment["model"] in {"encdec", "unet"}
            and "test" in experiment["results"]
        ]
        losses = ("bce", "focal", "weighted_bce")
        x = np.arange(len(losses))
        width = 0.35
        for index, model in enumerate(("encdec", "unet")):
            values = [
                next(
                    (
                        experiment["results"]["test"]["dice"]
                        for experiment in selected
                        if experiment["model"] == model
                        and experiment["loss"] == loss
                    ),
                    np.nan,
                )
                for loss in losses
            ]
            bars = axis.bar(
                x + (index - 0.5) * width, values, width, label=model
            )
            for bar, value in zip(bars, values):
                missing = np.isnan(value)
                axis.annotate(
                    "N/A" if missing else f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, 0 if missing else value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(f"{dataset}: loss ablation")
        axis.set_xticks(x)
        axis.set_xticklabels(losses)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Test Dice")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    save_figure(figure, output)


def save_convergence(experiments, output):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, dataset in zip(axes, ("PH2", "DRIVE")):
        for experiment in experiments:
            if experiment["dataset"] != dataset or not experiment["epochs"]:
                continue
            epochs = experiment["epochs"]
            label = f'{experiment["model"]}-{experiment["loss"]}'
            axis.plot(
                [item["epoch"] for item in epochs],
                [item["dice"] for item in epochs],
                marker=".",
                label=label,
            )
        axis.set_title(f"{dataset}: validation Dice during training")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation Dice")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No epoch history available",
                      ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    save_figure(figure, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, nargs="?",
                        help="LSF output file, e.g. modelrun.123.out")
    parser.add_argument("--records-dir", type=Path,
                        help="Read completed .pth.json records instead of a text log.")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                        help="Destination directory; existing report outputs are protected.")
    args = parser.parse_args()
    if (args.log is None) == (args.records_dir is None):
        parser.error("Provide either a log path or --records-dir")
    if os.path.lexists(args.output_dir) and not args.output_dir.is_dir():
        parser.error(f"Output path is not a directory: {args.output_dir}")
    outputs = {
        "summary": args.output_dir / "metrics_summary.csv",
        "metrics": args.output_dir / "test_metrics.png",
        "ablation": args.output_dir / "loss_ablation.png",
        "convergence": args.output_dir / "training_curves.png",
    }
    for path in outputs.values():
        if os.path.lexists(path):
            parser.error(f"Report output already exists: {path}. "
                         "Choose a new --output-dir.")

    experiments = parse_log(args.log) if args.log is not None else parse_records(args.records_dir)
    if not experiments:
        raise SystemExit(f"No completed experiment results found in {args.log or args.records_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(experiments, outputs["summary"])
    save_metric_bars(experiments, outputs["metrics"])
    save_ablation(experiments, outputs["ablation"])
    save_convergence(experiments, outputs["convergence"])
    print(f"Parsed {len(experiments)} experiments")
    print(f"Saved figures and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
