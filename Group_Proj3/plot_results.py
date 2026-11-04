"""Create report figures from the sequential modelrun LSF output."""

import argparse
import csv
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


def write_summary(experiments, output):
    fields = ["dataset", "model", "loss", "split", "loss_value", *METRICS]
    with output.open("w", newline="") as handle:
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


def save_metric_bars(experiments, output):
    rows = [
        (experiment, experiment["results"]["test"])
        for experiment in experiments
        if "test" in experiment["results"]
    ]
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
    figure.savefig(output, dpi=200)
    plt.close(figure)


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
                    experiment["results"]["test"]["dice"]
                    for experiment in selected
                    if experiment["model"] == model
                    and experiment["loss"] == loss
                )
                for loss in losses
            ]
            bars = axis.bar(
                x + (index - 0.5) * width, values, width, label=model
            )
            for bar, value in zip(bars, values):
                axis.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
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
    figure.savefig(output, dpi=200)
    plt.close(figure)


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
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="LSF output file, e.g. modelrun.123.out")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    experiments = parse_log(args.log)
    if not experiments:
        raise SystemExit(f"No completed experiment results found in {args.log}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(experiments, args.output_dir / "metrics_summary.csv")
    save_metric_bars(experiments, args.output_dir / "test_metrics.png")
    save_ablation(experiments, args.output_dir / "loss_ablation.png")
    save_convergence(experiments, args.output_dir / "training_curves.png")
    print(f"Parsed {len(experiments)} experiments")
    print(f"Saved figures and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
