"""Aggregate verified Part 2 records and generate the PH2 comparison figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.dataset.PH2Dataset import PH2Dataset
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.records import checkpoint_path, file_hash, load_record, validate_reference


METRICS = ("dice", "iou", "accuracy", "sensitivity", "specificity")


def save_figure(figure, output):
    try:
        with Path(output).open("xb") as handle:
            figure.savefig(handle, format="png", dpi=160)
    finally:
        plt.close(figure)


def label(record):
    config = record["config"]
    if config.get("supervision", "full") == "full":
        return "fully supervised"
    return f"{config['strategy']} {config['positive_clicks']}+{config['negative_clicks']}"


def read_experiments(run_dir):
    manifest = json.loads((run_dir / "experiments.json").read_text(encoding="utf-8"))
    runs = []
    for experiment in manifest["experiments"]:
        path = run_dir / Path(experiment["record"]).name
        if not path.is_file():
            print(f"Missing run, excluded from report: {path.name}")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("status") != "completed":
            print(f"Incomplete run, excluded from report: {path.name}")
            continue
        if "record_sha256" in experiment and file_hash(path) != experiment["record_sha256"]:
            raise ValueError(f"Run record checksum changed: {path}")
        runs.append((path, raw))
    if not runs or runs[0][1]["config"].get("supervision", "full") != "full":
        raise ValueError("The report needs a completed fully supervised reference")
    reference = runs[0][1]
    for path, record in runs:
        validate_reference(reference, record["config"])
        if record["splits"] != reference["splits"]:
            raise ValueError(f"Split IDs differ from the reference: {path}")
        if "initial_parameter_sha256" in reference and record.get("initial_parameter_sha256") != reference["initial_parameter_sha256"]:
            raise ValueError(f"Initialization differs from the reference: {path}")
    return manifest, runs


def write_summary(runs, output):
    fields = ["experiment", "supervision", "strategy", "positive_clicks", "negative_clicks",
              "mean_actual_clicks", "seed", "click_seed", "split", "loss_value", *METRICS]
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _, record in runs:
            config = record["config"]
            mode = config.get("supervision", "full")
            actual = record.get("annotation_summary", {})
            common = {"experiment": label(record), "supervision": mode,
                      "strategy": config.get("strategy", "") if mode == "points" else "",
                      "positive_clicks": config["positive_clicks"] if mode == "points" else "",
                      "negative_clicks": config["negative_clicks"] if mode == "points" else "",
                      "mean_actual_clicks": sum(value["mean_sampled"] for value in actual.values()) if actual else "",
                      "seed": config["seed"], "click_seed": config.get("click_seed", "") if mode == "points" else ""}
            for split in ("train", "val", "test"):
                values = record["final_metrics"][split]
                writer.writerow({**common, "split": split, "loss_value": values["loss"],
                                 **{metric: values[metric] for metric in METRICS}})


def save_comparison(records, output, title):
    figure, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(len(records))
    for index, metric in enumerate(METRICS):
        axis.bar(x + (index - 2) * 0.15,
                 [record["final_metrics"]["test"][metric] for record in records],
                 0.15, label=metric)
    axis.set_xticks(x, [label(record) for record in records], rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test score")
    axis.set_title(title)
    axis.legend(ncol=5)
    figure.tight_layout()
    save_figure(figure, output)


def save_click_count(records, output):
    reference = records[0]
    points = sorted((record for record in records[1:] if record["config"]["strategy"] == "uniform"),
                    key=lambda record: record["config"]["positive_clicks"] + record["config"]["negative_clicks"])
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, metric in zip(axes.flat, METRICS):
        for split, color in (("val", "tab:blue"), ("test", "tab:orange")):
            axis.plot([record["config"]["positive_clicks"] + record["config"]["negative_clicks"] for record in points],
                      [record["final_metrics"][split][metric] for record in points],
                      marker="o", color=color, label=f"uniform {split}")
            axis.axhline(reference["final_metrics"][split][metric], color=color,
                         linestyle="--", label=f"full {split}")
        axis.set_title(metric)
        axis.set_xlabel("Requested total clicks per training image")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.7, "Balanced positive/negative clicks.\n"
                       "Dashed lines: fully supervised reference.\n"
                       "Actual capped counts are saved in CSV/JSON.\n"
                       "No repeated-seed confidence intervals.", fontsize=10)
    figure.tight_layout()
    save_figure(figure, output)


def save_history(records, output):
    figure, axis = plt.subplots(figsize=(11, 6))
    for record in records:
        history = record["history"]
        axis.plot([row["epoch"] for row in history], [row["val_metrics"]["dice"] for row in history],
                  label=label(record))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Full-mask validation Dice")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output)


def load_model(record, path, device):
    weights = checkpoint_path(record, path)
    if "checkpoint_sha256" in record and file_hash(weights) != record["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint checksum changed: {weights}")
    model = EncDec() if record["config"]["model"] == "encdec" else UNet()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def save_examples(runs, data_root, strategy_clicks, output):
    reference = runs[0][1]
    selected = [runs[0]] + [(path, record) for path, record in runs[1:]
                           if record["config"]["positive_clicks"] == strategy_clicks
                           and record["config"]["negative_clicks"] == strategy_clicks]
    dataset = PH2Dataset("test", reference["splits"]["test"][:2],
                         size=reference["config"]["size"], data_root=data_root)
    figure, axes = plt.subplots(len(dataset), len(selected) + 2,
                                figsize=(3 * (len(selected) + 2), 3 * len(dataset)), squeeze=False)
    samples = [dataset[index] for index in range(len(dataset))]
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["image"].permute(1, 2, 0))
        axes[row, 0].set_title(sample["image_id"])
        axes[row, 1].imshow(sample["target"][0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for column, (path, record) in enumerate(selected, start=2):
        model = load_model(record, path, device)
        with torch.no_grad():
            for row, sample in enumerate(samples):
                probability = torch.sigmoid(model(sample["image"].unsqueeze(0).to(device)))[0, 0].cpu()
                axes[row, column].imshow(probability >= 0.5, cmap="gray", vmin=0, vmax=1)
                axes[row, column].set_title(label(record))
    for axis in axes.flat:
        axis.set_axis_off()
    figure.tight_layout()
    save_figure(figure, output)


def save_click_examples(runs, data_root, output):
    points = runs[1:]
    if not points:
        return
    reference = runs[0][1]
    dataset = PH2Dataset("train", reference["splits"]["train"][:2],
                         size=reference["config"]["size"], data_root=data_root)
    figure, axes = plt.subplots(len(dataset), len(points), figsize=(3 * len(points), 3 * len(dataset)), squeeze=False)
    for column, (path, record) in enumerate(points):
        annotations_path = path.parent / Path(record["annotations"]).name
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["cases"]
        for row in range(len(dataset)):
            sample = dataset[row]
            axis = axes[row, column]
            axis.imshow(sample["image"].permute(1, 2, 0))
            mask = sample["target"][0].numpy()
            if mask.min() != mask.max():
                axis.contour(mask, levels=[0.5], colors="white", linewidths=0.5)
            for name, color in (("positive", "lime"), ("negative", "cyan")):
                coordinates = annotations[sample["image_id"]][name]["coordinates_yx"]
                if coordinates:
                    y, x = np.asarray(coordinates).T
                    axis.scatter(x, y, c=color, s=14, edgecolors="black", linewidths=0.3)
            axis.set_title(f"{sample['image_id']}\n{label(record)}")
            axis.set_axis_off()
    figure.suptitle("Training annotations: positive = green, negative = cyan; white = lesion boundary")
    figure.tight_layout()
    save_figure(figure, output)


def write_discussion(records, manifest, output):
    baseline = records[0]["final_metrics"]["val"]["dice"]
    threshold = 0.95 * baseline
    uniform = sorted((record for record in records[1:] if record["config"]["strategy"] == "uniform"),
                      key=lambda record: record["config"]["positive_clicks"] + record["config"]["negative_clicks"])
    reached = [record for record in uniform if record["final_metrics"]["val"]["dice"] >= threshold]
    if baseline <= 1e-6:
        approach = "The full reference has near-zero validation Dice; a relative approach threshold is not informative."
    elif reached:
        approach = f"The smallest evaluated uniform budget reaching 95% of full validation Dice is {label(reached[0])}."
    else:
        approach = "No evaluated uniform budget reached 95% of full validation Dice; no extrapolation is made."
    config = records[0]["config"]
    lines = ["# PH2 fully versus weakly supervised results", "",
             f"Completed records included: {len(records)} of {len(manifest['experiments'])} planned.",
             f"Configuration: {config['model']}, size {config['size']}, {config['epochs']} epochs, seed {config['seed']}.", "",
             "All losses in metrics_summary.csv are dense evaluation BCE, including for sparse models.",
             "Sparse training loss is stored separately in each epoch history and uses annotated pixels only.", "",
             "| Experiment | Test Dice | Test IoU | Accuracy | Sensitivity | Specificity |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for record in records:
        values = record["final_metrics"]["test"]
        lines.append("| " + " | ".join([label(record)] + [f"{values[key]:.6f}" for key in METRICS]) + " |")
    lines += ["", approach, "", "The approach criterion is 0.95 times the full reference's validation Dice,",
              "specified before comparison. Test scores do not select hyperparameters or checkpoints.", "",
              "Full validation masks are used for evaluation/checkpoint selection in both methods.",
              "The sparse annotation budget applies to training images only. Full training masks",
              "generate clicks once and are otherwise used only for final evaluation/visualization.", "",
              "Clicks are fixed across epochs, unique and sampled at the resized image resolution.",
              "Interior/boundary sampling may fall back to other same-class pixels; exact fallback",
              "counts and coordinates are saved in .points.json and annotation_summary.", "",
              "These are single-training-seed and single-click-seed comparisons, without confidence",
              "intervals. Low-resolution masks, sampling from perfect GT and fully labeled validation",
              "limit claims about real user annotations. Scores alone do not establish causal reasons.", "",
              "Representative images use the first two fixed IDs, not the best test predictions.",
              "Training curves, click-count plots, strategy comparisons and all split scores are separate artifacts.", ""]
    with output.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to RUN_DIR/report.")
    parser.add_argument("--data-root", help="Override the data location after copying HPC outputs.")
    args = parser.parse_args()
    manifest, runs = read_experiments(args.run_dir)
    output = args.output_dir or args.run_dir / "report"
    if output.exists():
        parser.error(f"Report directory already exists: {output}; choose a new --output-dir")
    data_root = args.data_root or manifest["config"].get("data_root")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records = [record for _, record in runs]
    write_summary(runs, output / "metrics_summary.csv")
    save_comparison(records, output / "metrics_comparison.png", "PH2: full versus point supervision")
    save_click_count(records, output / "click_count.png")
    strategy_clicks = manifest["config"]["strategy_clicks"]
    selected = [records[0]] + [record for record in records[1:]
                               if record["config"]["positive_clicks"] == strategy_clicks
                               and record["config"]["negative_clicks"] == strategy_clicks]
    save_comparison(selected, output / "sampling_strategies.png", f"Sampling strategies at {strategy_clicks}+{strategy_clicks} clicks")
    save_history(records, output / "training_curves.png")
    save_examples(runs, data_root, strategy_clicks, output / "predictions.png")
    save_click_examples(runs, data_root, output / "click_examples.png")
    write_discussion(records, manifest, output / "results.md")
    print(f"Saved {len(records)} verified experiment records to {output}")


if __name__ == "__main__":
    main()
