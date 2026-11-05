"""Save representative PH2 and DRIVE dataset visualizations."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lib.dataset.DRIVEDataset import DRIVEDataset
from lib.dataset.PH2Dataset import PH2Dataset
from lib.splits import drive_splits, ph2_splits


def overlay(image, mask, color):
    result = image.astype(np.float32).copy()
    result[mask] = 0.55 * result[mask] + 0.45 * np.asarray(color)
    return np.clip(result, 0, 255).astype(np.uint8)


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=180)
    plt.close(figure)


def save_ph2(dataset, path):
    sample = dataset[0]
    image = sample["image"].permute(1, 2, 0).numpy()
    mask = sample["target"][0].numpy().astype(bool)
    image = (image * 255).astype(np.uint8)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"PH2 image: {sample['image_id']}")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Lesion ground truth")
    axes[2].imshow(overlay(image, mask, (255, 40, 40)))
    axes[2].set_title("Lesion overlay")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    save_figure(figure, path)


def save_drive(dataset, path):
    sample = dataset[0]
    image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    vessels = sample["target"][0].numpy().astype(bool)
    fov = sample["valid_region"][0].numpy().astype(bool)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = (
        (image, f"DRIVE image: {sample['image_id']}", None),
        (vessels, "Vessel ground truth", "gray"),
        (fov, "Field of view", "gray"),
        (overlay(image, vessels, (40, 255, 40)), "Vessel overlay", None),
    )
    for axis, (panel, title, cmap) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    save_figure(figure, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    ph2_ids = ph2_splits(args.split_seed, args.data_root)["train"]
    drive_ids = drive_splits(args.split_seed, args.data_root)["train"]
    ph2 = PH2Dataset("train", ph2_ids, size=args.size, data_root=args.data_root)
    from lib.dataset.DRIVEDataset import DRIVEDataset
    drive = DRIVEDataset("train", drive_ids, size=args.size, data_root=args.data_root)
    save_ph2(ph2, args.output_dir / "ph2_example.png")
    save_drive(drive, args.output_dir / "drive_example.png")
    print(f"Saved Part 1 visualizations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
