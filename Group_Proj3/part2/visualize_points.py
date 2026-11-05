"""Save PH2 examples with simulated weak-supervision clicks."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.dataset.PH2Dataset import PH2Dataset
from lib.points import sample_points, image_seed
from lib.splits import ph2_splits


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        figure.savefig(handle, format="png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--click-seed", type=int, default=42)
    parser.add_argument("--positive-clicks", type=int, default=10)
    parser.add_argument("--negative-clicks", type=int, default=10)
    parser.add_argument("--strategy", choices=("uniform", "interior", "boundary"),
                        default="uniform")
    parser.add_argument("--margin", type=int, default=3)
    args = parser.parse_args()
    ids = ph2_splits(args.split_seed, args.data_root)["train"][:2]
    dataset = PH2Dataset("train", ids, size=args.size, data_root=args.data_root)
    figure, axes = plt.subplots(len(dataset), 2, figsize=(8, 4 * len(dataset)),
                                squeeze=False)
    for row, sample in enumerate(dataset):
        image = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["target"]
        points, _ = sample_points(
            mask, args.positive_clicks, args.negative_clicks,
            image_seed(args.click_seed, sample["image_id"]),
            args.strategy, args.margin,
        )
        axes[row, 0].imshow(image)
        axes[row, 0].contour(
            mask[0].numpy(), levels=[0.5], colors="white", linewidths=0.7
        )
        axes[row, 0].set_title(f"{sample['image_id']} image and lesion boundary")
        axes[row, 1].imshow(image)
        positive = points[0].numpy() == 1
        negative = points[0].numpy() == 0
        y_positive, x_positive = np.where(positive)
        y_negative, x_negative = np.where(negative)
        axes[row, 1].scatter(x_positive, y_positive, c="lime", edgecolors="black",
                             s=24, label="positive click")
        axes[row, 1].scatter(x_negative, y_negative, c="cyan", edgecolors="black",
                             s=24, label="negative click")
        axes[row, 1].set_title(
            f"{args.strategy}: {args.positive_clicks}+{args.negative_clicks} clicks"
        )
        axes[row, 1].legend(loc="lower right", fontsize=8)
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle("Part 2 simulated point annotations", y=1.01)
    figure.tight_layout()
    save_figure(figure, args.output_dir / "point_annotations.png")
    print(f"Saved Part 2 visualizations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
