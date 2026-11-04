"""Inspect real dataset files and validate every resized image/mask pair."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from lib.dataset.DRIVEDataset import DRIVEDataset
from lib.dataset.PH2Dataset import PH2Dataset
from lib.splits import drive_splits, ph2_splits


def inspect_dataset(dataset):
    roles = {}
    foreground = valid_pixels = 0
    for index, sample in enumerate(dataset.samples):
        if isinstance(dataset, PH2Dataset):
            paths = {"image": sample[1], "lesion": sample[2]}
        else:
            paths = {"image": sample[1], "fov": sample[2]}
            if sample[3] is not None:
                paths["vessels"] = sample[3]
        sizes = []
        for role, path in paths.items():
            info = roles.setdefault(role, {"formats": Counter(), "dimensions_wh": Counter(),
                                          "pixel_values": set()})
            with Image.open(path) as source:
                sizes.append(source.size)
                info["formats"][source.format] += 1
                info["dimensions_wh"][f"{source.width}x{source.height}"] += 1
                if role != "image":
                    info["pixel_values"].update(np.unique(np.asarray(source.convert("L"))).tolist())
        if len(set(sizes)) != 1:
            raise ValueError(f"Native image/mask dimensions differ for {sample[0]}: {sizes}")
        batch = dataset[index]
        expected = (1, dataset.height, dataset.width)
        if batch["image"].shape != (3, *expected[1:]):
            raise ValueError(f"Unexpected image shape for {sample[0]}")
        if not bool(torch.isfinite(batch["image"]).all()) or not bool(
            ((batch["image"] >= 0) & (batch["image"] <= 1)).all()
        ):
            raise ValueError(f"Image is not finite and in [0, 1]: {sample[0]}")
        for key in ("target", "valid_region"):
            if key in batch:
                value = batch[key]
                if value.shape != expected or not bool(((value == 0) | (value == 1)).all()):
                    raise ValueError(f"Invalid binary mask or shape for {sample[0]}: {key}")
        valid_pixels += int(batch["valid_region"].sum())
        if "target" in batch:
            foreground += int((batch["target"] * batch["valid_region"]).sum())
    for info in roles.values():
        info["pixel_values"] = sorted(info["pixel_values"])
    return {"samples": len(dataset), "files": roles, "resized_hw": [dataset.height, dataset.width],
            "valid_pixels": valid_pixels,
            "foreground_pixels": foreground if dataset.samples and "target" in dataset[0] else None}


def inspect_data(data_root=None, size=128, seed=42):
    root = Path("/dtu/datasets1/02516") if data_root is None else Path(data_root)
    report = {"data_root": str(root.resolve()), "split_seed": seed, "datasets": {}}
    for name, split_fn, dataset_cls in (("PH2", ph2_splits, PH2Dataset),
                                         ("DRIVE", drive_splits, DRIVEDataset)):
        splits = split_fn(seed, data_root=data_root)
        ids = sorted(sample_id for values in splits.values() for sample_id in values)
        dataset = dataset_cls("train", sample_ids=ids, size=size, data_root=data_root)
        report["datasets"][name] = {**inspect_dataset(dataset), "splits": splits,
                                     "split_counts": {key: len(value) for key, value in splits.items()}}
    official_dir = root / "DRIVE" / "test"
    if official_dir.is_dir():
        official = DRIVEDataset("official_test", size=size, data_root=data_root)
        report["datasets"]["DRIVE_official_test"] = inspect_dataset(official)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are protected.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = inspect_data(args.data_root, args.size, args.split_seed)
    contents = json.dumps(report, indent=2, allow_nan=False)
    if args.output is None:
        print(contents)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(contents + "\n")
        print(f"Validated datasets; saved {args.output}")


if __name__ == "__main__":
    main()
