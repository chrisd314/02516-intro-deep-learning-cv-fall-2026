"""Deterministic train/validation/test splits for the project datasets."""

import random
from pathlib import Path
import re


def _partition(ids, seed, fractions=(0.7, 0.15, 0.15)):
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset IDs must be unique before splitting")
    random.Random(seed).shuffle(ids)
    n_train = int(len(ids) * fractions[0])
    n_val = int(len(ids) * fractions[1])
    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    if not all(splits.values()):
        raise ValueError("Not enough samples for nonempty train/val/test splits")
    return splits


def ph2_splits(seed=42, data_root=None):
    root = Path("/dtu/datasets1/02516") if data_root is None else Path(data_root)
    data_path = root / "PH2_Dataset_images"
    ids = sorted(
        path.name
        for path in data_path.iterdir()
        if path.is_dir() and re.fullmatch(r"IMD\d+", path.name)
    )
    if not ids:
        raise FileNotFoundError(f"No PH2 case folders found in {data_path}")
    return _partition(ids, seed)


def drive_splits(seed=42, data_root=None):
    root = Path("/dtu/datasets1/02516") if data_root is None else Path(data_root)
    data_path = root / "DRIVE/training/images"
    ids = sorted(
        match.group(1)
        for path in data_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        for match in [re.match(r"^(\d+)", path.stem)]
        if match is not None
    )
    if not ids:
        raise FileNotFoundError(f"No DRIVE training images found in {data_path}")
    return _partition(ids, seed)
