"""Deterministic train/validation/test splits for the project datasets."""

import random
from pathlib import Path
import re


def _partition(ids, seed, fractions=(0.7, 0.15, 0.15)):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n_train = int(len(ids) * fractions[0])
    n_val = int(len(ids) * fractions[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def ph2_splits(seed=42):
    data_path = Path("/dtu/datasets1/02516/PH2_Dataset_images")
    ids = sorted(
        path.name
        for path in data_path.iterdir()
        if path.is_dir() and re.fullmatch(r"IMD\d+", path.name)
    )
    if not ids:
        raise FileNotFoundError(f"No PH2 case folders found in {data_path}")
    return _partition(ids, seed)


def drive_splits(seed=42):
    data_path = Path("/dtu/datasets1/02516/DRIVE/training/images")
    ids = sorted(
        match.group(1)
        for path in data_path.iterdir()
        if path.is_file()
        for match in [re.match(r"^(\d+)", path.stem)]
        if match is not None
    )
    if not ids:
        raise FileNotFoundError(f"No DRIVE training images found in {data_path}")
    return _partition(ids, seed)
