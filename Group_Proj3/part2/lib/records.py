"""Small JSON and reference helpers shared by the standalone Part 2 scripts."""

import hashlib
import json
from pathlib import Path


def save_run_record(path, record, create=False):
    """Same strict JSON writer as Part 1; preserve occupied initial outputs."""
    contents = json.dumps(record, indent=2, allow_nan=False)
    with open(path, "x" if create else "w", encoding="utf-8") as handle:
        handle.write(contents + "\n")


def load_record(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError(f"Run is not completed: {path}")
    return record


def validate_splits(splits, available):
    if set(splits) != {"train", "val", "test"} or not all(splits.values()):
        raise ValueError("Need nonempty train, val and test ID lists")
    flattened = [value for values in splits.values() for value in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Dataset split IDs overlap or contain duplicates")
    if set(flattened) != set(available):
        raise ValueError("Reference split IDs do not match the available PH2 dataset")


def validate_reference(record, config):
    reference = record["config"]
    if reference.get("dataset") != "PH2" or reference.get("loss") != "bce":
        raise ValueError("Reference must be a PH2 fully supervised BCE run")
    if reference.get("supervision", "full") != "full":
        raise ValueError("Reference must be fully supervised")
    for key in ("model", "size", "batch_size", "epochs", "lr", "seed", "split_seed", "num_workers"):
        if reference.get(key) != config.get(key):
            raise ValueError(f"Reference mismatch for {key}: {reference.get(key)!r} != {config.get(key)!r}")
    if record.get("metric_aggregation") != "global_valid_pixels":
        raise ValueError("Reference must use global valid-pixel metrics")
    if record.get("loss_aggregation") != "mean_valid_pixels":
        raise ValueError("Reference must use valid-pixel loss averaging")
    if record.get("prediction_threshold") != 0.5 or record.get("metric_smoothing") != 1e-7:
        raise ValueError("Reference metric threshold/smoothing differ")
    expected_preprocessing = {"size": config["size"], "image_range": [0, 1],
                              "image_resize": "bilinear", "mask_resize": "nearest",
                              "mask_foreground": "nonzero", "augmentation": "none"}
    if record.get("preprocessing") != expected_preprocessing:
        raise ValueError("Reference preprocessing differs from the Part 2 configuration")


def checkpoint_path(record, record_path):
    # Prefer the colocated weights when an HPC run directory has been copied.
    local = Path(record_path).with_suffix("")
    if local.suffix == ".pth" and local.is_file():
        return local
    configured = Path(record["config"]["checkpoint"])
    if not configured.is_absolute():
        configured = Path(record["working_directory"]) / configured
    if not configured.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {configured}; keep it beside its .pth.json")
    return configured


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
