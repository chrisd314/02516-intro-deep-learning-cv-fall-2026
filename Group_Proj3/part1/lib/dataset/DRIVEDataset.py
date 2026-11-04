from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATA_PATH = Path("/dtu/datasets1/02516/DRIVE")
_RESAMPLING = getattr(Image, "Resampling", Image)


def _files_by_id(folder, extensions):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        match = re.match(r"^(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Cannot find a numeric image ID in {path.name}")

        sample_id = match.group(1)
        if sample_id in files:
            raise ValueError(f"Duplicate ID {sample_id} in {folder}")
        files[sample_id] = path

    return files


def _binary_mask_tensor(path, pil_size, expected_size=None):
    with Image.open(path) as source:
        if expected_size is not None and source.size != expected_size:
            raise ValueError(f"Image and mask sizes differ for {path}: "
                             f"{expected_size} vs {source.size}")
        mask = source.convert("L").resize(
            pil_size, resample=_RESAMPLING.NEAREST
        )

    array = np.asarray(mask, dtype=np.uint8)
    # Vessel foreground and valid FOV pixels are nonzero.
    array = (array > 0).astype(np.float32)
    return torch.from_numpy(array.copy()).unsqueeze(0)


class DRIVEDataset(Dataset):
    def __init__(
        self,
        split,
        sample_ids=None,
        image_transform=None,
        size=256,
        data_root=None,
    ):
        if split not in {"train", "val", "test", "official_test"}:
            raise ValueError(
                "split must be 'train', 'val', 'test', or 'official_test'"
            )

        self.split = split
        self.image_transform = image_transform
        self.height, self.width = (
            (size, size) if isinstance(size, int) else tuple(size)
        )
        self.pil_size = (self.width, self.height)
        data_path = DATA_PATH if data_root is None else Path(data_root) / "DRIVE"

        if split == "official_test":
            split_dir = data_path / "test"
            image_dir = split_dir / "images"
            fov_dir = split_dir / "mask"
            target_dir = None
        else:
            # The labeled 20-image set is the source for train/val/test metrics.
            if sample_ids is None:
                raise ValueError(
                    f"Pass fixed sample_ids for the labeled '{split}' split"
                )
            split_dir = data_path / "training"
            image_dir = split_dir / "images"
            target_dir = split_dir / "1st_manual"
            fov_dir = split_dir / "mask"

        image_files = _files_by_id(
            image_dir, {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        )
        fov_files = _files_by_id(
            fov_dir, {".gif", ".png", ".bmp", ".tif", ".tiff"}
        )

        if set(image_files) != set(fov_files):
            raise ValueError("DRIVE image IDs and FOV-mask IDs do not match")

        if target_dir is not None:
            target_files = _files_by_id(
                target_dir, {".gif", ".png", ".bmp", ".tif", ".tiff"}
            )
            if set(image_files) != set(target_files):
                raise ValueError(
                    "DRIVE image IDs and vessel-mask IDs do not match"
                )
            self.target_files = target_files
        else:
            self.target_files = None

        if sample_ids is None:
            selected_ids = sorted(image_files)
        else:
            # Keep IDs as strings; zfill preserves IDs such as "01".
            selected_ids = [str(value).zfill(2) for value in sample_ids]
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError("sample_ids contains duplicates")

            unknown_ids = sorted(set(selected_ids) - set(image_files))
            if unknown_ids:
                raise ValueError(f"Unknown DRIVE sample IDs: {unknown_ids}")

        self.samples = [
            (
                sample_id,
                image_files[sample_id],
                fov_files[sample_id],
                None if self.target_files is None
                else self.target_files[sample_id],
            )
            for sample_id in selected_ids
        ]
        self.sample_ids = [sample[0] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, image_path, fov_path, target_path = self.samples[index]

        with Image.open(image_path) as source:
            original_size = source.size
            image = source.convert("RGB").resize(
                self.pil_size, resample=_RESAMPLING.BILINEAR
            )

        if self.image_transform is None:
            array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        else:
            image_tensor = self.image_transform(image)

        valid_region = _binary_mask_tensor(fov_path, self.pil_size, original_size)

        result = {
            "image": image_tensor,
            "valid_region": valid_region,
            "image_id": sample_id,
        }

        if target_path is not None:
            result["target"] = _binary_mask_tensor(
                target_path, self.pil_size, original_size
            )

        return result
