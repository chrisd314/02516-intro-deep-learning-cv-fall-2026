from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATA_PATH = Path("/dtu/datasets1/02516/PH2_Dataset_images")
_RESAMPLING = getattr(Image, "Resampling", Image)


def _normalise_case_id(value):
    value = str(value).upper()

    if value.startswith("IMD"):
        value = value[3:]

    if not value.isdigit():
        raise ValueError(f"Invalid PH2 case ID: {value}")

    return f"IMD{int(value):03d}"


def _binary_mask_tensor(mask_path, pil_size):
    with Image.open(mask_path) as source:
        mask = source.convert("L").resize(
            pil_size, resample=_RESAMPLING.NEAREST
        )

    array = np.asarray(mask, dtype=np.uint8)
    # Lesion foreground is nonzero; nearest-neighbor resizing preserves labels.
    array = (array > 0).astype(np.float32)
    return torch.from_numpy(array.copy()).unsqueeze(0)


class PH2Dataset(Dataset):
    def __init__(self, split, sample_ids=None, image_transform=None, size=256,
                 split_seed=42, data_root=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        if sample_ids is None:
            from lib.splits import ph2_splits
            sample_ids = ph2_splits(split_seed, data_root=data_root)[split]

        self.split = split
        self.image_transform = image_transform
        self.height, self.width = (
            (size, size) if isinstance(size, int) else tuple(size)
        )
        self.pil_size = (self.width, self.height)

        data_path = DATA_PATH if data_root is None else Path(data_root) / "PH2_Dataset_images"
        if not data_path.is_dir():
            raise FileNotFoundError(f"PH2 dataset folder not found: {data_path}")

        # Build the available-case map from the actual folders. IDs are not
        # consecutive, so don't generate them from a numeric range.
        case_dirs = {
            path.name: path
            for path in data_path.iterdir()
            if path.is_dir() and re.fullmatch(r"IMD\d+", path.name)
        }

        selected_ids = [_normalise_case_id(case_id) for case_id in sample_ids]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("sample_ids contains duplicate case IDs")

        unknown_ids = sorted(set(selected_ids) - set(case_dirs))
        if unknown_ids:
            raise ValueError(f"PH2 case folders not found: {unknown_ids}")

        self.samples = []
        for case_id in selected_ids:
            case_dir = case_dirs[case_id]
            image_path = (
                case_dir
                / f"{case_id}_Dermoscopic_Image"
                / f"{case_id}.bmp"
            )
            target_path = (
                case_dir
                / f"{case_id}_lesion"
                / f"{case_id}_lesion.bmp"
            )

            if not image_path.is_file():
                raise FileNotFoundError(f"PH2 image not found: {image_path}")
            if not target_path.is_file():
                raise FileNotFoundError(f"PH2 lesion mask not found: {target_path}")

            self.samples.append((case_id, image_path, target_path))
        self.sample_ids = [sample[0] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        case_id, image_path, target_path = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        with Image.open(target_path) as source:
            target_original = source.convert("L")

        if image.size != target_original.size:
            raise ValueError(
                f"Image and lesion-mask sizes differ for {case_id}: "
                f"{image.size} vs {target_original.size}"
            )

        image = image.resize(
            self.pil_size, resample=_RESAMPLING.BILINEAR
        )
        target = _binary_mask_tensor(target_path, self.pil_size)

        if self.image_transform is None:
            array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        else:
            image_tensor = self.image_transform(image)

        valid_region = torch.ones_like(target)

        return {
            "image": image_tensor,
            "target": target,
            "valid_region": valid_region,
            "image_id": case_id,
        }
