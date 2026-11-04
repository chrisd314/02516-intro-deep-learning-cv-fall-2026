"""Generate binary masks for a dataset split using a trained checkpoint."""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.dataset.DRIVEDataset import DRIVEDataset
from lib.dataset.PH2Dataset import PH2Dataset
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.splits import drive_splits, ph2_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PH2", "DRIVE"), default="PH2")
    parser.add_argument("--model", choices=("encdec", "unet"), default="unet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--split", choices=("train", "val", "test", "official_test"), default="test")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "DRIVE" and args.split == "official_test":
        dataset = DRIVEDataset("official_test", size=args.size, image_transform=transforms.ToTensor())
    else:
        split_ids = ph2_splits(args.split_seed) if args.dataset == "PH2" else drive_splits(args.split_seed)
        cls = PH2Dataset if args.dataset == "PH2" else DRIVEDataset
        dataset = cls(args.split, split_ids[args.split], size=args.size, image_transform=transforms.ToTensor())
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    model = (EncDec() if args.model == "encdec" else UNet()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    index = 0
    with torch.no_grad():
        for batch in loader:
            probabilities = torch.sigmoid(model(batch["image"].to(device)))
            predictions = (probabilities >= 0.5).squeeze(1).cpu().numpy()
            for prediction in predictions:
                Image.fromarray((prediction.astype(np.uint8) * 255)).save(
                    os.path.join(args.output_dir, f"prediction_{index:03d}.png")
                )
                index += 1
    print(f"saved {index} masks to {args.output_dir}")


if __name__ == "__main__":
    main()
