"""Export PH2 binary predictions using the exact split/model settings of a run."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from lib.dataset.PH2Dataset import PH2Dataset
from lib.records import load_record
from plot_results import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True, help="Completed .pth.json sidecar beside its weights.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing outputs are protected.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    record = load_record(args.record)
    data_root = args.data_root or record["config"].get("data_root")
    dataset = PH2Dataset(args.split, record["splits"][args.split],
                         size=record["config"]["size"], data_root=data_root)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(record, args.record, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        for index, image_id in enumerate(dataset.sample_ids):
            image, _ = dataset.load_image(index)
            probability = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0]
            prediction = (probability >= 0.5).cpu().numpy().astype(np.uint8) * 255
            with (args.output_dir / f"{image_id}.png").open("xb") as handle:
                Image.fromarray(prediction).save(handle, format="PNG")
    print(f"Saved {len(dataset)} masks to {args.output_dir}")


if __name__ == "__main__":
    main()
