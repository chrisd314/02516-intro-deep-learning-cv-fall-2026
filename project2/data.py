"""Data loading for Project 2.

Videos are small (10 frames, 9 flows), so each split is decoded once, resized and
cached as a single tensor file. Training then keeps the whole split on the GPU and
does augmentation there, which avoids slow DataLoader workers on Windows.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

N_FRAMES = 10
N_FLOWS = N_FRAMES - 1
CACHE_SIZE = 128   # stored resolution; random/center crop to CROP_SIZE
CROP_SIZE = 112

DATASETS = {
    'leaky': 'datasets/ufc101',              # original Kaggle splits (subject leakage), in the repo
    'noleak': 'datasets/ucf101_noleakage',   # subject-disjoint splits + RAFT flows (not in git, see README)
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _load_split(root, split, with_flow):
    df = pd.read_csv(os.path.join(root, 'metadata', f'{split}.csv'))
    frames, flows = [], []
    for video_path in df.video_path:
        rel = os.path.splitext(video_path)[0]   # e.g. BodyWeightSquats/v_BodyWeightSquats_g01_c01
        fdir = os.path.join(root, 'frames', split, rel)
        clip = []
        for i in range(1, N_FRAMES + 1):
            img = Image.open(os.path.join(fdir, f'frame_{i}.jpg')).convert('RGB')
            img = img.resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
            clip.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
        frames.append(torch.stack(clip))                       # [T, 3, S, S] uint8
        if with_flow:
            odir = os.path.join(root, 'flows', split, rel)
            fl = np.stack([np.load(os.path.join(odir, f'flow_{i}_{i + 1}.npy'))
                           for i in range(1, N_FRAMES)])        # [9, 2, 224, 224]
            fl = torch.from_numpy(fl)
            scale = CACHE_SIZE / fl.shape[-1]                  # displacements shrink with the image
            fl = F.interpolate(fl, size=(CACHE_SIZE, CACHE_SIZE), mode='area') * scale
            flows.append(fl.half())
    out = {
        'frames': torch.stack(frames),                         # [N, T, 3, S, S] uint8
        'labels': torch.tensor(df.label.values, dtype=torch.long),
        'names': list(df.video_name),
        'actions': list(df.action),
    }
    if with_flow:
        out['flows'] = torch.stack(flows)                      # [N, 9, 2, S, S] fp16
    return out


def load_split(dataset, split, base_dir, cache_dir):
    """Return a dict with frames/labels(/flows) for one split, using a cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{dataset}_{split}_{CACHE_SIZE}.pt')
    if os.path.exists(cache):
        return torch.load(cache)
    root = os.path.join(base_dir, DATASETS[dataset])
    data = _load_split(root, split, with_flow=os.path.isdir(os.path.join(root, 'flows')))
    torch.save(data, cache)
    return data


def class_names(base_dir):
    df = pd.read_csv(os.path.join(base_dir, DATASETS['leaky'], 'metadata', 'train.csv'))
    return list(df.drop_duplicates('label').sort_values('label').action)


# ---------------------------------------------------------------- GPU augmentation
def _crop_offsets(n, train, device):
    if train:
        return (torch.randint(0, CACHE_SIZE - CROP_SIZE + 1, (n,), device=device),
                torch.randint(0, CACHE_SIZE - CROP_SIZE + 1, (n,), device=device))
    c = (CACHE_SIZE - CROP_SIZE) // 2
    return torch.full((n,), c, device=device), torch.full((n,), c, device=device)


def _crop(x, ys, xs):
    """x: [B, T, C, S, S]; same crop for all frames of a clip."""
    idx = torch.arange(CROP_SIZE, device=x.device)
    rows = (ys[:, None] + idx)[:, None, None, :, None]          # [B,1,1,K,1]
    cols = (xs[:, None] + idx)[:, None, None, None, :]          # [B,1,1,1,K]
    b = torch.arange(x.shape[0], device=x.device)[:, None, None, None, None]
    t = torch.arange(x.shape[1], device=x.device)[None, :, None, None, None]
    c = torch.arange(x.shape[2], device=x.device)[None, None, :, None, None]
    return x[b, t, c, rows, cols]


def prepare_rgb(frames_u8, train):
    """uint8 [B, T, 3, S, S] -> normalized float [B, T, 3, K, K] with clip-consistent aug."""
    B = frames_u8.shape[0]
    ys, xs = _crop_offsets(B, train, frames_u8.device)
    x = _crop(frames_u8, ys, xs).float() / 255.0
    if train:
        flip = torch.rand(B, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None, None], x.flip(-1), x)
        # light brightness/contrast jitter, same for the whole clip
        b = 1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device=x.device) - 0.5)
        c = 1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device=x.device) - 0.5)
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        x = ((x - m) * c + m) * b
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def prepare_flow(flows_f16, train, flow_std):
    """fp16 [B, 9, 2, S, S] -> float [B, 9, 2, K, K].

    Mean displacement is subtracted per flow field (simple camera-motion
    compensation, as in Simonyan & Zisserman) and values are scaled by a global std.
    """
    B = flows_f16.shape[0]
    ys, xs = _crop_offsets(B, train, flows_f16.device)
    x = _crop(flows_f16, ys, xs).float()
    x = x - x.mean(dim=(3, 4), keepdim=True)
    if train:
        flip = torch.rand(B, device=x.device) < 0.5
        flipped = x.flip(-1)
        flipped[:, :, 0] = -flipped[:, :, 0]   # horizontal flip negates the u (x) component
        x = torch.where(flip[:, None, None, None, None], flipped, x)
    return x / flow_std
