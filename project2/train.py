"""Train/evaluate one model on one dataset variant.

Example:
    python train.py --model early --dataset leaky --seed 0

Model selection uses the validation set (best epoch); the test set is only
evaluated once with the selected checkpoint. Results are appended to
results/results.csv and per-run outputs go to results/runs/<run>/.
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import load_split, prepare_rgb, prepare_flow
from models import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))

# per-model defaults (epochs are over videos, except perframe which iterates over frames)
DEFAULTS = {
    'perframe':    dict(epochs=10, bs=64, lr=3e-4),
    'late':        dict(epochs=30, bs=16, lr=3e-4),
    'early':       dict(epochs=30, bs=16, lr=3e-4),
    '3d_scratch':  dict(epochs=60, bs=16, lr=1e-3),
    '3d_inflated': dict(epochs=30, bs=16, lr=3e-4),
    'flow':        dict(epochs=40, bs=16, lr=3e-4),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--dataset', default='leaky', choices=['leaky', 'noleak'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int)
    p.add_argument('--bs', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--data_dir', default=os.path.dirname(HERE))
    p.add_argument('--cache_dir', default=os.path.join(HERE, 'cache'))
    p.add_argument('--out_dir', default=os.path.join(HERE, 'results'))
    a = p.parse_args()
    for k, v in DEFAULTS[a.model].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def flow_std_of(train):
    f = train['flows'].float()
    f = f - f.mean(dim=(3, 4), keepdim=True)
    return f.std().item()


class Runner:
    def __init__(self, args, device):
        self.a = args
        self.device = device
        self.use_flow = args.model == 'flow'
        self.splits = {}
        for s in ['train', 'val', 'test']:
            d = load_split(args.dataset, s, args.data_dir, args.cache_dir)
            if self.use_flow and 'flows' not in d:
                raise SystemExit(f'dataset {args.dataset} has no optical flow')
            key = 'flows' if self.use_flow else 'frames'
            self.splits[s] = (d[key].to(device), d['labels'].to(device))
            if s == 'train' and self.use_flow:
                self.flow_std = flow_std_of(d)

    def inputs(self, x, train):
        if self.use_flow:
            return prepare_flow(x, train, self.flow_std)
        return prepare_rgb(x, train)

    def train_batches(self, frame_level):
        x, y = self.splits['train']
        N, T = x.shape[:2]
        if frame_level:   # every frame is an independent sample
            idx = torch.randperm(N * T, device=self.device)
            for i in range(0, len(idx), self.a.bs):
                j = idx[i:i + self.a.bs]
                clip = x[j // T, j % T].unsqueeze(1)                  # [B, 1, 3, S, S]
                yield self.inputs(clip, True)[:, 0], y[j // T]
        else:
            idx = torch.randperm(N, device=self.device)
            for i in range(0, N, self.a.bs):
                j = idx[i:i + self.a.bs]
                yield self.inputs(x[j], True), y[j]

    @torch.no_grad()
    def predict(self, model, split, bs=16):
        """Video-level class probabilities [N, C] (+ frame-level probs for perframe)."""
        model.eval()
        x, y = self.splits[split]
        probs, frame_probs = [], []
        for i in range(0, len(x), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(self.inputs(x[i:i + bs], False)).float()
            if model.kind == 'frame':
                p = out.softmax(-1)                                   # [B, T, C]
                frame_probs.append(p)
                probs.append(p.mean(1))
            else:
                probs.append(out.softmax(-1))
        probs = torch.cat(probs)
        fp = torch.cat(frame_probs) if frame_probs else None
        return probs, y, fp


def accuracy(probs, y):
    return (probs.argmax(-1) == y).float().mean().item()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')

    run = f'{a.dataset}_{a.model}_s{a.seed}'
    run_dir = os.path.join(a.out_dir, 'runs', run)
    os.makedirs(run_dir, exist_ok=True)

    r = Runner(a, device)
    model = MODELS[a.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    frame_level = model.kind == 'frame'

    steps_per_epoch = -(-len(r.splits['train'][0]) * (10 if frame_level else 1) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)

    history, best = [], (-1.0, -1, None)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot_loss = tot_correct = tot_n = 0
        for xb, yb in r.train_batches(frame_level):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += (logits.argmax(-1) == yb).sum().item()
            tot_n += len(yb)
        vp, vy, _ = r.predict(model, 'val')
        val_loss = F.nll_loss(vp.clamp_min(1e-8).log(), vy).item()
        val_acc = accuracy(vp, vy)
        history.append(dict(epoch=ep, train_loss=tot_loss / tot_n, train_acc=tot_correct / tot_n,
                            val_loss=val_loss, val_acc=val_acc))
        print(f'[{run}] ep {ep:3d}  train loss {tot_loss / tot_n:.3f} acc {tot_correct / tot_n:.3f}'
              f'  | val loss {val_loss:.3f} acc {val_acc:.3f}', flush=True)
        if val_acc > best[0]:
            best = (val_acc, ep, copy.deepcopy(model.state_dict()))
    train_time = time.time() - t0

    model.load_state_dict(best[2])
    vp, vy, _ = r.predict(model, 'val')
    tp, ty, tfp = r.predict(model, 'test')
    test_acc = accuracy(tp, ty)
    frame_acc = (tfp.argmax(-1) == ty[:, None]).float().mean().item() if tfp is not None else ''
    print(f'[{run}] best epoch {best[1]}  val {best[0]:.3f}  TEST {test_acc:.3f}'
          + (f'  (single-frame test acc {frame_acc:.3f})' if tfp is not None else ''))

    np.save(os.path.join(run_dir, 'val_probs.npy'), vp.cpu().numpy())
    np.save(os.path.join(run_dir, 'test_probs.npy'), tp.cpu().numpy())
    np.save(os.path.join(run_dir, 'test_labels.npy'), ty.cpu().numpy())
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(dict(args=vars(a), history=history), f, indent=1)

    row = dict(dataset=a.dataset, model=a.model, seed=a.seed, epochs=a.epochs, lr=a.lr, bs=a.bs,
               best_epoch=best[1], val_acc=round(best[0], 4), test_acc=round(test_acc, 4),
               test_frame_acc=round(frame_acc, 4) if frame_acc != '' else '',
               params_M=round(n_params / 1e6, 2), train_time_s=round(train_time, 1))
    csv_path = os.path.join(a.out_dir, 'results.csv')
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == '__main__':
    main()
