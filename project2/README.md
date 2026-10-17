# Project 2 – Video classification (UCF-101 subset)

## Layout
| File | Purpose |
|---|---|
| `data.py` | Loads a split from the CSVs, resizes frames to 128 px (flows too) and caches it as one tensor file in `cache/`. GPU-side augmentation (clip-consistent random 112 crop, h-flip, brightness/contrast; flow flip negates *u*). |
| `models.py` | `perframe`, `late`, `early`, `3d_scratch`, `3d_inflated`, `flow` (temporal stream). All 2D models use an ImageNet ResNet-18. |
| `train.py` | Trains one model on one dataset variant, selects the best epoch on **val**, evaluates **test** once, appends to `results/results.csv`, saves `results/runs/<run>/`. |
| `run_all.ps1` | Every experiment × 3 seeds, locally (skips finished runs). |
| `make_figures.py` | Mean ± std tables (`results/tables.tex`, `results/summary.csv`), two-stream score fusion, figures in `results/figures/`. |
| `jobscript.sh` | Same thing as an HPC batch job (`bsub -app c02516_1g.10gb < jobscript.sh`). |

## Data (paths relative to the repo root)
```
datasets/ufc101/              # original Kaggle splits ("leaky") – already in the repo
datasets/ucf101_noleakage/    # subject-disjoint splits + RAFT flows – NOT in git (2.7 GB)
```
Get the no-leak data from the HPC (`/dtu/datasets1/02516/ucf101_noleakage.zip`) and unzip it into
`datasets/`, e.g. from your laptop:
```
scp <studentid>@login1.hpc.dtu.dk:/dtu/datasets1/02516/ucf101_noleakage.zip .
tar -xf ucf101_noleakage.zip -C ../datasets      # run from project2/
```
On the HPC itself just link it: `ln -s /dtu/datasets1/02516/ucf101_noleakage ../datasets/ucf101_noleakage`.

The first run decodes each split once into `project2/cache/` (~1.1 GB, git-ignored, ~2 min).

## Run (from project2/)
```
python train.py --model early --dataset leaky --seed 0
powershell -File run_all.ps1 -Python path\to\python.exe   # everything, 3 seeds (~25 min on a laptop GPU)
python make_figures.py                                    # tables + figures from results/
```
`results/` contains the numbers (`results.csv`, `summary.csv`, `tables.tex`) and figures of all 33 runs used in
the report. Per-run predictions (`results/runs/`) are not in git, so re-drawing the figures requires retraining.
Model weights are not saved (only per-run predictions and training curves).

## Setup used for all runs
- Input: 10 frames, 112×112 crop (from 128). Flows: 9 fields × (u, v) = 18 channels, mean flow subtracted per field.
- AdamW (wd 0.05), one-cycle LR, label smoothing 0.1, dropout 0.5 before the classifier, bf16 autocast.
- LR 3e-4 for pretrained models, 1e-3 for 3D-from-scratch. Epochs: per-frame 10 (over frames, ≈ same number of steps as 30 video epochs), video models 30, 3D-scratch 60, flow 40.
- Model selection: best validation accuracy; test is evaluated once on that checkpoint.
- Two-stream: spatial = per-frame RGB model (10-frame average), temporal = flow stream; class scores averaged.
