#!/bin/sh
### Run one experiment on the DTU HPC (course queue). From the project2/ folder, submit with:
###   bsub -app c02516_1g.10gb < jobscript.sh
### Change MODEL / DATASET / SEED below (or copy the file per experiment).
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J p2_train
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o p2_%J.out
#BSUB -e p2_%J.err

MODEL=early          # perframe | late | early | 3d_scratch | 3d_inflated | flow
DATASET=noleak       # leaky | noleak
SEED=0

# Expects a venv with torch/torchvision/pandas/pillow/matplotlib, e.g. ~/venvs/idlcv
source ~/venvs/idlcv/bin/activate

# The leaky data is in the repo (datasets/ufc101). For the no-leak data, link the course copy once:
#   ln -s /dtu/datasets1/02516/ucf101_noleakage ../datasets/ucf101_noleakage
# The decoded cache is ~1.1 GB; point --cache_dir somewhere with space if your home quota is tight.
python train.py --model $MODEL --dataset $DATASET --seed $SEED
