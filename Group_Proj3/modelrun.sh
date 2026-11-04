#!/bin/sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J idlcv-segmentation
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o modelrun.%J.out
#BSUB -e modelrun.%J.err

set -eu

PROJECT_DIR="/zhome/bf/0/233720/IDLCV/Group_Proj3"
cd "$PROJECT_DIR"

ENVIRONMENT="/zhome/bf/0/233720/IDLCV/~venvs/IDLCV_venv/bin/activate"
if [ ! -f "$ENVIRONMENT" ]; then
    echo "Environment activation script not found: $ENVIRONMENT" >&2
    exit 1
fi
. "$ENVIRONMENT"

for dataset in PH2 DRIVE
do
    for model in encdec unet
    do
        for loss in bce focal weighted_bce
        do
            echo "Training dataset=$dataset model=$model loss=$loss"
            python train.py \
                --dataset "$dataset" \
                --model "$model" \
                --loss "$loss" \
                --size 128 \
                --batch-size 6 \
                --epochs 20 \
                --lr 1e-3 \
                --split-seed 42 \
                --num-workers 0
        done
    done
done