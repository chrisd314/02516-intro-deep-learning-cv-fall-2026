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

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
if [ ! -f train.py ]; then
    echo "Run from Group_Proj3/part1 or set PROJECT_DIR to its absolute path." >&2
    exit 1
fi

# Use the already activated course environment, or an explicitly supplied one.
if [ -n "${ENVIRONMENT:-}" ]; then
    if [ ! -f "$ENVIRONMENT" ]; then
        echo "Environment activation script not found: $ENVIRONMENT" >&2
        exit 1
    fi
    . "$ENVIRONMENT"
fi
python -c "import torch, torchvision, numpy, PIL, matplotlib"

DATA_ROOT="${DATA_ROOT:-/dtu/datasets1/02516}"
RUN_DIR="${RUN_DIR:-runs/part1_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "$(dirname "$RUN_DIR")"
# Exclusive directory creation stops accidental reuse of an experiment run.
mkdir "$RUN_DIR"
echo "Saving this run to $RUN_DIR"

for dataset in PH2 DRIVE
do
    for model in encdec unet
    do
        for loss in bce focal weighted_bce
        do
            dataset_lower=$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]')
            python -u train.py \
                --dataset "$dataset" \
                --model "$model" \
                --loss "$loss" \
                --size 128 \
                --batch-size 6 \
                --epochs 20 \
                --lr 1e-3 \
                --split-seed 42 \
                --seed 42 \
                --num-workers 0 \
                --num-threads 4 \
                --data-root "$DATA_ROOT" \
                --checkpoint "$RUN_DIR/${dataset_lower}_${model}_${loss}.pth"
        done
    done
done

python plot_results.py --records-dir "$RUN_DIR" --output-dir "$RUN_DIR/report"
