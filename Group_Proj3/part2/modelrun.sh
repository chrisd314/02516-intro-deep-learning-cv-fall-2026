#!/bin/bash
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J idlcv-points
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o part2.%J.out
#BSUB -e part2.%J.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${LS_SUBCWD:-$PWD}}"
cd "$PROJECT_DIR"
if [[ ! -f run_experiments.py || ! -f lib/points.py ]]; then
    echo "Set PROJECT_DIR to Group_Proj3/part2, not the Part 1 folder." >&2
    exit 1
fi

# Supply the course's module names if the virtual environment needs them.
if [[ -n "${MODULES:-}" ]]; then
    read -r -a task_modules <<< "$MODULES"
    module load "${task_modules[@]}"
fi
if [[ -n "${ENVIRONMENT:-}" ]]; then
    if [[ ! -f "$ENVIRONMENT" ]]; then
        echo "Environment activation file not found: $ENVIRONMENT" >&2
        exit 1
    fi
    source "$ENVIRONMENT"
fi
python -c "import torch, torchvision, numpy, PIL, matplotlib; assert torch.cuda.is_available(), 'No CUDA GPU available in this job'"

DATA_ROOT="${DATA_ROOT:-/dtu/datasets1/02516}"
RUN_DIR="${RUN_DIR:-runs/part2_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}}"
read -r -a task_counts <<< "${CLICK_COUNTS:-1 5 10 20 50}"
read -r -a task_strategies <<< "${STRATEGIES:-uniform interior boundary}"
task_args=(--data-root "$DATA_ROOT" --output-dir "$RUN_DIR" --require-cuda
    --model "${MODEL:-unet}" --size "${SIZE:-128}" --batch-size "${BATCH_SIZE:-6}"
    --epochs "${EPOCHS:-20}" --lr "${LR:-0.001}" --seed "${SEED:-42}"
    --split-seed "${SPLIT_SEED:-42}" --click-seed "${CLICK_SEED:-42}"
    --num-workers "${NUM_WORKERS:-0}" --num-threads 4 --margin "${MARGIN:-3}"
    --strategy-clicks "${STRATEGY_CLICKS:-10}"
    --click-counts "${task_counts[@]}" --strategies "${task_strategies[@]}")
if [[ -n "${REFERENCE_RECORD:-}" ]]; then
    task_args+=(--reference-record "$REFERENCE_RECORD")
fi

# Python creates a fresh run directory; do not mkdir RUN_DIR in advance.
python -u run_experiments.py "${task_args[@]}"
