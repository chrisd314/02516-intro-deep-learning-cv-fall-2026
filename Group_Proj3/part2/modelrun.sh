#!/bin/bash
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J idlcv-points
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o /zhome/bf/0/233720/IDLCV/Proj3_Git/part2/part2.%J.out
#BSUB -e /zhome/bf/0/233720/IDLCV/Proj3_Git/part2/part2.%J.err

set -euo pipefail

PROJECT_DIR="/zhome/bf/0/233720/IDLCV/Proj3_Git/part2"
cd "$PROJECT_DIR"
if [[ ! -f run_experiments.py || ! -f lib/points.py ]]; then
    echo "Set PROJECT_DIR to Group_Proj3/part2, not the Part 1 folder." >&2
    exit 1
fi

ENVIRONMENT="/zhome/bf/0/233720/IDLCV/~venvs/IDLCV_venv/bin/activate"
if [[ ! -f "$ENVIRONMENT" ]]; then
    echo "Environment activation file not found: $ENVIRONMENT" >&2
    exit 1
fi
source "$ENVIRONMENT"
python -c "import torch, torchvision, numpy, PIL, matplotlib; assert torch.cuda.is_available(), 'No CUDA GPU available in this job'"

DATA_ROOT="${DATA_ROOT:-/dtu/datasets1/02516}"
RUN_DIR="${RUN_DIR:-runs/part2_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}}"
VISUAL_DIR="${RUN_DIR}_visualizations"
read -r -a task_counts <<< "${CLICK_COUNTS:-1 5 10 20 50}"
read -r -a task_strategies <<< "${STRATEGIES:-uniform interior boundary}"
python -u visualize_points.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$VISUAL_DIR" \
    --size "${SIZE:-128}" \
    --split-seed "${SPLIT_SEED:-42}" \
    --click-seed "${CLICK_SEED:-42}" \
    --positive-clicks "${STRATEGY_CLICKS:-10}" \
    --negative-clicks "${STRATEGY_CLICKS:-10}" \
    --strategy "${VISUAL_STRATEGY:-uniform}" \
    --margin "${MARGIN:-3}"
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
