#!/usr/bin/env bash
# Train one arm with GRAFT's SFT trainer (Qwen3-32B recipe from sft/scripts/launch_gr_32b_split.sh),
# 4 epochs, adapter checkpoints every 10% of optimizer steps (plus 25%).
#
# Usage: train_arm.sh <arm> <data-dir> [extra train_gr.py args...]
#   arm = noint    -> --no-routing, plain AdamW (the paper's "No intervention" baseline)
#   arm = <other>  -> GRAFT: classic routing, split-moment AdamW, kappa 2, 1% anchors from unlabeled
# Env: GRAFT_DIR (jnward/GRAFT checkout), GRAFT_SFT_CKPT_DIR, MODEL (default Qwen/Qwen3-32B),
#      EPOCHS (default 4), NPROC (default 8), STEP_SIZE (default 48).
set -euo pipefail
ARM=$1; DATA_DIR=$(realpath "$2"); shift 2

GRAFT_DIR=${GRAFT_DIR:-/root/work/GRAFT}
MODEL=${MODEL:-Qwen/Qwen3-32B}
EPOCHS=${EPOCHS:-4}
NPROC=${NPROC:-8}
STEP_SIZE=${STEP_SIZE:-48}
SAVE_FRACS=${SAVE_FRACS:-0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
export GRAFT_SFT_DATA_DIR=$DATA_DIR
export GRAFT_SFT_CKPT_DIR=${GRAFT_SFT_CKPT_DIR:-/workspace/ckpts}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$GRAFT_DIR/sft:${PYTHONPATH:-}"

COMMON=(--model-name "$MODEL" --d-retain 64 --d-forget 64 --match-rslora --lora-alpha 32 --lora-r 16
        --lr 5e-4 --epochs "$EPOCHS" --warmup-ratio 0.05 --max-length 16384 --filter-overlong
        --step-size "$STEP_SIZE" --no-evals --save-fracs "$SAVE_FRACS" --run-name "$ARM")
if [[ $ARM == noint ]]; then
  ARM_FLAGS=(--no-routing)
else
  ARM_FLAGS=(--classifier-retain-recall 0.01 --retain-from-unlabeled --unclassified-trains-both
             --split-moment --kappa 2)
fi

cd "$GRAFT_DIR/sft"
mkdir -p logs
LOG=logs/train_${ARM}_$(date +%Y%m%d_%H%M%S).log
echo "arm=$ARM data=$DATA_DIR ckpts=$GRAFT_SFT_CKPT_DIR/$ARM log=$LOG"
accelerate launch --config_file scripts/accelerate_ddp.yaml --num_processes "$NPROC" \
  scripts/gr/train_gr.py "${COMMON[@]}" "${ARM_FLAGS[@]}" "$@" 2>&1 | tee "$LOG"
