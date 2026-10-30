#!/bin/bash
# Single-GPU training.
#
#   bash run_train_single.sh                 # GPU 0
#   GPUS=1 bash run_train_single.sh          # select a GPU
#   CONFIG=hiptrack_got GPUS=1 bash run_train_single.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SCRIPT="${SCRIPT:-hiptrack}"
CONFIG="${CONFIG:-hiptrack_got}"
# checkpoints/logs -> ckpt/{checkpoints,logs}, tensorboard -> ckpt/tensorboard (lib/train/admin/local.py)
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/ckpt}"
USE_LMDB="${USE_LMDB:-0}"
USE_WANDB="${USE_WANDB:-0}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] Config: $SCRIPT/$CONFIG"
echo "[INFO] Save dir: $SAVE_DIR"

mkdir -p "$SAVE_DIR"

python3 tracking/train.py \
  --script "$SCRIPT" \
  --config "$CONFIG" \
  --save_dir "$SAVE_DIR" \
  --mode single \
  --use_lmdb "$USE_LMDB" \
  --use_wandb "$USE_WANDB"
