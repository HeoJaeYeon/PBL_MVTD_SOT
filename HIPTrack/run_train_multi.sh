#!/bin/bash
# Multi-GPU (DDP) training.
#
#   bash run_train_multi.sh                          # all GPUs visible to this shell
#   GPUS=0,1,2,3 bash run_train_multi.sh             # select GPUs
#   CONFIG=hiptrack_got GPUS=0,1 bash run_train_multi.sh
#
# NCCL settings are not forced. Set them only if your machine needs them, e.g.
#   NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 bash run_train_multi.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SCRIPT="${SCRIPT:-hiptrack}"
CONFIG="${CONFIG:-hiptrack_got}"
# checkpoints/logs -> ckpt/{checkpoints,logs}, tensorboard -> ckpt/tensorboard (lib/train/admin/local.py)
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/ckpt}"
USE_LMDB="${USE_LMDB:-0}"
USE_WANDB="${USE_WANDB:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# GPUs: GPUS > CUDA_VISIBLE_DEVICES > all GPUs
if [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  elif command -v nvidia-smi > /dev/null; then
    NPROC_PER_NODE=$(nvidia-smi -L | grep -c "^GPU")
  else
    echo "[ERROR] cannot detect GPUs: set GPUS=0,1,... or NPROC_PER_NODE" >&2
    exit 1
  fi
fi

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[INFO] NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[INFO] Config: $SCRIPT/$CONFIG"
echo "[INFO] Save dir: $SAVE_DIR"

mkdir -p "$SAVE_DIR"

python3 tracking/train.py \
  --script "$SCRIPT" \
  --config "$CONFIG" \
  --save_dir "$SAVE_DIR" \
  --mode multiple \
  --nproc_per_node "$NPROC_PER_NODE" \
  --use_lmdb "$USE_LMDB" \
  --use_wandb "$USE_WANDB"
