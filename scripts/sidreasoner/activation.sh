#!/usr/bin/env bash
# SIDReasoner Stage 2: 推理激活 SFT(多卡 torchrun, 输入为 Stage 1 checkpoint)
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NGPUS=${NGPUS:-4}
fi
MASTER_PORT=${MASTER_PORT:-29519}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python -m torch.distributed.run \
    --nproc_per_node "${NGPUS}" --master_port "${MASTER_PORT}" \
    -m src.train.sidreasoner_activation
