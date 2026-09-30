#!/usr/bin/env bash
# SIDReasoner Stage 1: SID-语言对齐 SFT(多卡 torchrun)
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NGPUS=${NGPUS:-4}
fi
MASTER_PORT=${MASTER_PORT:-12340}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

exec .venv/bin/python -m torch.distributed.run \
    --nproc_per_node "${NGPUS}" --master_port "${MASTER_PORT}" \
    -m src.train.sidreasoner_sft
