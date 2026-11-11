#!/usr/bin/env bash
# SIDReasoner Stage 1(GEM): SID-语言对齐 SFT 的 GEM 损失变体(多卡 torchrun)
# GPU 选择规则同 sft.sh; 默认 NGPUS_WANT=2(适配 2x A800-80G 服务器),
# 全局 batch_size 保持 config 中的 1024 不变, 梯度累积按卡数自动调整。
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(bash scripts/sidreasoner/pick_gpus.sh "${NGPUS_WANT:-2}")
  export CUDA_VISIBLE_DEVICES
  echo ">>> 未指定 GPU, 自动选择空闲卡: ${CUDA_VISIBLE_DEVICES}"
fi
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
MASTER_PORT=${MASTER_PORT:-12341}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python -m torch.distributed.run \
    --nproc_per_node "${NGPUS}" --master_port "${MASTER_PORT}" \
    -m src.train.sidreasoner_sft_gem
