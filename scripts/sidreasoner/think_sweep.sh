#!/usr/bin/env bash
# 批量思考模式评测: 对一个实验目录下所有 global_step_*/actor_merged
# 从大到小逐 checkpoint 评测(带温度采样 + 多 seed 聚合由 config 决定)。
#
# 关键特性: 启动时把 config/sidreasoner.py 快照到 temp/config_snapshots/,
# 整个 sweep 期间经 CONFIG_SNAPSHOT_SIDREASONER 固定使用该快照 ——
# sweep 运行中修改 config/ 不影响本任务后续 checkpoint。
#
# 用法:
#   DATASET=games bash scripts/sidreasoner/think_sweep.sh \
#       checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B
# 可选环境变量:
#   FROM=1700 TO=100   步数范围过滤(默认全部)
#   CUDA_VISIBLE_DEVICES=0   指定 GPU(评测需独占一整卡)
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"
ROOT_DIR=${1:?用法: DATASET=games bash scripts/sidreasoner/think_sweep.sh <模型实验目录>}
FROM=${FROM:-999999}
TO=${TO:-0}
PY=.venv/bin/python

# ---- 快照 config, 本 sweep 全程固定 ----
SNAP_DIR=temp/config_snapshots
SNAP_FILE=$SNAP_DIR/sidreasoner_$(date +%Y%m%d_%H%M%S).py
mkdir -p "$SNAP_DIR"
cp config/sidreasoner.py "$SNAP_FILE"
export CONFIG_SNAPSHOT_SIDREASONER=$(pwd)/"$SNAP_FILE"
echo ">>> config 快照: $SNAP_FILE"
echo ">>> think_temperature=$($PY -c "from config import load_config; print(load_config('sidreasoner')['evaluation'].get('think_temperature'))")  eval_seeds=$($PY -c "from config import load_config; print(load_config('sidreasoner')['evaluation'].get('eval_seeds'))")"

# ---- 收集 checkpoint(从大到小, 按 FROM/TO 过滤) ----
STEPS=$(ls -d "$ROOT_DIR"/global_step_* 2>/dev/null \
    | sed 's/.*global_step_//' | sort -rn \
    | awk -v from="$FROM" -v to="$TO" '$1 <= from && $1 >= to')
[ -n "$STEPS" ] || { echo "未找到 global_step_* checkpoint: $ROOT_DIR"; exit 1; }
echo ">>> 待评测 checkpoint(共 $(echo "$STEPS" | wc -l | tr -d ' ') 个): $(echo $STEPS | tr '\n' ' ')"

# ---- 逐个评测(每个 checkpoint 独立进程, 用同一份快照 config) ----
for S in $STEPS; do
    MODEL="$ROOT_DIR/global_step_$S/actor_merged"
    [ -d "$MODEL" ] || { echo ">>> 跳过(无 actor_merged): $MODEL"; continue; }
    echo "===== [$(date '+%F %T')] think eval: $MODEL ====="
    DATASET=$DATASET MODEL=$MODEL $PY -m src.eval.sidreasoner_eval_think
done
echo "===== sweep 完成: $ROOT_DIR (config 快照: $SNAP_FILE) ====="
