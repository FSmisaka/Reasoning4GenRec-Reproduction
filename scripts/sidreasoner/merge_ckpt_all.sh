#!/usr/bin/env bash
# SIDReasoner Stage 3: 批量合并所有 global_step_N 的 FSDP actor checkpoint
# 迁移自官方 merge_fsdp_ckpt_ALL.sh(改为调用本仓库的 merge_ckpt.py)
# 用法: DATASET=games [EVAL_INTERVAL=100] [CKPT_ROOT=...] [RL_SUFFIX=...] \
#       bash scripts/sidreasoner/merge_ckpt_all.sh
#   CKPT_ROOT 未设置时按 DATASET + 当前 SID 集后缀 + RL_SUFFIX 自动推导:
#   checkpoints/RecRL_Reasoning/<Category>_stage3_rl_Qwen3-1.7B[_own][RL_SUFFIX]
#   RL_SUFFIX 用于非默认 reward 的 Stage 3 产物(如 rollout_beam -> _rolloutbeam,
#   与 rl.sh 的实验名后缀一致; make 命令行变量会自动传入)。
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
if [[ -z "${CKPT_ROOT:-}" ]]; then
  : "${DATASET:?需要 DATASET 或 CKPT_ROOT 二者之一}"
  CATEGORY=$(${PY} -c "from config import load_config; import os; print(load_config('sidreasoner')['data']['categories'][os.environ['DATASET']])")
  SID_SUFFIX=$(${PY} -c "from config import load_config; from src.data.sid_switch import sid_suffix; print(sid_suffix(load_config('sidreasoner')['data']))")
  CKPT_ROOT="checkpoints/RecRL_Reasoning/${CATEGORY}_stage3_rl_Qwen3-1.7B${SID_SUFFIX}${RL_SUFFIX:-}"
fi
EVAL_INTERVAL=${EVAL_INTERVAL:-100}

if ! [[ "${EVAL_INTERVAL}" =~ ^[0-9]+$ ]] || (( EVAL_INTERVAL <= 0 )); then
    echo "EVAL_INTERVAL 必须为正整数: ${EVAL_INTERVAL}" >&2
    exit 1
fi
if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "目录不存在: ${CKPT_ROOT}" >&2
    exit 1
fi

matches=()
while IFS= read -r actor_dir; do
    step_dir="$(basename "$(dirname "${actor_dir}")")"
    if [[ "${step_dir}" =~ ^global_step_([0-9]+)$ ]]; then
        step="${BASH_REMATCH[1]}"
        if (( step % EVAL_INTERVAL == 0 )); then
            matches+=("${step}:${actor_dir}")
        fi
    fi
done < <(find "${CKPT_ROOT}" -maxdepth 2 -type d -name "actor" -path "*/global_step_*/*")

if [[ ${#matches[@]} -eq 0 ]]; then
    echo "未找到符合条件的 actor checkpoint(${CKPT_ROOT}, 每 ${EVAL_INTERVAL} 步)" >&2
    exit 1
fi

for entry in $(printf '%s\n' "${matches[@]}" | sort -t: -k1,1n); do
    step="${entry%%:*}"
    actor_dir="${entry#*:}"
    output_dir="${actor_dir}_merged"
    echo "merging step ${step}: ${actor_dir} -> ${output_dir}"
    ${PY} scripts/sidreasoner/merge_ckpt.py \
        --checkpoint "${actor_dir}" --output-dir "${output_dir}"
done
echo "all merges completed."
