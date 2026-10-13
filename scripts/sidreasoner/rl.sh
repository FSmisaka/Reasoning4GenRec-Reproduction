#!/usr/bin/env bash
# SIDReasoner Stage 3: GRPO 强化学习(需要 SIDReasoner 官方 verl fork)
# 前置: 克隆 https://github.com/HappyPointer/SIDReasoner 到某个目录,
# 并将 config/sidreasoner.py 中 rl.verl_home 指向该目录(或用
# VERL_HOME 环境变量覆盖), 详见 README 的 SIDReasoner 小节。
set -euo pipefail
set -x
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

PY=${PY:-.venv/bin/python}
VERL_HOME=${VERL_HOME:-$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['rl']['verl_home'])")}
if [[ ! -d "${VERL_HOME}/verl" ]]; then
    echo "错误: 未找到 verl fork (${VERL_HOME}/verl)。" >&2
    echo "请克隆 SIDReasoner 官方仓库并在 config/sidreasoner.py 中设置 rl.verl_home。" >&2
    exit 1
fi

read -r NNODES TRAIN_BS MAX_PROMPT MAX_RESP LR ROLLOUT_N KL_COEF \
    EPOCHS SAVE_FREQ TEST_FREQ PROJECT USE_WANDB \
    FILTER_OVERLONG USE_REMOVE_PAD ACTOR_MB USE_KL_LOSS KL_LOSS_TYPE \
    ENTROPY_COEF GRAD_CKPT ACTOR_PARAM_OFFLOAD ACTOR_OPT_OFFLOAD \
    LOGPROB_MB TP_SIZE GPU_MEM_UTIL REF_PARAM_OFFLOAD USE_KL_REWARD \
    CRITIC_WARMUP < <(${PY} - <<'EOF'
from config import load_config
rl = load_config("sidreasoner")["rl"]
print(rl["nnodes"], rl["train_batch_size"],
      rl["max_prompt_length"], rl["max_response_length"], rl["lr"],
      rl["rollout_n"], rl["kl_loss_coef"], rl["total_epochs"],
      rl["save_freq"], rl["test_freq"], rl["project_name"],
      "True" if rl.get("use_wandb", True) else "False",
      "True" if rl.get("filter_overlong_prompts", True) else "False",
      "True" if rl.get("use_remove_padding", True) else "False",
      rl.get("actor_micro_batch_size_per_gpu", 8),
      "True" if rl.get("use_kl_loss", True) else "False",
      rl.get("kl_loss_type", "low_var_kl"),
      rl.get("entropy_coeff", 0),
      "True" if rl.get("enable_gradient_checkpointing", True) else "False",
      "True" if rl.get("actor_param_offload", False) else "False",
      "True" if rl.get("actor_optimizer_offload", False) else "False",
      rl.get("log_prob_micro_batch_size_per_gpu", 8),
      rl.get("tensor_model_parallel_size", 1),
      rl.get("gpu_memory_utilization", 0.8),
      "True" if rl.get("ref_param_offload", True) else "False",
      "True" if rl.get("use_kl_in_reward", False) else "False",
      rl.get("critic_warmup", 0))
EOF
)

# GPU 选择: 显式指定优先; 否则自动挑选至多 NGPUS_WANT 张空闲卡,
# n_gpus_per_node 以实际挑选到的卡数为准
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(bash scripts/sidreasoner/pick_gpus.sh "${NGPUS_WANT:-4}")
  export CUDA_VISIBLE_DEVICES
  echo ">>> 未指定 GPU, 自动选择空闲卡: ${CUDA_VISIBLE_DEVICES}"
fi
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
if [[ "${USE_WANDB}" == "True" ]]; then
    LOGGER="['console','wandb']"
else
    LOGGER="['console']"
fi

CATEGORY=$(${PY} -c "from config import load_config; import os; print(load_config('sidreasoner')['data']['categories'][os.environ['DATASET']])")
ROOT=$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['data']['root'])")
DATA_DIR=$(${PY} -c "from config import load_config; rl = load_config('sidreasoner')['rl']; print(rl['data_dir'] or '')" 2>/dev/null || true)
DATA_DIR=${DATA_DIR:-${ROOT}/rec_reasoning_verl/${CATEGORY}}
STAGE2_CKPT="runs/sidreasoner/${CATEGORY}_stage2_activation/final_checkpoint"
EXPERIMENT_NAME="${CATEGORY}_stage3_rl_Qwen3-1.7B"
REWARD_PATH="$(pwd)/src/reward/direct_recommendation_steprule.py"

export PYTHONPATH="${VERL_HOME}:${PYTHONPATH:-}"
mkdir -p ./logs

${PY} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$(pwd)/${DATA_DIR}/train.parquet" \
    data.val_files="$(pwd)/${DATA_DIR}/test.parquet" \
    data.train_batch_size=${TRAIN_BS} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    data.filter_overlong_prompts=${FILTER_OVERLONG} \
    data.truncation='error' \
    actor_rollout_ref.model.path="$(pwd)/${STAGE2_CKPT}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PAD} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MB} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEF} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRAD_CKPT} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPT_OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
    algorithm.use_kl_in_reward=${USE_KL_REWARD} \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.logger="${LOGGER}" \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name="rule_base_reward" \
    trainer.project_name="${PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${EPOCHS} "$@" \
    2>&1 | tee "./logs/${EXPERIMENT_NAME}.log"
