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

read -r N_GPUS NNODES TRAIN_BS MAX_PROMPT MAX_RESP LR ROLLOUT_N KL_COEF \
    EPOCHS SAVE_FREQ TEST_FREQ PROJECT USE_WANDB < <(${PY} - <<'EOF'
from config import load_config
rl = load_config("sidreasoner")["rl"]
print(rl["n_gpus_per_node"], rl["nnodes"], rl["train_batch_size"],
      rl["max_prompt_length"], rl["max_response_length"], rl["lr"],
      rl["rollout_n"], rl["kl_loss_coef"], rl["total_epochs"],
      rl["save_freq"], rl["test_freq"], rl["project_name"],
      "True" if rl.get("use_wandb", True) else "False")
EOF
)
if [[ "${USE_WANDB}" == "True" ]]; then
    LOGGER="['console','wandb']"
else
    LOGGER="['console']"
fi

CATEGORY=$(${PY} -c "from config import load_config; import os; print(load_config('sidreasoner')['data']['categories'][os.environ['DATASET']])")
ROOT=$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['data']['root'])")
DATA_DIR=$(${PY} -c "from config import load_config; import os; rl = load_config('sidreasoner')['rl']; print(rl['data_dir'] or None)" 2>/dev/null || true)
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
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$(pwd)/${STAGE2_CKPT}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
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
