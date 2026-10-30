"""
SIDReasoner 模型配置

Reasoning over Semantic IDs Enhances Generative Recommendation
(He et al., KDD 2026, arXiv 2603.23183)
原始实现: https://github.com/HappyPointer/SIDReasoner

三阶段管线(数据集通过 DATASET 环境变量选择):
- Stage 1 SID-语言对齐 SFT:  src/train/sidreasoner_sft.py,
  超参对应原始 sft_Qwen3.py + sft_Qwen3_enrich.sh 默认值
- Stage 2 推理激活 SFT:      src/train/sidreasoner_activation.py,
  对应 sft_reasoning_activation.py + .sh 默认值
- Stage 3 GRPO 强化学习:      scripts/sidreasoner/rl.sh,
  需要 SIDReasoner 官方 verl fork(见 README), 超参对应
  RL_training_script.sh 默认值

前置准备(见 README 的 SIDReasoner 小节):
- 下载 Qwen/Qwen3-1.7B 模型, 将 model.base_model 指向其路径
- 安装 peft/datasets/wandb; 思考模式评估与 Stage 3 另需
  vllm 与 verl fork
- 训练/评估数据全部位于 data/raw/Amazon/(与官方数据集目录一致,
  无需再从 Google Drive 下载)

修改超参数直接编辑本文件,训练脚本不接受任何命令行超参。
"""

from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "data": {
        "root": "data/raw/Amazon",
        "categories": {
            "games": "Video_Games",
            "office": "Office_Products",
            "industrial": "Industrial_and_Scientific",
        },
    },
    "model": {
        "base_model": "/thuir/wangyiyao/Qwen3-1.7B",
    },
    "training": {
        "sft": {
            "batch_size": 1024,
            "micro_batch_size": 1,
            "epochs": 10,
            "lr": 3e-4,
            "cutoff_len": 1024,
            "warmup_steps": 20,
            "patience": 3,
            "save_total_limit": 10,
            "seed": 42,
            "sample": -1,
            "general_max_len": 3072,
            "general_sample": 60000,
            "mask_assistant": True,
            "train_new_token_embeddings_only": False,
            "gradient_checkpointing": True,
            "wandb_project": "SIDReasoner",
            "report_to": "wandb",
            "out": None,
        },
        "activation": {
            "batch_size": 1024,
            "micro_batch_size": 2,
            "epochs": 5,
            "lr": 1e-5,
            "cutoff_len": 1024,
            "warmup_steps": 10,
            "patience": 5,
            "save_total_limit": 10,
            "seed": 42,
            "sample": -1,
            "mask_assistant": True,
            "train_new_token_embeddings_only": False,
            "gradient_checkpointing": True,
            "wandb_project": "SIDReasoner",
            "report_to": "wandb",
            "base_model": None,
            "out": None,
        },
    },
    "rl": {
        "verl_home": "../SIDReasoner",
        "use_wandb": True,
        "n_gpus_per_node": 4,  # 实际卡数由 rl.sh 按所选 GPU 自动判定, 此项不生效
        "nnodes": 1,
        # ---- 算法超参(默认值 = 官方 RL_training_script.sh) ----
        "train_batch_size": 256,
        "max_prompt_length": 1024,
        "max_response_length": 1024,
        "lr": 5e-7,
        "rollout_n": 16,
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "use_kl_loss": True,
        "use_kl_in_reward": False,
        "entropy_coeff": 0,
        "critic_warmup": 0,
        "filter_overlong_prompts": True,
        "total_epochs": 10,
        "save_freq": 100,
        "test_freq": 50,
        # ---- 工程参数(显存/并行, 可按硬件调整; 默认值 = 官方脚本) ----
        "actor_micro_batch_size_per_gpu": 8,
        "log_prob_micro_batch_size_per_gpu": 8,  # 同时作用于 rollout 与 ref 两处
        "actor_param_offload": False,
        "actor_optimizer_offload": False,
        "ref_param_offload": True,
        "gpu_memory_utilization": 0.8,
        "tensor_model_parallel_size": 1,
        "enable_gradient_checkpointing": True,
        "use_remove_padding": True,
        "project_name": "RecRL_Reasoning",
        "data_dir": None,
        # ---- Stage 3 奖励(rl.sh 据此选择 reward 文件与是否开 beam search) ----
        # steprule: 论文原版(采样 SID 0.25/0.5/1.0 步进 + 0.1 格式分), 默认;
        # rollout_beam: OneRec-Think Rollout-Beam(arXiv:2510.11639 Eq.6-7),
        #   每条 rollout 的 CoT 之后做宽度 K 的 beam search 取 K 个候选 SID,
        #   奖励 = 候选中与 ground truth 匹配 code 数(共 L 层)的最大值。
        #   需配合修改过的 SIDReasoner verl fork(rollout 侧 beam search 基建)。
        "reward": "steprule",
        "sid_beam_size": 4,  # rollout_beam: beam 宽度 K, 必须 >1 才会在 rollout 后做 beam search
        "sid_length": 3,     # rollout_beam: SID 层数 L(<a_*><b_*><c_*> 为 3)
        # rollout_beam: 转发给 src/reward/sid_rewards/ 的构造参数:
        "reward_kwargs": {
            "reward_type": "onerec_rollout_beam",
            "match_mode": "position",   # position=论文式逐层比对; prefix=前缀连续匹配(StepRule 式对照)
            "normalize": False,         # 匹配数除以 L, 奖励归一到 [0,1]
            "format_reward_weight": 0.0,  # 采样 SID 合法性的附加格式分(论文无此项, >0 时按 DATASET 自动找 info 文件)
        },
    },
    "evaluation": {
        "num_beams": 10,
        "batch_size": 8,
        "max_new_tokens": 256,
        "think_max_new_tokens": 256,
        "length_penalty": 0.0,
        "padding_side": "left",
        "seed": 42,
        "eval_model": "stage1",
        "think_eval_model": None,
        "think_batch_size": 4,
        "k_list": [1, 5, 10],
    },
}
