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
            # ---- GEM 变体(Stage 1: src.train.sidreasoner_sft_gem) ----
            # 把 SFT 的 CE 损失换成 GEM(ICLR 2025, arXiv:2408.16673,
            # 官方实现 github.com/liziniu/GEM): 带熵正则的逆 KL 分布
            # 匹配损失, 训练精度接近 CE 的同时保留输出多样性。
            # beta: 目标分布温度, 官方论文/脚本默认 0.7, 越小越保多样性,
            #   beta->1 退化为接近 CE 的行为;
            # h: 官方 Qwen 训练脚本用 "logsigmoid"(自适应权重), "linear"
            #   为等权重, 官方注明两者差异通常可忽略;
            # print_entropy: 训练日志附带 token 熵(监测多样性, 官方可选项)。
            # 评估/早停仍用 CE eval_loss(官方 GEM 行为), 与基线可比。
            "gem": {"beta": 0.7, "h": "logsigmoid", "print_entropy": True},
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
            # ---- GEM 变体(Stage 2: src.train.sidreasoner_activation_gem) ----
            # 含义同 training.sft.gem; Stage 2 GEM 默认从 GEM Stage 1 的
            # final_checkpoint 继续训练(显式配置过 training.sft.out 时需
            # 手动把 base_model 指向对应 GEM checkpoint)
            "gem": {"beta": 0.7, "h": "logsigmoid", "print_entropy": True},
        },
    },
    "rl": {
        "verl_home": "../SIDReasoner",
        "use_wandb": False,
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
        "actor_micro_batch_size_per_gpu": 32,
        "log_prob_micro_batch_size_per_gpu": 64,  # 同时作用于 rollout 与 ref 两处(纯前向, 不影响反向峰值)
        "actor_param_offload": False,
        "actor_optimizer_offload": False,
        "ref_param_offload": True,
        "gpu_memory_utilization": 0.50,
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
        # rollout_beam_div: 在 rollout_beam 基础上乘 OneReason 多样性因子
        #   R_div = max(0, m^(1)-1)/(K-1)(arXiv:2606.06260 Eq.9), m^(1) 为
        #   K 个候选第 1 层 code 去重个数; 实验名自动加 _rolloutbeam_div。
        #   steprule/rollout_beam/rollout_beam_div 均需修改过的 SIDReasoner
        #   verl fork(rollout 侧 beam search 基建; steprule 不需要)。
        "reward": "steprule",
        "sid_beam_size": 4,  # rollout_beam*: beam 宽度 K, 必须 >1 才会在 rollout 后做 beam search
        "sid_length": 3,     # rollout_beam*: SID 层数 L(<a_*><b_*><c_*> 为 3)
        # rollout_beam*: 转发给 src/reward/sid_rewards/ 的构造参数。
        # 注意: reward_type 默认由 rl.sh 按 rl.reward 自动下发
        # (steprule 不发 / rollout_beam -> onerec_rollout_beam /
        #  rollout_beam_div -> onereason_div_rollout_beam);
        # 只有接入自定义 reward 变体时才在 reward_kwargs 里显式写
        # "reward_type": "my_reward"(显式值优先于分支默认), 见 README。
        "reward_kwargs": {
            "match_mode": "position",   # position=论文式逐层比对; prefix=前缀连续匹配(StepRule 式对照)
            "normalize": False,         # 匹配数除以 L, 奖励归一到 [0,1]
            "format_reward_weight": 0.0,  # 采样 SID 合法性的附加格式分(论文无此项, >0 时按 DATASET 自动找 info 文件)
            "div_level": 1,             # rollout_beam_div: 计算去重的 SID 层(1=首层 <a_*>, OneReason 论文设定)
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
        # ---- 思考模式(CoT)解码设置(参考 OneRec-Think 官方 Stage 评测脚本
        # test/eval_parallel_8gpu_cot.sh: --think_temperature 1.5 --think_top_p 0.95) ----
        # think_temperature > 0 时 CoT 用采样解码(vLLM per-request seed 可复现),
        # 并按 eval_seeds 逐 seed 评测后报告均值/标准差/95% 置信区间;
        # 0.0 保持原贪心解码行为(单 seed, 输出文件名不带 _seed 后缀)。
        "think_temperature": 0.0,
        "think_top_p": 0.95,
        "eval_seeds": [42, 43, 44, 45, 46],
        "ci_level": 0.95,
        # 多 seed 评测默认只写 *_summary.json(已含逐 seed 指标);
        # 需要单 seed 明细文件(含 predict/cot 字段, 供多样性分析等
        # 导入)时置 True, 会额外写 <前缀>_seed<seed>.json
        "save_per_seed_results": False,
    },
}
