# Reasoning4GenRec 复现

调参直接改 `config/` 下与模型同名的配置文件, 命令行不接受超参;
数据集通过目标名选择(`games` / `office` / `industrial`)。
模型同名文件在 `src/models/`(模型)、`src/train/`(训练入口)、`scripts/`(启动脚本)中一一对应。

# 1. 环境

```bash
source /thuir/wangyiyao/miniconda3/etc/profile.d/conda.sh
conda create -p /thuir/wangyiyao/conda_envs/r4gr python=3.11
conda activate /thuir/wangyiyao/conda_envs/r4gr
python3 -m venv .venv && source .venv/bin/activate
make setup
```

# 2. 基线模型

```bash
make games-sasrec          # SASRec
make games-sasrec-mini     # SASRec(MiniOneRec 实现)
make games-tiger           # TIGER
make games-caser           # Caser(WSDM'18)
make games-gru4rec         # GRU4Rec(RecSys'18)
make smoke                 # 冒烟测试
```

office / industrial 把 `games` 换掉即可。

# 3. SIDReasoner(KDD'26)

以 games 为例(office / industrial 同名替换), 论文 Table 2 的最终结果
= 三个阶段全部完成后, 以 Stage 3 模型做思考模式评估:

```bash
# 0) 一次性准备
make download-model        # Qwen3-1.7B
make install-vllm          # 思考模式评估需要

# 1) 三阶段训练(4 卡)
make games-sidreasoner-sft             # Stage 1: 增强式 SID-语言对齐 SFT
make games-sidreasoner-activation      # Stage 2: 推理激活 SFT
make games-sidreasoner-rl              # Stage 3: GRPO 强化学习(前置见下)

# 2) 合并 Stage 3 的 FSDP 分片 checkpoint
make games-sidreasoner-merge

# 3) 思考模式评估 + 计算指标
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
make games-sidreasoner-metrics         # 自动取最新 *_think_*.json, 可用 RESULT=xxx.json 指定

# 中间对照(非思考模式): MODEL 取 stage1 / stage2 / 任意 checkpoint 路径
make games-sidreasoner-eval MODEL=stage2
```

不想从头训练: 下载[官方 checkpoint](https://huggingface.co/Sober-Clever/SIDReasoner-Models)
后跳过步骤 1)-2), 直接跑 think + metrics。

## Stage 3 前置(一次性)

```bash
git clone https://github.com/HappyPointer/SIDReasoner ../SIDReasoner   # verl fork
make install-rl-env        # 或直接用官方 docker 镜像(推荐):
                           # hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0
```

- RL 的初始策略固定为 Stage 2 输出, 脚本自动引用, 无需配置。
- verl fork 位置默认 `../SIDReasoner`, 可用 config 的 `rl.verl_home` 或 `VERL_HOME=... make games-sidreasoner-rl` 覆盖。
- docker 镜像内运行: `PY=python3 bash scripts/sidreasoner/rl.sh`。
- 不想用 wandb: 把 config 的 `rl.use_wandb`、`training.*.report_to` 关掉。

## 论文参考值(Table 2, R@5 / N@5 / R@10 / N@10)

| 数据集 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| games | 0.0710 | 0.0460 | 0.1031 | 0.0563 |
| office | 0.1373 | 0.1119 | 0.1648 | 0.1208 |
| industrial | 0.1109 | 0.0905 | 0.1438 | 0.1010 |

# 4. 备忘

```bash
make games-sidreasoner-rl-data                          # 重新生成 RL parquet(默认用官方数据, 一般无需)
make games-sidreasoner-merge EVAL_INTERVAL=100          # 只合并特定步数间隔
make games-sidreasoner-sft CUDA_VISIBLE_DEVICES=0,1,2,3 # 指定训练用卡(不指定则自动选空闲卡)
make games-sidreasoner-eval CUDA_LIST="0 1"             # 指定评估用卡(默认 0 1)
DATASET=games bash scripts/sidreasoner/sft.sh           # 等价的手动运行方式
DATASET=games .venv/bin/python -m src.train.tiger
```

- 训练数据已随仓库就位(`data/raw/Amazon/`), 无需下载。
- Stage 1/2 输出在 `runs/sidreasoner/`, Stage 3 在 `checkpoints/RecRL_Reasoning/`, 评估结果在 `results/sidreasoner/`。
- 8 种对齐任务的 prompt 模板与 GPT-4o-mini 语料增强 prompt 见 `docs/sidreasoner_prompts.md`。
