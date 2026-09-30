# 环境

## conda

```bash
source /thuir/wangyiyao/miniconda3/etc/profile.d/conda.sh
conda create -p /thuir/wangyiyao/conda_envs/r4gr python=3.11
conda activate /thuir/wangyiyao/conda_envs/r4gr
```

## venv

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pandas numpy pyyaml
pip install peft datasets wandb  # SIDReasoner
```

# quick start

所有超参数都在 `config/` 下与模型同名的配置文件中定义,
训练脚本不接受任何命令行超参,调参直接改配置文件。
数据集通过 `DATASET` 环境变量选择(`games` / `office` / `industrial`, 见配置文件中的 `data.categories`)。
模型同名文件在 `src/models/`(模型)、`src/train/`(训练入口)、`scripts/<dataset>/`(启动脚本)中一一对应。

```bash
# SASRec
make games-sasrec
make office-sasrec
make industrial-sasrec

# SASRec(MiniOneRec 实现)
make games-sasrec-mini
make office-sasrec-mini
make industrial-sasrec-mini

# TIGER
make games-tiger
make office-tiger
make industrial-tiger

# 等价的手动运行方式
DATASET=games .venv/bin/python -m src.train.tiger
# 或 scripts/games/tiger.sh

# Caser(WSDM'18)
make games-caser
make office-caser
make industrial-caser

# GRU4Rec(RecSys'18)
make games-gru4rec
make office-gru4rec
make industrial-gru4rec

# SIDReasoner(KDD'26)
make games-sidreasoner-sft
make games-sidreasoner-activation
make games-sidreasoner-rl
make games-sidreasoner-eval

# 冒烟测试
make smoke
```

# SIDReasoner

迁移自 [SIDReasoner 官方仓库](https://github.com/HappyPointer/SIDReasoner)(KDD 2026, arXiv 2603.23183),
按本仓库的组织方式重新组织: 数据集类在 `src/data/sidreasoner_data.py`(原 `data_Qwen3.py` 原样迁移),
共享训练组件在 `src/train/sidreasoner_common.py`,三阶段入口在 `src/train/sidreasoner_{sft,activation}.py`
与 `scripts/sidreasoner/rl.sh`,评估在 `src/eval/sidreasoner_eval{,_think}.py`,超参数集中在 `config/sidreasoner.py`。

## 环境

Stage 1/2(SFT)与评估在现有 venv 基础上补充安装:

```bash
pip install peft datasets wandb
```

Stage 3(RL)基于官方 verl fork,推荐使用其测试过的镜像
`hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0`;
若自行搭建环境,官方依赖版本配方已迁移在
`scripts/sidreasoner/install_rl_env.sh`(torch 2.6 / vllm 0.8.5 /
flash-attn 2.7.4 / flashinfer 0.2.2 等),或参照
[verl 官方安装指南](https://verl.readthedocs.io/en/latest/start/install.html);
思考模式评估另需 `vllm`。

## 模型下载(登录服务器后执行)

```bash
# 基座模型(Qwen3-1.7B), 下载后把 config/sidreasoner.py 的
# model.base_model 指向本地路径(或保持 HF id 在线自动下载):
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B

# (可选)官方已训练好的各阶段 checkpoint, 可直接用于评估:
# https://huggingface.co/Sober-Clever/SIDReasoner-Models
```

数据无需下载: `data/raw/Amazon/` 即官方发布的数据集目录,
SFT 增强语料(item_enhanced_v2.json / integrated_narrative.csv /
general/sampled_data.arrow)与 RL parquet(rec_reasoning_verl/)均已就位。
RL parquet 如需重新生成: `DATASET=games .venv/bin/python -m src.data.sidreasoner_rl_data`。

## 训练(三阶段)

```bash
# Stage 1: 增强式 SID-语言对齐 SFT(多任务混合, 4 卡)
make games-sidreasoner-sft
# Stage 2: 推理激活 SFT(以 Stage 1 的 final_checkpoint 为基座)
make games-sidreasoner-activation
# Stage 3: GRPO 强化学习(需要 verl fork, 见下)
make games-sidreasoner-rl
```

Stage 1/2 输出位于 `runs/sidreasoner/<Category>_stage{1,2}_*/final_checkpoint`,
日志写入 wandb(把 config 中 `report_to` 改为 `"none"` 可关闭)。
数据集切换用 `games/office/industrial` 目标即可; 卡数通过
`CUDA_VISIBLE_DEVICES` 或 `NGPUS` 控制。

## Stage 3 前置配置(GRPO 强化学习)

`make games-sidreasoner-rl` 前需逐项完成:

1. **跑完 Stage 1/2**: RL 的初始策略是 Stage 2 的输出
   `runs/sidreasoner/<Category>_stage2_activation/final_checkpoint`,
   脚本会自动引用它, 无需配置, 但需保证该目录存在。
2. **克隆 verl fork**: RL 经由官方修改过的 verl 启动, 需克隆
   SIDReasoner 官方仓库(内含 `verl/`), 并二选一指明位置:
   - 改 `config/sidreasoner.py` 的 `rl.verl_home`(默认 `"../SIDReasoner"`,
     即本仓库同级目录), 或
   - 运行时用环境变量 `VERL_HOME=/path/to/SIDReasoner` 覆盖。
   脚本启动前会检查 `${verl_home}/verl` 是否存在, 不在会直接报错。
3. **安装 verl 运行环境**: 推荐直接用官方测试过的 Docker 镜像
   `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0`
   (镜像内用 `PY=python3 bash scripts/sidreasoner/rl.sh` 跑, `PY` 可覆盖);
   或参照 [verl 官方安装指南](https://verl.readthedocs.io/en/latest/start/install.html)
   在 venv 中装好 ray/vllm/flashinfer 等依赖后直接 make。
4. **wandb**(可选): `trainer.logger` 默认 `['console','wandb']`,
   不想用 wandb 就把 config 的 `rl.use_wandb` 改为 `False`。
5. **GPU**: 默认 `n_gpus_per_node=4`(原始配置), 单机多卡;
   batch/rollout/kl 等超参均在 `config/sidreasoner.py` 的 `rl` 节,与
   原始 `RL_training_script.sh` 默认值一致。

RL 数据(`data/raw/Amazon/rec_reasoning_verl/<Category>/{train,test}.parquet`)
与奖励函数(`src/reward/direct_recommendation_steprule.py`, stepwise 前缀
奖励 + 格式奖励)已就位, 无需额外准备。checkpoint 由 verl 保存在
`checkpoints/RecRL_Reasoning/<Category>_stage3_rl_Qwen3-1.7B/`。

## 复现论文指标(完整流程)

论文 Table 2 的 SIDReasoner 行对应的是 **三个阶段全部完成后、以
Stage 3 模型做思考模式评估** 的结果。只评 Stage 1/2 是中间对照,
不能代表论文最终指标。以 games 数据集为例:

```bash
# 0) 一次性准备(见上文): Qwen3-1.7B、peft/datasets/wandb、verl fork、vllm

# 1) Stage 1: 增强式 SID-语言对齐
make games-sidreasoner-sft

# 2) Stage 2: 推理激活
make games-sidreasoner-activation

# 3) Stage 3: GRPO 强化学习(耗时最长, 4 卡)
make games-sidreasoner-rl

# 4) 合并 Stage 3 的 FSDP 分片 checkpoint(取最后一个 global_step)
.venv/bin/python scripts/sidreasoner/merge_ckpt.py \
    --checkpoint checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor \
    --output-dir  checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged

# 5) 思考模式评估: 把 config/sidreasoner.py 的
#    evaluation.think_eval_model 指向 actor_merged 后运行
DATASET=games .venv/bin/python -m src.eval.sidreasoner_eval_think

# 6) 计算指标(输出 HR@K / NDCG@K, 即论文的 R@K / N@K)
DATASET=games .venv/bin/python -m src.eval.sidreasoner_metrics calc \
    --path results/sidreasoner/Video_Games_think_actor_merged.json
```

office / industrial 把 `games` 换成对应数据集名即可。论文参考值
(Table 2, R@5 / N@5 / R@10 / N@10):

| 数据集 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| games | 0.0710 | 0.0460 | 0.1031 | 0.0563 |
| office | 0.1373 | 0.1119 | 0.1648 | 0.1208 |
| industrial | 0.1109 | 0.0905 | 0.1438 | 0.1010 |

若不想从头训练, 可下载官方 checkpoint
([Sober-Clever/SIDReasoner-Models](https://huggingface.co/Sober-Clever/SIDReasoner-Models))
跳过 1)-4), 直接做 5)-6)(思考模式)。
需要按训练步评估多个 checkpoint 时, 可用批量合并工具:
`CKPT_ROOT=checkpoints/RecRL_Reasoning/<Category>_stage3_rl_Qwen3-1.7B EVAL_INTERVAL=100 bash scripts/sidreasoner/merge_ckpt_all.sh`。

8 种对齐任务的 prompt 模板与 GPT-4o-mini 语料增强 prompt
(复现/再生语料的唯一记录)见 `docs/sidreasoner_prompts.md`。

## 评估(中间对照)

评估哪个模型由 `config/sidreasoner.py` 的 `evaluation.eval_model` 决定,
可取 `"stage1"`、`"stage2"` 或 checkpoint 的完整路径(如官方发布模型),
命令本身不带任何参数:

```bash
# 1. 想评估哪个模型, 先改 config/sidreasoner.py 的一处配置:
#    evaluation.eval_model = "stage1"   # Stage 1 输出(默认)
#    evaluation.eval_model = "stage2"   # Stage 2 输出
#    evaluation.eval_model = "./models/SIDReasoner-Models/Office_Products_stage1"  # 任意路径

# 2. 然后直接运行(默认评估 Stage 1, 无需额外参数):
make games-sidreasoner-eval
make office-sidreasoner-eval
make industrial-sidreasoner-eval

# 等价的手动方式(用几张卡就写哪几张, 默认 0 1):
CUDA_LIST="0 1" DATASET=games bash scripts/sidreasoner/eval.sh
```

评估流程为: 测试集按卡拆分 -> 各卡并行 trie 约束束搜索(num_beams=10)
-> 合并 -> 打印 HR@K/NDCG@K, 结果 json 保存在
`results/sidreasoner/final_result_<Category>.json`。

思考模式评估(先 vLLM 生成思维链再约束束搜, 需 `pip install vllm`):
在 config 的 `evaluation.think_eval_model` 填入 checkpoint 路径后运行:

```bash
DATASET=games .venv/bin/python -m src.eval.sidreasoner_eval_think
```
