# Reasoning4GenRec-Reproduction

论文 **"Reasoning over Semantic IDs Enhances Generative Recommendation"（SIDReasoner, KDD 2026）** 的复现仓库，同时复现论文中的基线方法 **TIGER** 与 **SASRec**。

- 论文：<https://arxiv.org/abs/2603.23183>
- 官方代码：<https://github.com/HappyPointer/SIDReasoner>（Zenodo: [10.5281/zenodo.20508816](https://doi.org/10.5281/zenodo.20508816)）

## 复现范围

| 维度 | 内容 |
|---|---|
| 方法 | **SIDReasoner**（主）、**TIGER**、**SASRec**（基线） |
| 数据集 | Amazon Reviews 2023：**Video Games**、**Office Products** |
| 预处理 | 5-core 过滤；滑窗截断（max len = 10）；按时序 8:1:1 切分 train/val/test |
| 指标 | 全目录排名下的 Recall@{5,10}、NDCG@{5,10} |

SIDReasoner 是两阶段框架：

1. **Enriched SID–Language Alignment**：RQ-VAE 将 item 元数据量化为 Semantic ID（SID）；扩展 LLM 词表加入 SID token；在多任务（item 预测 / SID 翻译）+ 教师模型（GPT-4o-mini）合成的增强语料上做多任务对齐训练；
2. **Reinforced Reasoning Enhancement**：先做 1 epoch 的冷启动激活 SFT（固定 reason-then-recommend 输出格式），再用 verl 实现的 GRPO 做结果驱动强化学习（前缀奖励 `1/2^(L-m)` + 格式奖励）。

## 总体流水线

```
              Amazon Reviews 2023（Games / Office）
                            │
                ┌───────────▼───────────┐
                │  1. r4gr.data         │ → DatasetBundle（5-core / 滑窗=10 / 时序 8:1:1）
                └───────────┬───────────┘
        ┌───────────────────┼─────────────────────┐
        ▼                   │                     ▼
┌───────────────┐           │            ┌────────────────┐
│ 2. r4gr.semid │           │            │ 3. SASRec      │
│  RQ-VAE       │           │            │  训练 + 评估    │
└───────┬───────┘           │            └────────┬───────┘
   ┌────┴─────┐             │                     │
   ▼          ▼             │                     │
┌──────┐ ┌─────────────────────────────────┐     │
│TIGER │ │ 4. SIDReasoner                  │     │
│      │ │  4a. synthesis（teacher 语料合成）│     │
│      │ │  4b. alignment（多任务 SFT）      │     │
│      │ │  4c. activation（1 epoch SFT）   │     │
│      │ │  4d. GRPO（verl）                │     │
└──┬───┘ └───────────────┬─────────────────┘     │
   │                     │                       ▼
   │                     │             ┌──────────────────┐
   └─────────────────────┴───────────► │   5. r4gr.eval    │ → results/（复现 vs 论文）
                                         │  全目录排名       │
                                         └──────────────────┘
```

阶段之间**只通过文件工件通信**（`data/` 下的 parquet、`artifacts/` 下的 SidTable / jsonl 语料 / checkpoint）：任何阶段可独立重跑、断点续跑、单独替换实现。

## 目录结构

```
Reasoning4GenRec-Reproduction/
├── README.md
├── .gitignore
├── .env.example            # 教师模型 API key 模板（复制为 .env，不入库）
├── pyproject.toml          # 依赖与打包（src 布局；extras: dev / teacher / grpo）
├── Makefile                # 常用任务入口
├── configs/
│   ├── data/               # games.yaml / office.yaml（路径、5-core、滑窗、切分比例）
│   ├── model/              # sasrec.yaml / tiger.yaml / rqvae.yaml / sidreasoner.yaml
│   ├── stage/              # synthesis / alignment / activation / grpo 的训练超参
│   └── exp/                # 实验组合（data × model × stage 的完整引用）
├── src/r4gr/
│   ├── data/               # DatasetBundle 构建（深模块）
│   ├── semid/              # RQ-VAE 训练 + SidTable + 合法前缀 Trie（深模块）
│   ├── models/
│   │   ├── sasrec/         # 判别式基线
│   │   └── tiger/          # 生成式检索基线（编码-解码 + 约束束搜）
│   ├── reasoner/           # SIDReasoner 全流水线
│   │   ├── corpus/         # 多任务模板 + 教师语料合成（teacher adapter 在此）
│   │   ├── alignment/      # 词表扩展 + 多任务 SFT
│   │   ├── activation/     # 冷启动激活 SFT（1 epoch）
│   │   └── grpo/           # verl 集成 + reward 定义
│   ├── train/              # SASRec / TIGER 的通用训练循环
│   ├── eval/               # 模型无关的评估（全目录排名）
│   └── cli.py              # 唯一命令行入口
├── scripts/                # 端到端流水线脚本（如 run_games_sasrec.sh）
├── tests/                  # 按接口测试（不 mock 内部实现）
├── results/                # 入库：复现指标 vs 论文指标的对照表
├── data/                   # 不入库：raw / interim / processed
├── artifacts/              # 不入库：SidTable、扩展词表、合成语料、checkpoint
└── runs/                   # 不入库：日志、tensorboard / wandb
```

## 模块设计

设计原则：**深模块**——小接口后面藏尽可能多的行为。每个模块一行列出调用方需要知道的全部（接口），复杂度全部留在实现里：

| 模块 | 接口（调用方需要知道的全部） | 隐藏在实现里的复杂度 |
|---|---|---|
| `r4gr.data` | `build(cfg) -> DatasetBundle`（交互序列、train/val/test、item 元数据、item 目录） | 下载、5-core、按时间排序、滑窗截断、8:1:1 切分、去重 |
| `r4gr.semid` | `fit(meta, cfg) -> SidTable`；`assign(item) -> codes[L]`；`trie`（合法 SID 前缀） | 文本编码器、RQ-VAE 训练、残差量化、码本管理、碰撞处理 |
| `r4gr.models.sasrec` | Ranker 契约：`rank(context) -> 全目录得分` | 自注意力编码 + item embedding + softmax |
| `r4gr.models.tiger` | 同上 Ranker 契约 | 编码-解码 transformer + Trie 约束束搜 |
| `r4gr.reasoner.corpus` | `build_tasks(bundle, sid) -> jsonl`；`enrich(bundle, sid, teacher) -> jsonl` | 多任务模板（item 预测 / SID 翻译）、教师提示词、teacher adapter（OpenAI 兼容 API / 本地 vLLM） |
| `r4gr.reasoner.alignment` | `train(cfg, corpus) -> ckpt`（含扩展词表） | 词表扩展、全参微调、早停（patience=2，按 eval loss 选点） |
| `r4gr.reasoner.activation` | `train(cfg, ckpt, sft_data) -> ckpt` | reason-then-recommend 模板 |
| `r4gr.reasoner.grpo` | `train(cfg, ckpt) -> ckpt`；`Reward: (rollout, gt) -> float` | verl 集成、rollout 采样、前缀奖励 + 格式奖励 |
| `r4gr.eval` | `evaluate(ranker, bundle, cutoffs) -> MetricsTable` | 全目录排名、Recall/NDCG 计算、落盘 |
| `r4gr.cli` | `r4gr <stage> --config ...` | 配置解析、日志、分布式启动 |

**接缝（seam）位置**：

- **工件接缝**：各阶段以文件为界面，天然可独立重跑与缓存；
- **Ranker 接缝**：三个模型满足同一 Ranker 契约 → 评估器与评估测试只写一份，三个消费者共享同一接缝；
- **Teacher 接缝**：教师模型在 `reasoner/corpus` 内部，adapter 对接 OpenAI 兼容 API 与本地 vLLM——两个 adapter，真实接缝，离线可替换；
- **Reward 接缝**：reward 是纯函数 `(rollout, ground_truth) -> float`，不依赖任何在线状态，可直接单元测试。

测试策略：全部通过模块接口测试，不穿透到内部实现——数据管道校验切分无泄漏与时序正确性；RQ-VAE 校验重建误差下降与 SID 唯一性；reward 校验前缀奖励公式与格式奖励的 trie 命中；评估器用玩具目录对照手算指标；LLM 各阶段用 2 层小模型做冒烟测试。

## 环境安装

```bash
git clone <this-repo>
cd Reasoning4GenRec-Reproduction
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # 基础 + 测试
pip install -e ".[teacher]"    # 教师语料合成（openai sdk）
pip install -e ".[grpo]"       # GRPO 阶段（verl、vllm）
cp .env.example .env           # 填入教师模型 API key
```

## 快速开始

目标命令行接口（随实现逐步可用）：

```bash
# 1. 数据
r4gr data build --config configs/data/games.yaml

# 2. 语义 ID（TIGER 与 SIDReasoner 共用）
r4gr semid fit --config configs/exp/games_semid.yaml

# 3. 基线
r4gr train sasrec --config configs/exp/games_sasrec.yaml
r4gr train tiger --config configs/exp/games_tiger.yaml

# 4. SIDReasoner 四个阶段
r4gr synth enrich --config configs/exp/games_synth.yaml
r4gr train alignment --config configs/exp/games_alignment.yaml
r4gr train activation --config configs/exp/games_activation.yaml
r4gr train grpo --config configs/exp/games_grpo.yaml

# 5. 评估与汇总
r4gr eval --config configs/exp/games_sasrec.yaml --ckpt runs/.../best.pt
r4gr report                      # 汇总 runs/ -> results/
```

Office 数据集将上述命令中的 `games` 换为 `office`。

## 与论文对齐的关键设置

| 项 | 取值 |
|---|---|
| 底座模型 | Qwen3-1.7B，全参数微调 |
| SID 构造 | RQ-VAE 残差量化；文本编码器与码本配置以官方代码为准 |
| 对齐训练 | AdamW，batch size 1024，早停 patience=2（按 eval loss 选点） |
| 激活阶段 | 1 epoch SFT |
| GRPO（verl） | rollout 数 16；KL 系数 1e-3；batch size 256；lr 5e-7；格式奖励权重 λ=0.1 |
| 奖励函数 | `R = 1/2^(L-m) + 0.1 * format`，m 为最长正确前缀长度 |
| 教师模型 | GPT-4o-mini（可用 OpenAI 兼容 API 替代） |
| 评估 | 全目录排名（非负采样），Recall/NDCG@{5,10} |

### 复现目标（论文 Table 2，Games / Office）

| 模型 | Games R@5 | Games N@5 | Games R@10 | Games N@10 | Office R@5 | Office N@5 | Office R@10 | Office N@10 |
|---|---|---|---|---|---|---|---|---|
| SASRec | 0.0501 | 0.0345 | 0.0723 | 0.0416 | 0.1019 | 0.0824 | 0.1167 | 0.0871 |
| TIGER | 0.0489 | 0.0300 | 0.0763 | 0.0402 | 0.1270 | 0.1037 | 0.1429 | 0.1121 |
| SIDReasoner | 0.0710 | 0.0460 | 0.1031 | 0.0563 | 0.1373 | 0.1119 | 0.1648 | 0.1208 |

复现结果将记录在 `results/` 下，与上表逐项对照。

## 测试

```bash
pytest tests/ -x
```

## 路线图

- [ ] Phase 0：仓库骨架、pyproject、配置系统、CI
- [ ] Phase 1：`r4gr.data`（Games / Office 就绪，切分无泄漏测试通过）
- [ ] Phase 2：SASRec + `r4gr.eval`（第一条端到端基线跑通）
- [ ] Phase 3：`r4gr.semid` + TIGER
- [ ] Phase 4：SIDReasoner 语料合成 → 对齐训练 → 激活训练
- [ ] Phase 5：GRPO（verl）
- [ ] Phase 6：结果汇总、与论文指标对齐分析

## 引用

```bibtex
@inproceedings{he2026sidreasoner,
  title     = {Reasoning over Semantic IDs Enhances Generative Recommendation},
  author    = {He, Yingzhi and Sun, Yan and Tan, Junfei and Chen, Yuxin and Kong, Xiaoyu and Shen, Chunxu and Wang, Xiang and Zhang, An and Chua, Tat-Seng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026}
}
```
