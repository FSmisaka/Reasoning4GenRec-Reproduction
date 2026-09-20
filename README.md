# Reasoning4GenRec-Reproduction

论文 **"Reasoning over Semantic IDs Enhances Generative Recommendation"（SIDReasoner, KDD 2026）** 的复现仓库，同时复现论文中的基线方法 **TIGER** 与 **SASRec**。

- 论文：<https://arxiv.org/abs/2603.23183>
- 官方代码：<https://github.com/HappyPointer/SIDReasoner>（Zenodo: [10.5281/zenodo.20508816](https://doi.org/10.5281/zenodo.20508816)）

## 复现范围

| 维度 | 内容 |
|---|---|
| 方法 | **SIDReasoner**（主）、**TIGER**、**SASRec**（基线） |
| 数据集 | Amazon Reviews 2018 5-core：**Video Games**、**Office Products**（SIDReasoner 发布的数据包，时间窗 2016-10 ~ 2018-11） |
| 预处理 | 5-core 过滤；滑窗截断（max len = 10）；按时序 8:1:1 切分 train/val/test（数据包已提供成品） |
| 指标 | 全目录排名下的 Recall@{5,10}、NDCG@{5,10} |

复现结果与论文 Table 2 的对照见 [`results/reproduction.md`](results/reproduction.md)。

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
├── Makefile                # 常用任务入口（games-sasrec / office-tiger / smoke ...）
├── configs/                # 预留：数据 / 模型 / 实验配置（当前基线用命令行参数）
├── src/
│   ├── data/
│   │   ├── bundle.py       # CSV → DatasetBundle（train/valid/test + catalog）
│   │   └── sid.py          # SidTable：item↔SID、SID 冲突表、合法前缀 Trie
│   ├── models/
│   │   ├── sasrec/
│   │   │   ├── minionerec.py   # MiniOneRec 原版 SASRec（论文基线实现）
│   │   │   └── model.py        # 备用：共享 embedding 的 SASRec 变体
│   │   └── tiger/
│   │       └── model.py        # T5-style encoder-decoder + trie 约束束搜
│   ├── train/
│   │   ├── train_sasrec_mini.py  # SASRec 训练（MiniOneRec 协议，早停 NDCG@20）
│   │   ├── train_sasrec.py       # 备用 SASRec 训练
│   │   ├── train_tiger.py        # TIGER 训练（Adam/Adafactor + warmup/inv-sqrt）
│   │   └── common.py             # 种子、设备、论文指标对照表
│   ├── eval/
│   │   ├── evaluate.py     # 全目录评测（SASRec / TIGER）
│   │   └── metrics.py      # Recall@K / NDCG@K
│   ├── semid/              # 预留：RQ-VAE（SID 已由数据包提供，暂不需要）
│   └── reasoner/           # 预留：SIDReasoner 四阶段
├── scripts/                # 端到端脚本（run_games_sasrec.sh 等）
├── tests/                  # 冒烟测试（tests/test_smoke.py）
├── results/                # 入库：复现指标 vs 论文指标对照
├── data/                   # 不入库：raw / interim / processed
└── runs/                   # 不入库：checkpoint、日志、metrics.json
```

## 基线实现说明

| 模型 | 实现来源 | 关键设计 |
|---|---|---|
| SASRec | 忠实移植 MiniOneRec 的 `sasrec.py` + `SASRecModules_ori.py`（SIDReasoner 基于 MiniOneRec 构建，数据列格式完全一致） | 单层 self-attention + FFN、独立输出头 `s_fc`（非共享 embedding）、右填充（pad=item_num）、BCE（每样本 1 正 1 负）、Adam(lr=1e-3, eps=1e-8, wd=1e-5)、早停 patience=20（valid NDCG@20） |
| TIGER | 依据 TIGER 原论文（Rajput et al., NeurIPS 2023）+ GAMER 的 T5 实现（官方 google-research 代码已下架） | T5-style encoder-decoder（随机初始化）、输入=历史物品 SID token 序列（每物品 3 token）、输出自回归 SID+EOS、CE loss、trie 约束束搜（beam=20，保证合法）、SID 冲突展开为多物品 |

核心模块：

| 模块 | 接口 | 说明 |
|---|---|---|
| `src.data.bundle` | `load_bundle(data_root, category) -> DatasetBundle` | 读取数据包 CSV → train/valid/test 的 (history ids/sids, target id/sid) |
| `src.data.sid` | `SidTable.from_index_json(path)`；`allowed_codes(prefix)` | item↔SID 双向映射、SID→items 冲突表、合法前缀 Trie |
| `src.models.sasrec.minionerec` | `SASRecMiniOneRec(hidden, item_num, state_size, dropout)` | MiniOneRec 原版 SASRec |
| `src.models.tiger.model` | `build_tiger(sid_table, ...)`；`beam_search_items(...)` | T5-style TIGER + 约束束搜 |
| `src.train.train_sasrec_mini` / `train_tiger` | 命令行入口 | 训练 + 逐 epoch valid 早停 + test 指标 + `runs/*/metrics.json` |
| `src.eval.evaluate` / `metrics` | `evaluate_sasrec/evaluate_tiger`；`rank_metrics` | 全目录排名 Recall/NDCG@{5,10,(20)} |

## 环境安装

```bash
cd Reasoning4GenRec-Reproduction
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pandas numpy pyyaml
```

数据：SIDReasoner 发布的数据包（Amazon 2018 5-core，Games/Office/Industrial），
解压后置于 `data/raw/Amazon/`（`train/ valid/ test/ info/ index/` 五个子目录）。

## 快速开始

```bash
# SASRec（MiniOneRec 原版实现）
make games-sasrec      # 或 bash scripts/run_games_sasrec.sh
make office-sasrec

# TIGER（4 层 d192，trie 约束束搜）
make games-tiger
make office-tiger

# 冒烟测试
make smoke
```

结果（checkpoint + metrics.json）写入 `runs/`。

## 与论文对齐的关键设置

| 项 | 取值 |
|---|---|
| 底座模型 | Qwen3-1.7B，全参数微调 |
| SID 构造 | RQ-VAE 残差量化；文本编码器与码本配置以官方代码为准 |
| SASRec（基线） | MiniOneRec 实现；hidden=32、dropout=0.3、batch=1024、Adam(lr=1e-3, wd=1e-5)、BCE |
| TIGER（基线） | T5-style（4 层 d192、4 头、d_ff=768）、batch=256、Adam(lr=1e-3, warmup=1000, inverse-sqrt)、beam=20 |
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

复现结果与逐项对照见 [`results/reproduction.md`](results/reproduction.md)。

## 测试

```bash
make smoke     # 等价于 .venv/bin/python tests/test_smoke.py
```

## 路线图

- [x] Phase 0：仓库骨架、Makefile、端到端脚本
- [x] Phase 1：数据加载（`src.data`，Games / Office 就绪，切分无泄漏校验通过）
- [x] Phase 2：SASRec + `src.eval`（Games / Office 复现完成，偏差 ≤3%）
- [x] Phase 3：TIGER（Games / Office 复现完成，偏差 ≤6%）
- [ ] Phase 4：SIDReasoner 语料合成 → 对齐训练 → 激活训练
- [ ] Phase 5：GRPO（verl）
- [ ] Phase 6：结果汇总、与论文指标对齐分析（见 results/reproduction.md）

## 引用

```bibtex
@inproceedings{he2026sidreasoner,
  title     = {Reasoning over Semantic IDs Enhances Generative Recommendation},
  author    = {He, Yingzhi and Sun, Yan and Tan, Junfei and Chen, Yuxin and Kong, Xiaoyu and Shen, Chunxu and Wang, Xiang and Zhang, An and Chua, Tat-Seng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026}
}
```
