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

# SASRec(MiniOneRec 原版实现)
make games-sasrec-mini
make office-sasrec-mini
make industrial-sasrec-mini

# TIGER(4 层 d128,trie 约束束搜)
make games-tiger
make office-tiger
make industrial-tiger

# 等价的手动运行方式
DATASET=games .venv/bin/python -m src.train.tiger
# 或 scripts/games/tiger.sh

# Caser(卷积序列推荐,WSDM'18)
make games-caser
make office-caser
make industrial-caser

# 冒烟测试
make smoke
```