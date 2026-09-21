# 环境

## conda

```bash
source /data/jiangchumeng/miniconda3/etc/profile.d/conda.sh
conda create -p /data/jiangchumeng/conda_envs/r4gr python=3.11
conda activate /data/jiangchumeng/conda_envs/r4gr
```

## venv

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pandas numpy pyyaml
```

# quick start

```bash
# SASRec（MiniOneRec 原版实现）
make games-sasrec
make office-sasrec

# TIGER（4 层 d192，trie 约束束搜）
make games-tiger
make office-tiger

# 冒烟测试
make smoke
```