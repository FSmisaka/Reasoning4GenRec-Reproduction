"""
SIDReasoner Stage 3(RL)训练数据构建

迁移自 SIDReasoner 官方仓库的 scripts/create_reasoning_rl_data.py。
从本地 CSV + index 构建 verl GRPO 所需的 parquet
(prompt 为 chat 消息列表, ground_truth 为目标 SID)。

注意: 本地 data/raw/Amazon/rec_reasoning_verl/<Category>/ 下
官方发布的 parquet 由更长历史(均值 ~15.6, 上限 30)的数据版本构建,
与当前截断为 10 的 CSV 不完全一致(上游已知差异); 重新生成的
parquet 格式相同但历史更短, 训练请默认使用官方 parquet
(scripts/sidreasoner/rl.sh 的默认行为)。

启动: DATASET=games .venv/bin/python -m src.data.sidreasoner_rl_data
"""

import argparse
import json
import os
import random
from tqdm import tqdm

import pandas as pd
from torch.utils.data import Dataset

from config import load_config
from src.train.common import resolve_category


class Reasoning_RL_Dataset(Dataset):
    """
    迁移自 create_reasoning_rl_data.py: RL 用的推理推荐数据集,
    学习目标是给定 SID 历史先生成推理再输出目标 SID。
    """

    def __init__(
        self,
        data_file,
        item_file,
        index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
    ):
        random.seed(seed)
        self.data = pd.read_csv(data_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        with open(item_file, "r") as f:
            self.item_feat = json.load(f)
        with open(index_file, "r") as f:
            self.indices = json.load(f)

        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        self.sid2title = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]["title"]
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
        self.get_inputs()

    def __len__(self):
        return len(self.data)

    def generate_prompt_title(self, history):
        return (
            f"The user has sequentially interacted with items {history}. "
            "Can you recommend the next item for him? Let's think step by "
            "step before making recommendation. Directly output the item "
            "SID after thinking."
        )

    def get_history(self, row):
        history_item_sid = eval(row["history_item_sid"])
        history_str = ", ".join(history_item_sid)
        target_sid = row["item_sid"]
        target_title = self.sid2title.get(target_sid, target_sid)
        last_history_sid = (
            history_item_sid[-1] if history_item_sid else None
        )
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": target_sid == last_history_sid,
        }

    def pre(self, idx):
        instruction = (
            "Below is an instruction that describes a task, paired with an "
            "input that provides further context. Write a response that "
            "appropriately completes the request.\nCan you recommend the "
            "next item for the user based on their interaction history?\n"
        )
        history_data = self.get_history(self.data.iloc[idx])
        if self.dedup and history_data["dedup"]:
            return None
        prompt = self.generate_prompt_title(history_data["history_str"])
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
        return {
            "input": messages,
            "target": history_data["target_sid"],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:
                inputs.append(result)
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


def convert_to_verl_format(ds, split, out_path, data_source):
    rows = []
    for idx in range(len(ds)):
        example = ds[idx]
        question_raw = example["input"]
        answer_raw = example["target"]
        rows.append(
            {
                "data_source": data_source,
                "prompt": question_raw,
                "ability": "Recommendation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")


def main(cfg=None):
    cfg = cfg or load_config("sidreasoner")
    data_cfg = cfg["data"]
    category = resolve_category(data_cfg)
    root = data_cfg["root"]
    fname = f"{category}_5_2016-10-2018-11.csv"

    rl_cfg = cfg["rl"]
    out_dir = rl_cfg.get("data_dir") or os.path.join(
        root, "rec_reasoning_verl", category
    )
    train_dataset = Reasoning_RL_Dataset(
        data_file=os.path.join(root, "train", fname),
        item_file=os.path.join(root, "index", f"{category}.item.json"),
        index_file=os.path.join(root, "index", f"{category}.index.json"),
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category=category,
        dedup=False,
    )
    eval_dataset = Reasoning_RL_Dataset(
        data_file=os.path.join(root, "test", fname),
        item_file=os.path.join(root, "index", f"{category}.item.json"),
        index_file=os.path.join(root, "index", f"{category}.index.json"),
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=True,
        seed=0,
        category=category,
        dedup=False,
    )
    convert_to_verl_format(
        train_dataset,
        split="train",
        out_path=os.path.join(out_dir, "train.parquet"),
        data_source=f"rec/{category}",
    )
    convert_to_verl_format(
        eval_dataset,
        split="test",
        out_path=os.path.join(out_dir, "test.parquet"),
        data_source=f"rec/{category}",
    )


if __name__ == "__main__":
    main()
