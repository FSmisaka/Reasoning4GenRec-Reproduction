"""
SIDReasoner 评估结果后处理

split:  把测试 CSV 均分给多张卡(迁移自 split.py)
merge:  合并多卡拆分评估产生的分片 json(迁移自 merge.py)
calc:   计算 HR@K / NDCG@K(迁移自 calc.py, K 取 1/3/5/10/20/50
        中不超过束宽的值; 匹配方式为 SID 串全等, 同名多物品共享排名)

用法:
  DATASET=games .venv/bin/python -m src.eval.sidreasoner_metrics \
      calc --path <result.json>
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def split(input_path, output_path, cuda_list, num_samples=-1):
    if isinstance(cuda_list, int):
        cuda_list = [cuda_list]
    df = pd.read_csv(input_path)
    if num_samples is not None and int(num_samples) > -1:
        num_samples = int(num_samples)
        df = df.head(num_samples) if num_samples > 0 else df.iloc[0:0]
    os.makedirs(output_path, exist_ok=True)
    df_len = len(df)
    cuda_list = list(cuda_list)
    for i in range(len(cuda_list)):
        start = i * df_len // len(cuda_list)
        end = (i + 1) * df_len // len(cuda_list)
        df[start:end].to_csv(
            f"{output_path}/{cuda_list[i]}.csv", index=True
        )


def merge(input_path, output_path, cuda_list):
    if isinstance(cuda_list, int):
        cuda_list = [cuda_list]
    cuda_list = list(cuda_list)
    result = []
    for gpu in cuda_list:
        shard = os.path.join(input_path, f"{gpu}.json")
        with open(shard, "r") as f:
            result.extend(json.load(f))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[merge] {len(result)} samples -> {output_path}")


def calc(path, item_path):
    if isinstance(path, str):
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC = 0

    with open(f"{item_path}.txt", "r") as f:
        items = f.readlines()
    item_names = [_.split("\t")[0].strip() for _ in items]
    item_ids = list(range(len(item_names)))
    item_dict = {}
    for i in range(len(item_names)):
        item_dict.setdefault(item_names[i], []).append(item_ids[i])

    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        with open(p, "r") as f:
            test_data = json.load(f)

        text = [
            [_.strip('"\n').strip() for _ in sample["predict"]]
            for sample in test_data
        ]

        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                ALLNDCG = np.zeros(len(topk_list))
                ALLHR = np.zeros(len(topk_list))
            if type(test_data[index]["output"]) == list:
                target_item = (
                    test_data[index]["output"][0].strip('"').strip(" ")
                )
            else:
                target_item = test_data[index]["output"].strip(' \n"')
            minID = 1000000
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    CC += 1
                if sample[i] == target_item:
                    minID = i
                    break
            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(valid_topk)
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")
        print(f"invalid predictions: {CC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["split", "merge", "calc"])
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--cuda_list", default="0")
    parser.add_argument("--path", default=None)
    parser.add_argument("--item_path", default=None)
    args = parser.parse_args()

    if args.mode == "split":
        split(
            args.input_path,
            args.output_path,
            [int(x) for x in args.cuda_list.split(",")],
        )
    elif args.mode == "merge":
        merge(
            args.input_path,
            args.output_path,
            [int(x) for x in args.cuda_list.split(",")],
        )
    else:
        from config import load_config
        from src.train.common import resolve_category

        cfg = load_config("sidreasoner")
        category = resolve_category(cfg["data"])
        if args.path is None:
            raise SystemExit("--path is required for calc")
        item_path = args.item_path or os.path.join(
            cfg["data"]["root"],
            "info",
            f"{category}_5_2016-10-2018-11.txt",
        )
        calc(args.path, item_path)
