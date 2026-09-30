"""
SIDReasoner 评估(非思考模式)

迁移自 SIDReasoner 官方仓库的 evaluate_Qwen3.py。
加载 checkpoint, 在测试集上做 trie 约束的束搜索
(以 "\n</think>\n\n" 之后的前缀哈希约束 SID 生成),
输出 top-N 预测的 SID 串列表(json)。

启动: DATASET=games .venv/bin/python -m src.eval.sidreasoner_eval
(多卡并行拆分见 scripts/sidreasoner/eval.sh)
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from src.data.sidreasoner_data import EvalSidDataset
from src.models.sidreasoner import category_phrase
from src.train.common import resolve_category

P = 998244353
MOD = int(1e9 + 9)


def get_hash(x):
    x = [str(_) for _ in x]
    return "-".join(x)


def set_seed(seed):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_last_sublist(lst, sub):
    if not sub:
        return None
    n, m = len(lst), len(sub)
    for start in range(n - m, -1, -1):
        if lst[start : start + m] == sub:
            return start
    return None


def main(cfg=None, base_model=None, test_data_path=None, result_json_data=None):
    cfg = cfg or load_config("sidreasoner")
    data_cfg, eval_cfg = cfg["data"], cfg["evaluation"]
    category = resolve_category(data_cfg)

    root = data_cfg["root"]
    fname = f"{category}_5_2016-10-2018-11.csv"
    info_file = os.path.join(root, "info", fname)
    if base_model is None:
        stage = eval_cfg.get("eval_model", "stage1")
        if stage == "stage2":
            base_model = os.path.join(
                "runs", "sidreasoner", f"{category}_stage2_activation",
                "final_checkpoint",
            )
        elif stage == "stage1":
            base_model = os.path.join(
                "runs", "sidreasoner", f"{category}_stage1_sft",
                "final_checkpoint",
            )
        else:
            base_model = stage
    if test_data_path is None:
        test_data_path = os.path.join(root, "test", fname)
    if result_json_data is None:
        os.makedirs(os.path.join("results", "sidreasoner"), exist_ok=True)
        result_json_data = os.path.join(
            "results", "sidreasoner",
            f"{category}_{os.path.basename(os.path.normpath(base_model))}.json",
        )

    seed = eval_cfg.get("seed", 42)
    set_seed(seed)
    phrase = category_phrase(category)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    prefix_prompt = "\n</think>\n\n"
    prefix_index = 3
    with open(info_file, "r") as f:
        info = f.readlines()
        semantic_ids = [line.split("\t")[0].strip() for line in info]
        info_semantic = [f"""{prefix_prompt}{_}\n""" for _ in semantic_ids]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])

    sep_ids = tokenizer(prefix_prompt, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        input_ids = input_ids.tolist()
        pos = find_last_sublist(input_ids, sep_ids)
        if pos is None:
            raise Exception(
                "Error: Prefix prompt not found in input IDs - "
                f"{tokenizer.decode(input_ids)}."
            )
        pos_after_sep = pos + len(sep_ids)
        generated_after_sep = input_ids[pos_after_sep:]
        current_pos = len(generated_after_sep)
        if current_pos == 0:
            hash_number = get_hash(sep_ids)
            if hash_number in hash_dict:
                return hash_dict[hash_number]
            return [eos_id]
        hash_number = get_hash(generated_after_sep)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return [eos_id]

    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = eval_cfg.get("padding_side", "left")

    val_dataset = EvalSidDataset(
        train_file=test_data_path,
        tokenizer=tokenizer,
        max_len=2560,
        category=phrase,
        test=True,
        seed=seed,
    )
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = (
        tokenizer.eos_token_id
    )
    model.config.bos_token_id = tokenizer.bos_token_id
    model = model.to(device)

    num_beams = eval_cfg["num_beams"]
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)
    length_penalty = eval_cfg.get("length_penalty", 0.0)
    batch_size = eval_cfg.get("batch_size", 8)

    def evaluate(encodings, **kwargs):
        maxLen = max([len(_["input_ids"]) for _ in encodings])
        padding_encodings = {"input_ids": []}
        attention_mask = []
        for _ in encodings:
            L = len(_["input_ids"])
            if tokenizer.padding_side == "left":
                padding_encodings["input_ids"].append(
                    [tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"]
                )
                attention_mask.append([0] * (maxLen - L) + [1] * L)
            else:
                padding_encodings["input_ids"].append(
                    _["input_ids"] + [tokenizer.pad_token_id] * (maxLen - L)
                )
                attention_mask.append([1] * L + [0] * (maxLen - L))

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=torch.tensor(
                    padding_encodings["input_ids"]
                ).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                length_penalty=length_penalty,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        batched_completions = generation_output.sequences[:, maxLen:]
        if base_model.lower().find("llama") > -1:
            output_raw = tokenizer.batch_decode(
                batched_completions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            output_raw = tokenizer.batch_decode(
                batched_completions, skip_special_tokens=True
            )
        output = [_.split("Response:\n")[-1].strip() for _ in output_raw]
        real_outputs = [
            output[i * num_beams : (i + 1) * num_beams]
            for i in range(len(output) // num_beams)
        ]
        return real_outputs

    from tqdm import tqdm

    outputs = []
    blocks = [
        encodings[i * batch_size : (i + 1) * batch_size]
        for i in range((len(encodings) + batch_size - 1) // batch_size)
    ]
    for enc in tqdm(blocks):
        outputs = outputs + evaluate(enc)

    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
    for i in range(len(test_data)):
        if "dedup" in test_data[i]:
            test_data[i].pop("dedup")

    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    with open(result_json_data, "w") as f:
        json.dump(test_data, f, indent=4)
    print(f"[eval] results saved to {result_json_data}")


if __name__ == "__main__":
    main()
