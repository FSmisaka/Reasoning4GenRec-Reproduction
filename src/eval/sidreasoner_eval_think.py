"""
SIDReasoner 评估(思考模式)

迁移自 SIDReasoner 官方仓库的 evaluate_Qwen3_think.py。
先用 vLLM 生成思维链(截断到 </think>), 再用 trie 约束束搜索
生成 SID。需要安装 vllm。

启动: DATASET=games .venv/bin/python -m src.eval.sidreasoner_eval_think
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from src.data.sidreasoner_data import Reasoning_Eval_Dataset
from src.models.sidreasoner import category_phrase
from src.train.common import resolve_category


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


def main(
    cfg=None,
    base_model=None,
    test_data_path=None,
    result_json_data=None,
):
    cfg = cfg or load_config("sidreasoner")
    data_cfg, eval_cfg = cfg["data"], cfg["evaluation"]
    category = resolve_category(data_cfg)

    root = data_cfg["root"]
    fname = f"{category}_5_2016-10-2018-11.csv"
    info_file = os.path.join(root, "info", fname)
    item_file = os.path.join(root, "index", f"{category}.item.json")
    index_file = os.path.join(root, "index", f"{category}.index.json")
    if test_data_path is None:
        test_data_path = os.path.join(root, "test", fname)
    if base_model is None:
        base_model = os.environ.get("MODEL") or eval_cfg["think_eval_model"]
    if not base_model:
        raise SystemExit(
            "未指定待评模型: 用 MODEL=<checkpoint路径> 环境变量"
            "(如 make games-sidreasoner-think MODEL=...)或 config 的 "
            "evaluation.think_eval_model 指定"
        )
    if result_json_data is None:
        os.makedirs(os.path.join("results", "sidreasoner"), exist_ok=True)
        result_json_data = os.path.join(
            "results", "sidreasoner",
            f"{category}_think_{os.path.basename(os.path.normpath(base_model))}.json",
        )

    os.environ.setdefault("FLASH_ATTENTION", "1")
    seed = eval_cfg.get("seed", 42)
    set_seed(seed)
    phrase = category_phrase(category)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = True
    model.eval()
    from vllm import LLM

    llm = LLM(
        model=base_model,
        max_model_len=2048,
        max_num_seqs=32,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
    )

    prefix_prompt = "</think>\n\n"
    prefix_index = 2
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

    val_dataset = Reasoning_Eval_Dataset(
        data_file=test_data_path,
        item_file=item_file,
        index_file=index_file,
        sample=-1,
        tokenizer=tokenizer,
        max_len=2048,
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

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=eval_cfg.get("max_new_tokens", 1024),
        temperature=0.0,
        skip_special_tokens=False,
    )

    num_beams = eval_cfg["num_beams"]
    max_new_tokens = eval_cfg.get("max_new_tokens", 1024)
    length_penalty = eval_cfg.get("length_penalty", 0.0)
    batch_size = eval_cfg.get("think_batch_size", 4)

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
            else:
                padding_encodings["input_ids"].append(
                    _["input_ids"] + [tokenizer.pad_token_id] * (maxLen - L)
                )
            attention_mask.append([0] * (maxLen - L) + [1] * L)

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
        output = tokenizer.batch_decode(
            batched_completions, skip_special_tokens=True
        )
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [
            output[i * num_beams : (i + 1) * num_beams]
            for i in range(len(output) // num_beams)
        ]
        return real_outputs

    model = model.to(device)

    prompts = tokenizer.batch_decode(
        [_["input_ids"] for _ in encodings], skip_special_tokens=False
    )
    with torch.no_grad():
        outputs = llm.generate(prompts, sampling_params)
    cots = [cot.outputs[0].text.strip() for cot in outputs]
    cots = [cot[: cot.find("</think>")].strip() for cot in cots]
    prompt_cot = [
        prompt + cot.outputs[0].text.strip()
        for prompt, cot in zip(prompts, outputs)
    ]
    prompt_cot = [
        text[: text.find("</think>")].strip() + "\n</think>\n\n"
        for text in prompt_cot
    ]
    for i in range(len(encodings)):
        reasoning_output = tokenizer(
            prompt_cot[i],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2560,
        )
        encodings[i]["input_ids"] = reasoning_output["input_ids"][0].tolist()
        encodings[i]["attention_mask"] = reasoning_output["attention_mask"][
            0
        ].tolist()
    del llm

    blocks = [
        encodings[i * batch_size : (i + 1) * batch_size]
        for i in range((len(encodings) + batch_size - 1) // batch_size)
    ]

    from tqdm import tqdm

    outputs = []
    for enc in tqdm(
        blocks, desc="Generating reasoning trajectories and items..."
    ):
        outputs = outputs + evaluate(enc)

    for i, test in enumerate(test_data):
        test["prompt_cot"] = prompt_cot[i]
        test["predict"] = outputs[i]
        test["cot"] = cots[i]
    for i in range(len(test_data)):
        if "dedup" in test_data[i]:
            test_data[i].pop("dedup")

    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    with open(result_json_data, "w") as f:
        json.dump(test_data, f, indent=4)
    print(f"[eval] results saved to {result_json_data}")


if __name__ == "__main__":
    main()
