"""
SIDReasoner 评估(思考模式)

迁移自 SIDReasoner 官方仓库的 evaluate_Qwen3_think.py。
先用 vLLM 生成思维链(截断到 </think>), 再用 trie 约束束搜索
生成 SID。需要安装 vllm。

CoT 解码温度可配置(OneRec-Think 官方 Stage 评测用
think_temperature=1.5 / think_top_p=0.95 采样生成思维链):
- think_temperature > 0: 采样解码, 按 evaluation.eval_seeds 逐 seed
  评测(vLLM per-request seed 保证可复现), 各 seed 指标在进程内计算,
  最终汇总输出 <C>_think_<model>_summary.json(各指标 mean/std/95% CI,
  含逐 seed 明细)。默认不再写单 seed 明细文件, 需要 cot 字段做
  多样性分析等用途时设 evaluation.save_per_seed_results: true。
- think_temperature == 0: 原贪心解码行为不变(单 seed, 文件名无 _seed)。

启动: DATASET=games .venv/bin/python -m src.eval.sidreasoner_eval_think
"""

import copy
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from src.data.sidreasoner_data import Reasoning_Eval_Dataset
from src.eval.sidreasoner_metrics import (
    aggregate_metrics,
    compute_metrics,
    print_aggregate,
)
from src.models.sidreasoner import category_phrase
from src.train.common import resolve_category

# 含 opencv 的 venv 里, vLLM v1 引擎 spawn 的子进程 re-import 入口模块时会触发
# cv2/typing 遮蔽标准库 typing 的崩溃(Engine core initialization failed)。
# 改为进程内运行引擎彻底绕开; 需要多进程时可用环境变量
# VLLM_ENABLE_V1_MULTIPROCESSING=1 显式覆盖。
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


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
    # info 目录为 SID\ttitle\titem_id 的 .txt(与 train/test 的 .csv 不同)
    info_file = os.path.join(root, "info", f"{category}_5_2016-10-2018-11.txt")
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

    # ---- 解码设置(采样 vs 贪心) ----
    think_temperature = float(eval_cfg.get("think_temperature", 0.0))
    think_top_p = float(eval_cfg.get("think_top_p", 0.95))
    ci_level = float(eval_cfg.get("ci_level", 0.95))
    if think_temperature > 0:
        seeds = list(eval_cfg.get("eval_seeds", [42, 43, 44, 45, 46]))
        multi_seed = True
    else:
        # 贪心解码确定, 退化为单 seed 旧行为
        seeds = [eval_cfg.get("seed", 42)]
        multi_seed = False
    # 多 seed 模式默认只写 summary(已含逐 seed 指标);
    # 需要单 seed 明细(如多样性分析导入 cot)时置 true
    save_per_seed = multi_seed and bool(
        eval_cfg.get("save_per_seed_results", False)
    )

    if result_json_data is None:
        from src.data.sid_switch import sid_suffix
        from src.eval.result_routing import build_result_path

        # 两级目录路由: results/sidreasoner/<Dataset>/think_<variant>[_temp]/<step>.json
        result_json_data = build_result_path(
            category=category,
            kind="think",
            model=base_model,
            sid_sfx=sid_suffix(data_cfg),
            temperature=think_temperature,
        )
    result_prefix, result_ext = os.path.splitext(result_json_data)

    os.environ.setdefault("FLASH_ATTENTION", "1")
    phrase = category_phrase(category)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
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
        seed=seeds[0],
    )
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    # 每个 seed 复用的干净副本(beam search 前会被 CoT 覆写)
    raw_encodings = copy.deepcopy(encodings)

    model.config.pad_token_id = model.config.eos_token_id = (
        tokenizer.eos_token_id
    )
    model.config.bos_token_id = tokenizer.bos_token_id

    from vllm import SamplingParams

    def make_sampling_params(seed):
        # OneRec-Think 官方 CoT 评测: temperature=1.5, top_p=0.95(采样);
        # temperature=0 时 vLLM 即贪心解码, 与原实现一致
        kwargs = dict(
            max_tokens=eval_cfg.get("think_max_new_tokens", 256),
            temperature=think_temperature,
            skip_special_tokens=False,
        )
        if think_temperature > 0:
            kwargs.update(top_p=think_top_p, seed=seed)
        return SamplingParams(**kwargs)

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

    # ---- 第 1 阶段: vLLM 采样生成各 seed 的 CoT ----
    cots_by_seed = {}
    for seed in seeds:
        set_seed(seed)
        with torch.no_grad():
            outputs = llm.generate(prompts, make_sampling_params(seed))
        raw_cots = [cot.outputs[0].text.strip() for cot in outputs]
        close_found = ["</think>" in c for c in raw_cots]
        cots = [c[: c.find("</think>")].strip() for c in raw_cots]
        prompt_cot = [
            prompt + cot.outputs[0].text.strip()
            for prompt, cot in zip(prompts, outputs)
        ]
        prompt_cot = [
            text[: text.find("</think>")].strip() + "\n</think>\n\n"
            for text in prompt_cot
        ]
        cots_by_seed[seed] = {
            "raw_cots": raw_cots,
            "close_found": close_found,
            "cots": cots,
            "prompt_cot": prompt_cot,
        }
        print(
            f"[eval] seed={seed}: CoT 生成完成"
            f"({sum(close_found)}/{len(close_found)} 含 </think>)"
        )
    del llm

    # ---- 第 2 阶段: 逐 seed 做 trie 约束 beam search + 指标 ----
    from tqdm import tqdm

    per_seed_metrics = {}
    for seed in seeds:
        bundle = cots_by_seed[seed]
        encodings = copy.deepcopy(raw_encodings)
        for i in range(len(encodings)):
            reasoning_output = tokenizer(
                bundle["prompt_cot"][i],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2560,
            )
            encodings[i]["input_ids"] = reasoning_output["input_ids"][
                0
            ].tolist()
            encodings[i]["attention_mask"] = reasoning_output[
                "attention_mask"
            ][0].tolist()

        blocks = [
            encodings[i * batch_size : (i + 1) * batch_size]
            for i in range(
                (len(encodings) + batch_size - 1) // batch_size
            )
        ]
        outputs = []
        for enc in tqdm(
            blocks,
            desc=f"Generating reasoning trajectories and items (seed={seed})",
        ):
            outputs = outputs + evaluate(enc)

        test_data = val_dataset.get_all()
        for i, test in enumerate(test_data):
            test["prompt_cot"] = bundle["prompt_cot"][i]
            test["predict"] = outputs[i]
            test["cot"] = bundle["cots"][i]
            test["cot_len"] = len(
                tokenizer(
                    bundle["cots"][i], add_special_tokens=False
                )["input_ids"]
            )
            test["close_found"] = bundle["close_found"][i]
        for i in range(len(test_data)):
            if "dedup" in test_data[i]:
                test_data[i].pop("dedup")

        if save_per_seed:
            seed_json = f"{result_prefix}_seed{seed}{result_ext}"
            os.makedirs(os.path.dirname(seed_json), exist_ok=True)
            with open(seed_json, "w") as f:
                json.dump(test_data, f, indent=4)
            print(f"[eval] results saved to {seed_json}")
        elif not multi_seed:
            os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
            with open(result_json_data, "w") as f:
                json.dump(test_data, f, indent=4)
            print(f"[eval] results saved to {result_json_data}")

        per_seed_metrics[seed] = compute_metrics(test_data, semantic_ids)
        shown = {
            k: round(v, 6)
            for k, v in per_seed_metrics[seed].items()
            if k.startswith(("NDCG@", "HR@"))
        }
        print(f"[eval] seed={seed} metrics: {shown}")

    # ---- 第 3 阶段: 跨 seed 聚合(mean / std / 95% CI) ----
    if multi_seed:
        agg = aggregate_metrics(per_seed_metrics, ci_level=ci_level)
        summary = {
            "model": base_model,
            "category": category,
            "decoding": {
                "think_temperature": think_temperature,
                "think_top_p": think_top_p,
                "max_tokens": eval_cfg.get("think_max_new_tokens", 256),
                "num_beams": num_beams,
            },
            "seeds": seeds,
            "ci_level": ci_level,
            "per_seed": per_seed_metrics,
            "aggregate": agg,
        }
        summary_json = f"{result_prefix}_summary{result_ext}"
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[eval] multi-seed summary saved to {summary_json}")
        print_aggregate(agg, title=f"{category} think eval")


if __name__ == "__main__":
    main()
