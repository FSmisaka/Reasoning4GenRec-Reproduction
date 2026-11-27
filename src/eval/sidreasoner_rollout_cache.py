"""Cache Stage 2 training-style rollouts on the Stage 3 test parquet."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
from tempfile import NamedTemporaryFile

from config import load_config
from src.data.sid_switch import sid_suffix
from src.data.sidreasoner_user_embed import parse_history_sids_from_prompt
from src.reward.sid_rewards.base import BEAM_SEP
from src.train.common import resolve_category

CACHE_VERSION = 1


def default_cache_path(cfg, category):
    return Path("data/sidreasoner_rollouts") / (
        f"{category}{sid_suffix(cfg['data'])}_stage2_test.jsonl"
    )


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_catalog(path):
    titles = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            sid, title, *_ = line.rstrip("\n").split("\t")
            titles.setdefault(sid, title)
    return titles


def load_queries(parquet, catalog, tokenizer, rl, seed, limit):
    import pandas as pd

    frame = pd.read_parquet(
        parquet, columns=["prompt", "extra_info", "reward_model", "data_source"]
    )
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    queries = []
    for pos in indices:
        row = frame.iloc[pos]
        extra = row["extra_info"]
        messages = list(row["prompt"])
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > rl["max_prompt_length"]:
            if rl["filter_overlong_prompts"]:
                continue
            raise ValueError(f"测试 query {extra['index']} 超过 max_prompt_length。")
        queries.append({
            "extra_info": {"split": str(extra["split"]), "index": int(extra["index"])},
            "data_source": str(row["data_source"]),
            "ground_truth": str(row["reward_model"]["ground_truth"]),
            "history": [
                {"sid": sid, "title": catalog[sid]}
                for sid in parse_history_sids_from_prompt(messages)
            ],
            "prompt_token_ids": token_ids,
        })
        if limit is not None and len(queries) == limit:
            break
    return queries


def generate_batch(engine, tokenizer, prompts, sampling, rollout_n, beam_size, sid_length,
                   truncate_for_beam, beam_search):
    inputs = [{"prompt_token_ids": prompt} for prompt in prompts for _ in range(rollout_n)]
    outputs = engine.generate(inputs, sampling, use_tqdm=False)
    if len(outputs) != len(inputs) or any(len(out.outputs) != 1 for out in outputs):
        raise RuntimeError("vLLM 返回的 rollout 数与请求不一致。")

    response_ids = [list(out.outputs[0].token_ids) for out in outputs]
    think_marker = tokenizer.encode("</think>", add_special_tokens=False)
    newline_ids = {
        token for newline in ("\n", "\n\n")
        for token in tokenizer.encode(newline, add_special_tokens=False)
    }
    beam_prompts = [
        request["prompt_token_ids"] + truncate_for_beam(ids, think_marker, newline_ids)[0]
        for request, ids in zip(inputs, response_ids, strict=True)
    ]
    beams = beam_search(
        engine, tokenizer, beam_prompts, sampling, beam_width=beam_size, depth=sid_length
    )
    rollouts = [
        {
            "solution_str": tokenizer.decode(ids, skip_special_tokens=True),
            "beam_search_results": [piece.strip() for piece in beam.split(BEAM_SEP)],
        }
        for ids, beam in zip(response_ids, beams, strict=True)
    ]
    return [
        rollouts[start : start + rollout_n]
        for start in range(0, len(rollouts), rollout_n)
    ]


def main(argv=None, cfg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-queries", type=int, help="只缓存指定数量的 query，用于小规模试跑")
    parser.add_argument("--batch-size", type=int, help="离线重评分时的 query batch 大小")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--gpu-mem", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(value is not None and value < 1 for value in (
        args.n_queries, args.batch_size, args.generation_batch_size,
    )):
        parser.error("query 数和 batch 大小必须为正数。")

    cfg = cfg or load_config("sidreasoner")
    rl = cfg["rl"]
    category = resolve_category(cfg["data"])
    suffix = sid_suffix(cfg["data"])
    checkpoint = Path(args.model or (
        f"runs/sidreasoner/{category}_stage2_activation{suffix}/final_checkpoint"
    ))
    if not checkpoint.is_dir():
        parser.error(f"找不到 Stage 2 checkpoint: {checkpoint}")

    root = Path(cfg["data"]["root"])
    parquet = Path(rl.get("data_dir") or root / "rec_reasoning_verl" / category) / "test.parquet"
    catalog = read_catalog(root / "info" / f"{category}_5_2016-10-2018-11.txt")
    batch_size = args.batch_size or rl["train_batch_size"]
    rollout_n = 16
    beam_size, sid_length = rl["sid_beam_size"], rl["sid_length"]
    if beam_size < 2:
        parser.error("缓存需要 sid_beam_size >= 2，用于 Rollout-Beam 和排名统计。")

    verl_home = Path(os.environ.get("VERL_HOME", rl["verl_home"])).resolve()
    sys.path.insert(0, str(verl_home))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
        truncate_after_think_for_beam,
        vllm_beam_search_concat,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    queries = load_queries(parquet, catalog, tokenizer, rl, args.seed, args.n_queries)
    if not queries:
        raise ValueError(f"{parquet} 没有可生成的测试 query。")

    engine = LLM(
        model=str(checkpoint),
        tensor_parallel_size=rl["tensor_model_parallel_size"],
        dtype="bfloat16",
        gpu_memory_utilization=(
            args.gpu_mem if args.gpu_mem is not None else rl["gpu_memory_utilization"]
        ),
        max_model_len=rl["max_prompt_length"] + rl["max_response_length"] + sid_length,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sampling_args = {
        "n": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "max_tokens": rl["max_response_length"], "detokenize": False,
    }
    sampling = SamplingParams(**sampling_args)
    out = args.out or default_cache_path(cfg, category)
    meta = {
        "cache_version": CACHE_VERSION, "category": category, "sid_suffix": suffix,
        "model": str(checkpoint), "parquet": str(parquet), "n_queries": len(queries),
        "batch_size": batch_size, "rollout_n": rollout_n, "seed": args.seed,
        "beam_size": beam_size, "sid_length": sid_length, "sampling": sampling_args,
    }
    with atomic_output(out) as handle:
        handle.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            for pos in range(0, len(batch), args.generation_batch_size):
                chunk = batch[pos : pos + args.generation_batch_size]
                rollouts = generate_batch(
                    engine, tokenizer, [q.pop("prompt_token_ids") for q in chunk],
                    sampling, rollout_n, beam_size, sid_length,
                    truncate_after_think_for_beam, vllm_beam_search_concat,
                )
                for query, group in zip(chunk, rollouts, strict=True):
                    query["rollouts"] = group
            handle.write(json.dumps({"queries": batch}, ensure_ascii=False) + "\n")
            print(f"[rollout-cache] {start + len(batch)}/{len(queries)} queries", flush=True)
    print(f"[rollout-cache] {out}")
    return out


if __name__ == "__main__":
    main()
