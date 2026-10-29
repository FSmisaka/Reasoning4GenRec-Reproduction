"""
思维链多样性分析 · 阶段一: 采集三类语料

采集 SIDReasoner 多样性分析所需的思维链文本, 每条思维链一个文档,
持久化为 JSONL(每行一个 user 样本), 供 diversity_report.py 统计。

三类语料(对应实验关注的 3 个对象):
  1. teacher  : Stage 2 训练数据中的教师模型思维链
                (index/<Category>.integrated_narrative.csv 的 reasoning_path 列,
                与 ReasoningActivationDataset 相同的过滤/清洗规则)
  2. stage2   : Stage 2 训练完成后, 模型在测试集上输出的思维链
  3. stage3   : Stage 3 训练完成后, 模型在测试集上输出的思维链

stage2 / stage3 有两种获取方式:
  a) 本地生成(--model 名字=checkpoint路径): 复刻 sidreasoner_eval_think.py
     的第一阶段——用 Reasoning_Eval_Dataset(test=True) 构造与思考评测
     完全一致的 prompt, 贪心解码生成到 </think> 为止(MPS/CUDA/CPU 均可,
     不需要 vllm; 只生成思维链, 不做约束束搜索, 因此无法得到推荐指标,
     但多样性分析只需要思维链本身)。同一 --num/--seed 下各模型采样
     的测试用户完全一致, 保证跨模型可比。
  b) 导入(--import 名字=think结果json): 服务器上 make <ds>-sidreasoner-think
     产出的 results/sidreasoner/*_think_*.json 已含 cot 字段, 直接复用。

输出位置(永久化): results/diversity/<Category>/corpora/<名字>.jsonl
  每行: {"user_id", "history_str", "target_sid", "cot", "close_found", ...}
  采集参数记录在 results/diversity/<Category>/manifest.json。

用法(DATASET 环境变量选数据集, 同其他模块):
  DATASET=games .venv/bin/python -m src.analysis.collect_diversity_corpora \
      --teacher \
      --model stage2=runs/sidreasoner/Video_Games_stage2_activation/final_checkpoint \
      --import stage3=results/sidreasoner/Video_Games_think_actor_merged.json \
      --num 300 --seed 42
不带任何来源参数时, 自动采集 teacher + 本地默认路径的 stage2(若存在)。
也可用环境变量 TEACHER=1 / MODEL_SPECS="a=p1,b=p2" / IMPORT_SPECS=...
(便于经 make <ds>-diversity-collect 转发), 命令行参数优先。

服务器全量(复用已算好的 think JSON, 无需 GPU):
  DATASET=games .venv/bin/python -m src.analysis.collect_diversity_corpora \
      --teacher --num -1 --force \
      --import stage2=results/sidreasoner/Video_Games_think_stage2.json \
      --import stage3=results/sidreasoner/Video_Games_think_stage3_step1900.json
(--num -1 = teacher 不采样取全部约 4.9 万行; import 天然全量)
"""

import argparse
import json
import os
import sys
import time

import pandas as pd


def clean_teacher_cot(text):
    """清洗教师思维链(reasoning_path 列): 与 ReasoningActivationDataset
    一致——若以 <think> 开头则必须成对闭合(取闭合后内容), 否则视为
    畸形样本返回 None。"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = str(text).strip()
    if not text:
        return None
    if text.startswith("<think>"):
        if "</think>" not in text:
            return None
        text = text.split("</think>")[-1]
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()
    return text or None


def strip_think_tags(text):
    """清洗模型生成的思维链: 开头的 <think> 是思考块起始标签
    (闭合标签已在截断时移除), 因此剥掉标签保留内容; 空内容返回 None。"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = str(text).strip()
    if not text:
        return None
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()
    return text or None


def output_root(cfg):
    category = cfg["category"]
    return os.path.join("results", "diversity", category)


def resolve_paths(cfg):
    root = cfg["data"]["root"]
    category = cfg["category"]
    fname = f"{category}_5_2016-10-2018-11.csv"
    return {
        "narrative": os.path.join(
            root, "index", f"{category}.integrated_narrative.csv"
        ),
        "test": os.path.join(root, "test", fname),
        "index": os.path.join(root, "index", f"{category}.index.json"),
        "item": os.path.join(root, "index", f"{category}.item.json"),
    }


def load_manifest(root):
    path = os.path.join(root, "manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(root, manifest):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def write_corpus(root, name, records):
    os.makedirs(os.path.join(root, "corpora"), exist_ok=True)
    path = os.path.join(root, "corpora", f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[collect] {name}: {len(records)} 条 -> {path}")
    return path


# --------------------------------------------------------------------------
# 来源 1: 教师模型思维链(Stage 2 训练数据)
# --------------------------------------------------------------------------
def collect_teacher(cfg, num, seed):
    paths = resolve_paths(cfg)
    df = pd.read_csv(paths["narrative"])
    if num > 0 and num < len(df):
        df = df.sample(num, random_state=seed)
    records = []
    for row in df.itertuples(index=False):  # 全量数万行时比 iterrows 快
        cot = clean_teacher_cot(row.reasoning_path)
        if cot is None:
            continue
        try:
            history = ", ".join(eval(row.history_item_sid))
        except Exception:
            history = str(row.history_item_sid)
        records.append(
            {
                "user_id": str(getattr(row, "user_id", "")),
                "history_str": history,
                "target_sid": str(row.item_sid),
                "cot": cot,
                "close_found": True,
            }
        )
    return records


# --------------------------------------------------------------------------
# 来源 2: 本地生成指定 checkpoint 在测试集上的思维链
# --------------------------------------------------------------------------
def _as_id_list(x):
    """兼容 transformers 4.x(list)/5.x(BatchEncoding)两种返回。"""
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return [int(t) for t in x]


def collect_generate(cfg, model_path, num, seed, batch_size, max_new_tokens,
                     device="auto", dtype="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data.sidreasoner_data import Reasoning_Eval_Dataset
    from src.models.sidreasoner import category_phrase

    category = cfg["category"]
    paths = resolve_paths(cfg)
    phrase = category_phrase(category)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"
    torch_dtype = getattr(torch, dtype)

    print(f"[collect] 加载模型 {model_path} (device={device}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
    model.to(device).eval()
    model.config.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        for k in ("temperature", "top_p"):
            setattr(model.generation_config, k, None)

    # 与 sidreasoner_eval_think.py 相同的评测数据集与采样:
    # 同 num/seed 下不同模型拿到的测试用户完全一致
    dataset = Reasoning_Eval_Dataset(
        data_file=paths["test"],
        item_file=paths["item"],
        index_file=paths["index"],
        tokenizer=tokenizer,
        max_len=2048,
        sample=num,
        test=True,
        seed=seed,
        category=phrase,
    )
    user_ids = [str(u) for u in dataset.data["user_id"].tolist()]
    histories = dataset.get_all()
    encodings = [dataset[i] for i in range(len(dataset))]

    # 注意: 不用 HF stopping_criteria 截到 </think> —— 它是 batch 级全停
    # (batch 内任一样本闭合即截断整批, 其他样本被随机提前截断)。
    # 与官方 vLLM 评测的 per-request stop 对齐的做法是生成满
    # max_new_tokens 后按 "</think>" 文本截断。
    blocks = [
        encodings[i : i + batch_size]
        for i in range(0, len(encodings), batch_size)
    ]
    cots, close_found = [], []
    t0 = time.time()
    for bi, block in enumerate(blocks):
        block_ids = [_as_id_list(e["input_ids"]) for e in block]
        max_len = max(len(ids) for ids in block_ids)
        input_ids, attn = [], []
        for ids in block_ids:  # 左填充
            pad = max_len - len(ids)
            input_ids.append([tokenizer.pad_token_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = torch.tensor(attn, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(
            out[:, max_len:], skip_special_tokens=False
        )
        for text in texts:
            found = "</think>" in text
            close_found.append(found)
            cot = text.split("</think>")[0] if found else text
            cot = strip_think_tags(cot)
            cots.append(cot if cot is not None else "")
        done = (bi + 1) * batch_size
        rate = min(done, len(encodings)) / (time.time() - t0)
        print(
            f"[collect] {min(done, len(encodings))}/{len(encodings)} "
            f"({rate:.1f} 样本/s)",
            end="\r",
        )
    print()

    records = []
    for uid, hist, cot, found in zip(user_ids, histories, cots, close_found):
        if not cot:
            continue
        records.append(
            {
                "user_id": uid,
                "history_str": hist["history_str"],
                "target_sid": hist["output"],
                "cot": cot,
                "close_found": bool(found),
            }
        )
    del model
    return records


# --------------------------------------------------------------------------
# 来源 3: 导入服务器上 think 评测 JSON(含 cot 字段)
# --------------------------------------------------------------------------
def collect_import(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, item in enumerate(data):
        cot = strip_think_tags(item.get("cot", ""))
        if cot is None:
            continue
        # 官方 think JSON(sidreasoner_eval_think.py 的输出)没有 user_id
        # 字段, 用 #行号 兜底保持每条可区分
        records.append(
            {
                "user_id": str(item.get("user_id") or f"#{i}"),
                "history_str": item.get("history_str", ""),
                "target_sid": item.get("output", ""),
                "cot": cot,
                "close_found": bool(item.get("close_found", True)),
            }
        )
    return records


def parse_specs(pairs):
    """解析 "name=path,name=path" 形式的来源描述。"""
    specs = {}
    if not pairs:
        return specs
    for chunk in pairs:
        for item in chunk.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"来源格式应为 名字=路径, 收到: {item}")
            name, path = item.split("=", 1)
            specs[name.strip()] = path.strip()
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="采集思维链多样性分析语料(详见模块 docstring)"
    )
    parser.add_argument(
        "--teacher", action="store_true",
        default=bool(os.environ.get("TEACHER")),
        help="采集教师模型思维链(Stage 2 训练数据)",
    )
    parser.add_argument(
        "--model", action="append", default=[],
        help="本地生成: 名字=checkpoint路径(可重复; 亦可用环境变量 "
             "MODEL_SPECS='a=p1,b=p2' 一次传多个)",
    )
    parser.add_argument(
        "--import", dest="import_specs", action="append", default=[],
        help="导入已有 think 评测 JSON: 名字=json路径(可重复; 亦可用 "
             "环境变量 IMPORT_SPECS)",
    )
    parser.add_argument(
        "--num", type=int, default=int(os.environ.get("NUM", 300)),
        help="采样文档数(teacher 从训练语料采样, 模型从测试集采样; "
             "同一 num/seed 下各模型测试用户一致)",
    )
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("SEED", 42))
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=int(os.environ.get("BATCH_SIZE", 8)),
    )
    parser.add_argument(
        "--max-new-tokens", type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", 512)),
        help="思维链生成上限(默认 512, 比官方 think 评测的 256 宽松, "
             "减少截断偏差; MAX_NEW_TOKENS=256 可对齐官方)",
    )
    parser.add_argument(
        "--force", action="store_true",
        default=bool(os.environ.get("FORCE")),
        help="覆盖已存在的语料文件",
    )
    args = parser.parse_args()

    from config import load_config
    from src.train.common import resolve_category

    cfg = load_config("sidreasoner")
    cfg["category"] = resolve_category(cfg["data"])
    root = output_root(cfg)

    model_specs = parse_specs(
        args.model or ([os.environ["MODEL_SPECS"]]
                       if os.environ.get("MODEL_SPECS") else [])
    )
    import_specs = parse_specs(
        args.import_specs or ([os.environ["IMPORT_SPECS"]]
                              if os.environ.get("IMPORT_SPECS") else [])
    )

    # 默认: teacher + 本地默认 stage2 checkpoint(若存在)
    if not (args.teacher or model_specs or import_specs):
        args.teacher = True
        suffix = ""
        try:
            from src.data.sid_switch import sid_suffix

            suffix = sid_suffix(cfg["data"])
        except Exception:
            pass
        default_stage2 = os.path.join(
            "runs", "sidreasoner", f"{cfg['category']}_stage2_activation"
            f"{suffix}", "final_checkpoint"
        )
        if os.path.exists(default_stage2):
            model_specs["stage2"] = default_stage2
            print(f"[collect] 自动选用本地 stage2: {default_stage2}")

    manifest = load_manifest(root)
    plans = []  # (name, kind, callable)

    if args.teacher:
        plans.append(
            ("teacher", "train",
             lambda: collect_teacher(cfg, args.num, args.seed))
        )
    for name, path in model_specs.items():
        plans.append(
            (name, f"generate:{path}",
             lambda p=path: collect_generate(
                 cfg, p, args.num, args.seed, args.batch_size,
                 args.max_new_tokens,
             ))
        )
    for name, path in import_specs.items():
        plans.append((name, f"import:{path}",
                      lambda p=path: collect_import(p)))

    if not plans:
        raise SystemExit("未指定任何语料来源(见 --help)")

    for name, kind, fn in plans:
        out_path = os.path.join(root, "corpora", f"{name}.jsonl")
        if os.path.exists(out_path) and not args.force:
            print(f"[collect] 跳过 {name}(已存在, --force 覆盖): {out_path}")
            continue
        records = fn()
        if not records:
            print(f"[collect] {name}: 0 条有效样本, 不写文件")
            continue
        write_corpus(root, name, records)
        manifest[name] = {
            "kind": kind,
            "n_docs": len(records),
            "num": args.num,
            "seed": args.seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_manifest(root, manifest)

    print(f"[collect] 完成。运行报告: DATASET=<ds> "
          f".venv/bin/python -m src.analysis.diversity_report")


if __name__ == "__main__":
    main()
