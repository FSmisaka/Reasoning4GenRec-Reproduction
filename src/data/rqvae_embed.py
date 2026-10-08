"""
RQ-VAE 第 1 步: item.json 文本 -> BGE 向量

启动: DATASET=games make games-rqvae-embed
      (等价: DATASET=games .venv/bin/python -m src.data.rqvae_embed)
输出: runs/rqvae/<Category>/corpus.jsonl, embeddings.npy, embed_meta.json

说明:
- item.json 的 description/categories 字段是"列表的字符串形式",
  逐字段解析后按 title/brand/categories/description 拼接;
- BGE 官方用法: passage 不加 instruction 前缀, CLS pooling + L2 归一化;
- embeddings.npy 存在且 meta 与当前配置一致时跳过(FORCE=1 强制重算)。
"""

import ast
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from config import load_config
from src.train.common import pick_device, resolve_category


def _parse_maybe_list(s):
    """'["a", "b"]' / "['a', 'b']" -> 'a b'; 普通字符串原样返回。"""
    if not isinstance(s, str) or not s.strip():
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(s)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list):
            return " ".join(
                str(v).strip() for v in value if str(v).strip()
            )
        return str(value).strip()
    return s.strip()


def build_item_text(item_id, item, fields=("title", "brand", "categories", "description")):
    builders = {
        "title": lambda: (item.get("title") or "").strip(),
        "brand": lambda: (item.get("brand") or "").strip(),
        "categories": lambda: _parse_maybe_list(
            item.get("categories") or ""
        ),
        "description": lambda: _parse_maybe_list(
            item.get("description") or ""
        ),
    }
    parts = []
    for field in fields:
        if field == "brand":
            value = builders[field]()
            if value:
                parts.append(f"Brand: {value}")
        elif field in ("categories", "description"):
            value = builders[field]()
            if value:
                parts.append(f"{field.capitalize()}: {value}")
        else:
            value = builders[field]()
            if value:
                parts.append(value)
    if not parts:
        return f"Amazon item {item_id}"
    return ". ".join(parts)


def encode_corpus(texts, model_name, batch_size, max_length, device):
    """BGE 编码: CLS pooling + L2 归一化, 返回 (N, dim) float32。"""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="bge encode"):
        batch = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        chunks.append(hidden.float().cpu())
    return torch.cat(chunks).numpy()


def output_dir(category):
    return os.path.join("runs", "rqvae", category)


def main(cfg=None):
    cfg = cfg or load_config("rqvae")
    data_cfg = cfg["data"]
    category = resolve_category(data_cfg)
    root = data_cfg["root"]
    emb_cfg = cfg["embed"]

    out_dir = output_dir(category)
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "embed_meta.json")

    with open(os.path.join(root, "index", f"{category}.item.json")) as f:
        items = json.load(f)
    item_ids = sorted(items, key=int)

    meta = {
        "model": emb_cfg["model"],
        "max_length": emb_cfg["max_length"],
        "text_fields": list(emb_cfg["text_fields"]),
        "n_items": len(item_ids),
        "category": category,
    }
    if (
        os.path.exists(emb_path)
        and os.environ.get("FORCE") != "1"
        and os.path.exists(meta_path)
        and json.load(open(meta_path)) == meta
    ):
        print(f"[embed] 缓存命中, 跳过: {emb_path} (FORCE=1 可强制重算)")
        return

    texts = [
        build_item_text(k, items[k], tuple(emb_cfg["text_fields"]))
        for k in item_ids
    ]
    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for k, text in zip(item_ids, texts):
            f.write(json.dumps({"item_id": k, "text": text}) + "\n")

    device = pick_device()
    embeddings = encode_corpus(
        texts,
        emb_cfg["model"],
        emb_cfg["batch_size"],
        emb_cfg["max_length"],
        device,
    )
    np.save(emb_path, embeddings)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"[embed] {category}: {len(texts)} items -> {embeddings.shape} "
        f"on {device}, |x| in [{norms.min():.4f}, {norms.max():.4f}]"
    )


if __name__ == "__main__":
    main()
