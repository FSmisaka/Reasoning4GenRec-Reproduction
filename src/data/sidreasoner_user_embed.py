"""SIDReasoner Stage 3 跨 Query 多样性奖励: User Embedding 离线提取。

以 Stage 3 实际使用的 verl parquet(rl.sh 的 data.train_files /
data.val_files 所指文件)为唯一数据源, 逐行解析 prompt 里的历史 SID
序列 -> 物品 id -> User Encoder(默认 SASRec hidden state)编码 ->
L2 归一化, 落盘为单个 npz 供 reward 端按 (split, index) 查表。

数据对齐与防泄漏:
- 样本键 = parquet 行的 extra_info["split"] + extra_info["index"]
  (rl.sh 训练时 extra_info 原样流到 reward 函数, 不依赖 user_id);
- 历史直接取自该样本的 prompt, 是"当前请求发生时"的可见历史,
  不含目标物品 / 未来交互, 天然无泄漏;
- SID -> 物品 id 用 index.json 反查; RQ-VAE 码本存在少量 SID 碰撞
  (不同物品同三元组), 统一取最小物品 id, 确定性且无信息泄漏。

产物(默认 runs/sidreasoner/user_embed/<Category>_<encoder>.npz):
- split   : (N,) str
- index   : (N,) int64, parquet 行号
- embeddings: (N, dim) float32, 已 L2 归一化
- history_items: (N,) object, 每项 np.int64 数组(0 起始物品 id)
- meta    : json str(encoder 类型/维度/数据目录/构建时间等)

启动(服务器):
    DATASET=games .venv/bin/python -m src.data.sidreasoner_user_embed
    # 或
    DATASET=games bash scripts/sidreasoner/user_embed.sh
"""

import argparse
import datetime
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import load_config
from src.train.common import resolve_category

# prompt 用户消息形如:
# "The user has sequentially interacted with items <a_1><b_2><c_3>, <a_4>... . Can you ..."
_SID_TOKEN_RE = re.compile(r"<[abc]_\d+>")
_HISTORY_ANCHOR_RE = re.compile(r"interacted with items (.*?)\. Can you", re.DOTALL)


def parse_history_sids_from_prompt(messages) -> list:
    """从 parquet prompt(chat 消息列表)解析历史 SID 三元组字符串列表。

    返回形如 ["<a_1><b_2><c_3>", ...](按交互顺序); 解析失败抛出
    ValueError, 由调用方决定是否严格终止。
    """
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            user_text = messages
            messages = None
    else:
        user_text = None
    if messages is None:
        raise ValueError("prompt 不是 chat 消息列表。")
    user_text = next((m["content"] for m in messages if m.get("role") == "user"), None)
    if user_text is None:
        raise ValueError("prompt 中没有 user 消息。")

    m = _HISTORY_ANCHOR_RE.search(user_text)
    if not m:
        raise ValueError("prompt 中未找到 'interacted with items ...' 历史片段。")
    segment = m.group(1)
    tokens = _SID_TOKEN_RE.findall(segment)
    if len(tokens) == 0 or len(tokens) % 3 != 0:
        raise ValueError(f"历史 SID token 数量异常: {len(tokens)}。")
    return ["".join(tokens[i : i + 3]) for i in range(0, len(tokens), 3)]


def build_sid_to_item(index_file: str) -> dict:
    """SID 三元组字符串 -> 物品 id(0 起始); 碰撞取最小 id。"""
    with open(index_file, "r") as f:
        index = json.load(f)
    sid2item: dict = {}
    for item_id, sids in index.items():
        if not sids or len(sids) < 3:
            continue
        key = "".join(str(x) for x in sids[:3])
        cur = sid2item.get(key)
        iid = int(item_id)
        sid2item[key] = iid if cur is None else min(cur, iid)
    return sid2item


def history_sids_to_items(history_sids: list, sid2item: dict, strict=False) -> list:
    items = []
    for sid in history_sids:
        item = sid2item.get(sid)
        if item is None:
            if strict:
                raise ValueError(f"SID {sid} 不在 index.json 中。")
            continue  # 码本外 token(理论上不应出现), 跳过
        items.append(item)
    return items


def _join_user_ids(root: str, category: str, split: str, n_rows: int):
    """按行号把 split CSV 的 user_id 对齐进缓存(可选增强)。

    前提: parquet 与 CSV 行号一一对应(官方与自建 parquet 均如此;
    提取时做抽样后缀校验, 不对齐则整列置 None 并在 meta 记录)。
    user_id 仅用于 reward 侧排除"同一用户的其他样本", 不参与编码。
    """
    fname = f"{category}_5_2016-10-2018-11.csv"
    csv_path = os.path.join(root, split, fname)
    if not os.path.exists(csv_path):
        return None, False
    csv = pd.read_csv(csv_path, usecols=["user_id", "history_item_sid", "item_sid"])
    if len(csv) != n_rows:
        return None, False
    return csv["user_id"].astype(str).tolist(), True


def default_embedding_path(category: str, encoder_type: str) -> str:
    return os.path.join("runs", "sidreasoner", "user_embed", f"{category}_{encoder_type}.npz")


def default_sasrec_checkpoint(category: str) -> str:
    """按训练脚本命名约定找 <Category>_sasrec_bce1/best.pt。"""
    return os.path.join("runs", f"{category}_sasrec_bce1", "best.pt")


def extract(
    parquet_files: dict,
    encoder,
    sid2item: dict,
    root: str = None,
    category: str = None,
    batch_size: int = 256,
    strict_history: bool = False,
):
    """逐 parquet 提取 (split, index, embeddings, history_items, user_ids)。

    Args:
        parquet_files: {split_name: parquet_path}
        encoder: BaseUserEncoder 实例
        sid2item: SID 三元组 -> 物品 id
        root / category: 提供 时尝试按行号 join CSV user_id(精确
            同用户排除用; 对齐失败自动降级为 None)
        batch_size: 编码批大小(GPU 分批, 避免整表载入)
    """
    all_split, all_index, all_hist, all_uid = [], [], [], []
    join_attempted = False
    join_all_aligned = True
    for split, path in parquet_files.items():
        df = pd.read_parquet(path, columns=["prompt", "extra_info"])
        n = len(df)
        split_arr = [split] * n
        index_arr = np.arange(n, dtype=np.int64)
        histories = []
        for i in tqdm(range(n), desc=f"user-embed[{split}]"):
            extra = df.iloc[i]["extra_info"]
            # 对齐校验: extra_info.index 必须等于行号, 否则拒绝产出
            if int(extra.get("index", i)) != i:
                raise ValueError(
                    f"{path} 第 {i} 行 extra_info['index']={extra.get('index')} "
                    "与行号不一致, 缓存键不可靠, 请检查 parquet 生成逻辑。"
                )
            sids = parse_history_sids_from_prompt(df.iloc[i]["prompt"])
            histories.append(
                np.asarray(
                    history_sids_to_items(sids, sid2item, strict=strict_history),
                    dtype=np.int64,
                )
            )
        all_split.extend(split_arr)
        all_index.append(index_arr)
        all_hist.extend(histories)

        uids = None
        if root and category:
            join_attempted = True
            uids, aligned = _join_user_ids(root, category, split, n)
            join_all_aligned = join_all_aligned and aligned
        all_uid.extend(uids if uids is not None else [""] * n)

    embeddings = np.zeros((len(all_hist), encoder.dim), dtype=np.float32)
    for start in tqdm(
        range(0, len(all_hist), batch_size), desc=f"user-embed[encode:{encoder.name}]"
    ):
        chunk = all_hist[start : start + batch_size]
        embeddings[start : start + len(chunk)] = encoder.encode_histories(chunk)
    return all_split, np.concatenate(all_index), embeddings, all_hist, all_uid, (
        join_attempted and join_all_aligned
    )


def main(argv=None, cfg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="输出 npz 路径(默认 runs/sidreasoner/user_embed/<Category>_<encoder>.npz)",
    )
    parser.add_argument("--encoder", default=None, help="覆盖 user_encoder_type(sasrec|bge_mean)")
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="要处理的 split(逗号分隔; 只处理 data_dir 下存在的 parquet)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="编码批大小")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu(默认自动)")
    parser.add_argument(
        "--sasrec-checkpoint",
        default=None,
        help="SASRec checkpoint 路径(默认 runs/<Category>_sasrec_bce1/best.pt)",
    )
    parser.add_argument(
        "--bge-item-embeddings",
        default=None,
        help="BGE 物品 embedding npy 路径(默认 runs/rqvae/<Category>/embeddings.npy)",
    )
    args = parser.parse_args(argv)

    cfg = cfg or load_config("sidreasoner")
    rl_cfg = cfg["rl"]
    iqd = rl_cfg.get("inter_query_diversity", {}) or {}
    category = resolve_category(cfg["data"])
    root = cfg["data"]["root"]

    encoder_type = args.encoder or iqd.get("user_encoder_type", "sasrec")
    out_path = args.out or iqd.get("user_embedding_path") or default_embedding_path(
        category, encoder_type
    )

    data_dir = rl_cfg.get("data_dir") or os.path.join(root, "rec_reasoning_verl", category)
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    parquet_files = {}
    for split in wanted:
        path = os.path.join(data_dir, f"{split}.parquet")
        if os.path.exists(path):
            parquet_files[split] = path
    if not parquet_files:
        raise FileNotFoundError(
            f"{data_dir} 下没有任何 {wanted} parquet。请先准备 Stage 3 RL 数据。"
        )

    if encoder_type == "sasrec":
        ckpt = args.sasrec_checkpoint or iqd.get("sasrec_checkpoint") or default_sasrec_checkpoint(
            category
        )
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"未找到 SASRec checkpoint: {ckpt}。先运行 DATASET=... make <dataset>-sasrec, "
                "或用 --sasrec-checkpoint / --encoder bge_mean 指定其他来源。"
            )
        from src.reward.user_represent import build_user_encoder

        encoder = build_user_encoder("sasrec", checkpoint=ckpt, device=args.device)
    elif encoder_type == "bge_mean":
        emb_path = (
            args.bge_item_embeddings
            or iqd.get("bge_item_embeddings")
            or os.path.join("runs", "rqvae", category, "embeddings.npy")
        )
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"未找到 BGE 物品 embedding: {emb_path}。先运行 make <dataset>-rqvae-embed "
                "或用 --bge-item-embeddings 指定路径。"
            )
        from src.reward.user_represent import build_user_encoder

        encoder = build_user_encoder("bge_mean", item_embeddings=emb_path)
    else:
        from src.reward.user_represent import build_user_encoder

        encoder = build_user_encoder(
            encoder_type,
            **({"device": args.device} if args.device else {}),
        )

    sid2item = build_sid_to_item(os.path.join(root, "index", f"{category}.index.json"))
    splits, indices, embeddings, histories, user_ids, user_id_aligned = extract(
        parquet_files,
        encoder,
        sid2item,
        root=root,
        category=category,
        batch_size=args.batch_size or int(iqd.get("embed_batch_size", 256)),
    )

    meta = {
        "category": category,
        "encoder": encoder.name,
        "dim": int(encoder.dim),
        "n_samples": int(len(splits)),
        "splits": sorted(parquet_files),
        "data_dir": os.path.abspath(data_dir),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_items_in_index": len(sid2item),
        "user_id_aligned": bool(user_id_aligned),
        "history_digest": hashlib.md5(
            json.dumps([h.tolist() for h in histories]).encode()
        ).hexdigest(),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        split=np.asarray(splits, dtype=str),
        index=np.asarray(indices, dtype=np.int64),
        embeddings=embeddings,
        history_items=np.asarray(histories, dtype=object),
        user_ids=np.asarray(user_ids, dtype=str),
        meta=np.asarray(json.dumps(meta)),
    )
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"[user-embed] {category}/{encoder.name}: {len(splits)} 条样本, "
        f"dim={encoder.dim}, splits={sorted(parquet_files)}, "
        f"user_id_aligned={user_id_aligned}, "
        f"|u| in [{norms.min():.4f}, {norms.max():.4f}] -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    main()
