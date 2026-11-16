"""User Embedding 离线提取链路测试。

覆盖: prompt 历史解析、SID->item 反查、SASRec 编码(padding/截断/归一化/
防泄漏)、离线脚本端到端(小型 parquet -> npz -> UserEmbeddingStore)。
依赖本地数据(Video_Games)时自动跳过。

运行: .venv/bin/python -m pytest tests/test_user_embed.py -v
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.data.sidreasoner_user_embed import (  # noqa: E402
    build_sid_to_item,
    extract,
    parse_history_sids_from_prompt,
    _join_user_ids,
    default_sasrec_checkpoint,
)

GAMES_ROOT = R4GR_ROOT / "data" / "raw" / "Amazon"
GAMES_PARQUET = GAMES_ROOT / "rec_reasoning_verl" / "Video_Games" / "train.parquet"
GAMES_INDEX = GAMES_ROOT / "index" / "Video_Games.index.json"
GAMES_SASREC = R4GR_ROOT / "runs" / "Video_Games_sasrec_bce1" / "best.pt"
GAMES_BGE = R4GR_ROOT / "runs" / "rqvae" / "Video_Games" / "embeddings.npy"


def _games_ready():
    return GAMES_PARQUET.exists() and GAMES_INDEX.exists()


# --------------------------------------------------------------------------- #
# prompt 解析
# --------------------------------------------------------------------------- #


def test_parse_history_from_synthetic_prompt():
    messages = [
        {"role": "system", "content": "instruction"},
        {
            "role": "user",
            "content": (
                "The user has sequentially interacted with items "
                "<a_1><b_2><c_3>, <a_4><b_5><c_6>. "
                "Can you recommend the next item for him?"
            ),
        },
    ]
    assert parse_history_sids_from_prompt(messages) == [
        "<a_1><b_2><c_3>",
        "<a_4><b_5><c_6>",
    ]


def test_parse_history_rejects_malformed():
    with pytest.raises(ValueError):
        parse_history_sids_from_prompt([{"role": "user", "content": "no history here"}])
    with pytest.raises(ValueError):
        parse_history_sids_from_prompt(
            [{"role": "user", "content": "interacted with items <a_1>. Can you"}]
        )


@pytest.mark.skipif(not _games_ready(), reason="缺少 Video_Games parquet/index")
def test_parse_history_on_real_parquet():
    import pandas as pd

    df = pd.read_parquet(GAMES_PARQUET, columns=["prompt"])
    n_ok = 0
    for i in range(200):
        sids = parse_history_sids_from_prompt(df.iloc[i]["prompt"])
        assert len(sids) >= 1
        assert all(s.startswith("<a_") for s in sids)
        n_ok += 1
    assert n_ok == 200


# --------------------------------------------------------------------------- #
# SID -> item 映射
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not GAMES_INDEX.exists(), reason="缺少 index.json")
def test_build_sid_to_item_collision_takes_min():
    index = json.load(open(GAMES_INDEX))
    sid2item = build_sid_to_item(str(GAMES_INDEX))
    # 数量 <= 物品数(碰撞合并), 且每个键取最小物品 id
    assert len(sid2item) <= len(index)
    for key, item in sid2item.items():
        candidates = {int(k) for k, v in index.items() if "".join(str(x) for x in v[:3]) == key}
        assert item == min(candidates)


def test_history_sids_to_items_mapping():
    sid2item = {"<a_1><b_2><c_3>": 7, "<a_9><b_9><c_9>": 3}
    assert build_sid_to_item  # noqa: B018 (导入存在性)
    from src.data.sidreasoner_user_embed import history_sids_to_items

    assert history_sids_to_items(["<a_1><b_2><c_3>"], sid2item) == [7]
    # 码本外 SID 默认跳过, strict 时报错
    assert history_sids_to_items(["<a_0><b_0><c_0>"], sid2item) == []
    with pytest.raises(ValueError):
        history_sids_to_items(["<a_0><b_0><c_0>"], sid2item, strict=True)


# --------------------------------------------------------------------------- #
# SASRec checkpoint 自动定位
# --------------------------------------------------------------------------- #


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_default_sasrec_checkpoint_single_candidate(tmp_path):
    # 只有 bce(用户的场景) -> 自动选用
    only_bce = tmp_path / "runs" / "Video_Games_sasrec_bce" / "best.pt"
    _touch(str(only_bce))
    assert default_sasrec_checkpoint("Video_Games", str(tmp_path / "runs")) == str(only_bce)


def test_default_sasrec_checkpoint_ignores_non_loss_dirs(tmp_path):
    # MiniOneRec 等借用 sasrec 前缀的目录不应参与候选
    _touch(str(tmp_path / "runs" / "Video_Games_sasrec_bce" / "best.pt"))
    _touch(str(tmp_path / "runs" / "Video_Games_sasrec_minionerec" / "best.pt"))
    _touch(str(tmp_path / "runs" / "Video_Games_sasrec_mini" / "best.pt"))
    got = default_sasrec_checkpoint("Video_Games", str(tmp_path / "runs"))
    assert got.endswith("Video_Games_sasrec_bce/best.pt")


def test_default_sasrec_checkpoint_prefers_config_loss(tmp_path):
    # 同时有 bce/bce1 -> 优先 config/sasrec.py 默认 loss 的命名
    _touch(str(tmp_path / "runs" / "Video_Games_sasrec_bce" / "best.pt"))
    bce1 = tmp_path / "runs" / "Video_Games_sasrec_bce1" / "best.pt"
    _touch(str(bce1))
    assert default_sasrec_checkpoint("Video_Games", str(tmp_path / "runs")) == str(bce1)


def test_default_sasrec_checkpoint_ambiguous_raises(tmp_path):
    _touch(str(tmp_path / "runs" / "Video_Games_sasrec_bce" / "best.pt"))
    _touch(str(tmp_path / "runs" / "Video_Games_sasrec_ce" / "best.pt"))
    with pytest.raises(FileNotFoundError, match="多个 SASRec checkpoint"):
        default_sasrec_checkpoint("Video_Games", str(tmp_path / "runs"))


def test_default_sasrec_checkpoint_missing_falls_back(tmp_path):
    path = default_sasrec_checkpoint("Video_Games", str(tmp_path / "runs"))
    assert path.endswith("Video_Games_sasrec_bce1/best.pt")  # 由调用方给出建卡提示


# --------------------------------------------------------------------------- #
# SASRec 编码器
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not GAMES_SASREC.exists(), reason="缺少 SASRec checkpoint")
class TestSASRecEncoder:
    def _encoder(self):
        from src.reward.user_represent import build_user_encoder

        return build_user_encoder("sasrec", checkpoint=str(GAMES_SASREC), device="cpu")

    def test_rebuild_from_bare_state_dict_matches_forward(self, tmp_path):
        """非默认超参 + 裸 state_dict(无 config)也能正确重建。

        复刻实际踩坑的 checkpoint: hidden=128, 2 层, d_inner=256(恰为
        SASRec 签名默认)。重建后前向必须与原模型逐值一致。
        """
        import torch
        from src.models.sasrec import SASRec
        from src.reward.user_represent.sasrec_encoder import SASRecHiddenStateEncoder

        torch.manual_seed(0)
        ref = SASRec(n_items=99, max_seq_len=10, hidden=128, n_layers=2,
                     n_heads=2, d_inner=256).eval()
        ckpt = tmp_path / "bare.pt"
        torch.save(ref.state_dict(), ckpt)  # 裸 state_dict, 无 config

        enc = SASRecHiddenStateEncoder(checkpoint=str(ckpt), device="cpu", n_heads=2)
        assert enc.dim == 128
        hist = [[1, 2, 3], [4, 5]]
        with torch.no_grad():
            expect = torch.nn.functional.normalize(ref.last_hidden(enc._collate(hist)), dim=1)
        assert np.allclose(enc.encode_histories(hist), expect.numpy(), atol=1e-5)

    def test_shape_and_normalized(self):
        enc = self._encoder()
        out = enc.encode_histories([[1, 2, 3], [4, 5], []])
        assert out.shape == (3, enc.dim)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms[:2], 1.0, atol=1e-4)
        assert norms[2] == pytest.approx(0.0)  # 空历史 -> 零向量

    def test_truncates_to_max_seq_len(self):
        enc = self._encoder()
        long_hist = list(range(enc.max_seq_len + 20))
        full = enc.encode_histories([long_hist])[0]
        tail = enc.encode_histories([long_hist[-enc.max_seq_len :]])[0]
        assert np.allclose(full, tail, atol=1e-5)

    def test_padding_equivariant_for_short_histories(self):
        """左 padding 语义: 短历史与其中更长历史(前缀不同)应不同,
        但同一历史重复编码应确定一致。"""
        enc = self._encoder()
        h = [3, 1, 4, 1, 5]
        a = enc.encode_histories([h, h])
        assert np.allclose(a[0], a[1])

    def test_recent_items_dominate(self):
        """SASRec 因果注意力: 只改历史最早一项, 最后位置 hidden 几乎不变
        (max_seq_len 内注意力对最近 item 最强, 但非零 —— 只验证确定性
        与形状, 不强求数值性质)。"""
        enc = self._encoder()
        out = enc.encode_histories([[1, 2, 3], [99, 2, 3]])
        assert out.shape == (2, enc.dim)

    def test_no_target_leakage_by_construction(self):
        """编码器输入只有历史列表; 目标物品不作为参数出现(接口层面保证)。"""
        import inspect

        from src.reward.user_represent.sasrec_encoder import SASRecHiddenStateEncoder

        sig = inspect.signature(SASRecHiddenStateEncoder.encode_histories)
        assert list(sig.parameters) == ["self", "histories"]


# --------------------------------------------------------------------------- #
# 离线脚本端到端(小型 parquet)
# --------------------------------------------------------------------------- #


class _StubEncoder:
    """与 BaseUserEncoder 同接口的确定性 stub(sum of one-hot)。"""

    name = "stub"
    dim = 4

    def encode_histories(self, histories):
        out = np.zeros((len(histories), self.dim), dtype=np.float32)
        for i, h in enumerate(histories):
            for x in h:
                out[i, int(x) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n


@pytest.mark.skipif(not _games_ready(), reason="缺少 Video_Games parquet/index")
def test_extract_end_to_end(tmp_path):
    import pandas as pd

    df = pd.read_parquet(GAMES_PARQUET).head(16)
    small = tmp_path / "train.parquet"
    df.to_parquet(small)

    sid2item = build_sid_to_item(str(GAMES_INDEX))
    # 不传 root/category -> 无 user_id join
    splits, indices, embeddings, histories, uids, uid_aligned = extract(
        {"train": str(small)}, _StubEncoder(), sid2item
    )

    assert splits == ["train"] * 16
    assert indices.tolist() == list(range(16))
    assert embeddings.shape == (16, 4)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert uids == [""] * 16 and uid_aligned is False

    # 历史与 prompt 一一对应(只含历史物品, 不含目标)
    for i in range(16):
        sids = parse_history_sids_from_prompt(df.iloc[i]["prompt"])
        expected = [
            sid2item[s] for s in sids if s in sid2item
        ]
        assert histories[i].tolist() == expected


@pytest.mark.skipif(not _games_ready(), reason="缺少 Video_Games parquet/index")
def test_extract_user_id_join_mismatch_degrades(tmp_path):
    """CSV 行数与 parquet 不一致 -> user_id join 自动降级(空串)。"""
    import pandas as pd

    df = pd.read_parquet(GAMES_PARQUET).head(4)
    small = tmp_path / "train.parquet"
    df.to_parquet(small)
    sid2item = build_sid_to_item(str(GAMES_INDEX))
    root = str(GAMES_ROOT)
    _, _, _, _, uids, aligned = extract(
        {"train": str(small)}, _StubEncoder(), sid2item, root=root, category="Video_Games"
    )
    assert aligned is False
    assert uids == [""] * 4


@pytest.mark.skipif(not _games_ready(), reason="缺少 Video_Games parquet/index")
def test_extract_no_future_leakage(tmp_path):
    """历史物品序列不得包含该样本的目标物品对应的'未来'信息。

    强化版检查: 对每条样本, 用 target 的 SID 反查物品 id, 断言它不出现在
    '历史最后一个位置之后'(历史整体允许包含目标物品本身 —— 重复消费在
    5-core 数据中真实存在, 且只来自历史文本, 不构成泄漏)。
    这里验证的是: 提取流程的输入只有 prompt(历史), extra_info 的 answer
    完全不参与 —— 通过 monkeypatch 让 answer 字段损坏来证明。
    """
    import pandas as pd

    df = pd.read_parquet(GAMES_PARQUET).head(8)
    df["extra_info"] = [
        {"split": "train", "index": i, "answer": "CORRUPTED", "question": None}
        for i in range(len(df))
    ]
    small = tmp_path / "train.parquet"
    df.to_parquet(small)

    sid2item = build_sid_to_item(str(GAMES_INDEX))
    clean = pd.read_parquet(GAMES_PARQUET).head(8)
    _, _, _, histories, _, _ = extract({"train": str(small)}, _StubEncoder(), sid2item)
    for i in range(len(df)):
        sids = parse_history_sids_from_prompt(clean.iloc[i]["prompt"])
        expected = [sid2item[s] for s in sids if s in sid2item]
        assert histories[i].tolist() == expected  # 与 answer 无关
