"""BGE 物品 embedding 均值用户编码器(对照实现)。

物品向量来自 RQ-VAE 管线第一步的 BGE 语义编码
(src/data/rqvae_embed.py -> runs/rqvae/<Category>/embeddings.npy,
BAAI/bge-base-en-v1.5, title/brand, dim=768, 已 L2 归一化, 行序 =
sorted(item.json keys, key=int) 即 0 起始物品 id)。

用户向量 = 历史物品向量的均值, 再 L2 归一化。三个数据集都有现成
embedding, 可在无 SASRec checkpoint 时使用(Office/Industrial)。
"""

import numpy as np

from .base import BaseUserEncoder
from .registry import register_user_encoder


@register_user_encoder("bge_mean")
class MeanItemEmbeddingEncoder(BaseUserEncoder):
    """u_q = normalize(mean(BGE(item) for item in H_q))。"""

    name = "bge_mean"

    def __init__(self, item_embeddings: str):
        emb = np.load(item_embeddings)
        if emb.ndim != 2:
            raise ValueError(f"{item_embeddings} 应为 (N, dim) 二维数组。")
        self._emb = emb.astype(np.float32)
        self._n_items = int(emb.shape[0])

    @property
    def dim(self) -> int:
        return int(self._emb.shape[1])

    @property
    def n_items(self) -> int:
        return self._n_items

    def encode_histories(self, histories: list) -> np.ndarray:
        out = np.zeros((len(histories), self.dim), dtype=np.float32)
        for i, hist in enumerate(histories):
            ids = [int(x) for x in hist if 0 <= int(x) < self._n_items]
            if ids:
                out[i] = self._emb[ids].mean(axis=0)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return out / norm
