"""离线 User Embedding 缓存的读取与查表。

缓存文件由 src/data/sidreasoner_user_embed.py 生成(npz):
split / index / embeddings / history_items / meta。

reward 端只依赖本模块提供的 (split, index) -> 向量/历史 映射,
与 User Encoder 的具体实现完全解耦。
"""

import json

import numpy as np


class UserEmbeddingStore:
    """懒加载、进程内单例安全的查表器。"""

    def __init__(self, npz_path: str):
        import os

        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"User embedding 缓存不存在: {npz_path}。"
                "先运行 DATASET=... make <dataset>-sidreasoner-user-embed 生成。"
            )
        data = np.load(npz_path, allow_pickle=True)
        self.path = npz_path
        self.splits = data["split"].astype(str)
        self.indices = data["index"].astype(np.int64)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.history_items = [np.asarray(h, dtype=np.int64) for h in data["history_items"]]
        # user_id 为可选字段(精确同用户排除; 旧缓存/未对齐时缺失 -> 空)
        self.user_ids = (
            data["user_ids"].astype(str) if "user_ids" in data else np.asarray([""] * len(self.splits))
        )
        self.meta = json.loads(str(data["meta"][()])) if "meta" in data else {}
        if not (len(self.splits) == len(self.indices) == len(self.embeddings) == len(self.user_ids)):
            raise ValueError(f"{npz_path} 内部字段长度不一致。")

        self._lookup = {
            (self.splits[i], int(self.indices[i])): i for i in range(len(self.splits))
        }
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    def __len__(self):
        return len(self.splits)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def has(self, split: str, index: int) -> bool:
        return (str(split), int(index)) in self._lookup

    def lookup(self, keys: list):
        """按键列表批量查表。

        Args:
            keys: [(split, index), ...]

        Returns:
            (embeddings: (N, dim) float32 ndarray,
             histories: list[np.ndarray],
             user_ids: list[str], 空串表示未知)

        Raises:
            KeyError: 任一键缺失时, 报出前若干缺失键(提示重建缓存)。
        """
        missing = [k for k in keys if not self.has(*k)]
        if missing:
            raise KeyError(
                f"User embedding 缓存缺 {len(missing)} 个键, 例如 {missing[:5]}。"
                f"缓存: {self.path} (meta={self.meta})。"
                "请确认缓存与训练 parquet 来自同一 data_dir, 必要时重跑"
                " make <dataset>-sidreasoner-user-embed。"
            )
        rows = [self._lookup[(str(k[0]), int(k[1]))] for k in keys]
        return (
            self.embeddings[rows],
            [self.history_items[r] for r in rows],
            [str(self.user_ids[r]) for r in rows],
        )
