"""User Encoder 统一接口。

输入是每个样本"发生时可见"的历史交互物品 id 列表(0 起始, 与
index.json 的 key 一致), 输出 L2 归一化的定长用户向量。历史来源由
调用方(离线脚本)从 parquet prompt 解析, 保证不含目标物品/未来交互。
"""

import abc


class BaseUserEncoder(abc.ABC):
    """所有 User Representation 编码器的基类。

    子类必须实现 :meth:`encode_histories`, 返回 float32 ndarray,
    形状 (N, dim), 每行已 L2 归一化(零向量保持为零)。
    """

    #: 注册名, 由 @register_user_encoder 装饰器约定
    name = "base"

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """输出向量维度。"""

    @abc.abstractmethod
    def encode_histories(self, histories: list) -> "numpy.ndarray":
        """批量编码历史交互序列。

        Args:
            histories: 长度为 N 的列表, 每项为该样本发生时用户的历史
                物品 id 列表(按时间升序, 0 起始 id)。

        Returns:
            (N, dim) float32 ndarray, L2 归一化。
        """

    def finalize(self):
        """可选的资源释放钩子(由离线脚本在全部批次结束后调用)。"""
