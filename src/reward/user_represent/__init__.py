"""User Representation 编码器注册表。

离线脚本(src/data/sidreasoner_user_embed.py)通过 ``build_user_encoder``
按配置构建编码器; reward 侧(src/reward/interdiv/)只消费离线产出的
User Embedding 文件, 不依赖任何编码器实现 —— 更换用户表征方式时
只需新增编码器类并在此注册, reward 计算逻辑无需改动。

注册方式::

    from .base import BaseUserEncoder
    from .registry import register_user_encoder

    @register_user_encoder("my_encoder")
    class MyEncoder(BaseUserEncoder):
        ...
"""

from .registry import build_user_encoder, list_user_encoders, register_user_encoder
from .base import BaseUserEncoder  # noqa: F401  (供子类导入)


def _load_builtin_encoders():
    """导入内置编码器, 触发注册。"""
    from . import sasrec_encoder  # noqa: F401
    from . import bge_mean_encoder  # noqa: F401


_load_builtin_encoders()

__all__ = [
    "BaseUserEncoder",
    "build_user_encoder",
    "list_user_encoders",
    "register_user_encoder",
]
