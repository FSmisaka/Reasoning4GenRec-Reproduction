"""User Representation 编码器注册表(与 sid_rewards/registry.py 同风格)。"""

_USER_ENCODER_REGISTRY: dict = {}


def register_user_encoder(name: str):
    """类装饰器: 把 BaseUserEncoder 子类注册到 ``name``。"""

    def decorator(cls):
        key = str(name).lower()
        if key in _USER_ENCODER_REGISTRY:
            raise ValueError(f"user encoder '{name}' 已注册。")
        _USER_ENCODER_REGISTRY[key] = cls
        return cls

    return decorator


def list_user_encoders() -> list:
    return sorted(_USER_ENCODER_REGISTRY)


def build_user_encoder(name: str, **kwargs) -> "BaseUserEncoder":
    """按注册名构建编码器, 其余 kwargs 原样传给构造函数。"""
    from .base import BaseUserEncoder  # 延迟导入避免循环

    key = str(name).lower()
    if key not in _USER_ENCODER_REGISTRY:
        available = ", ".join(list_user_encoders()) or "<none>"
        raise KeyError(
            f"未知的 user_encoder_type '{name}', 可选: {available}。"
            "自定义编码器请继承 BaseUserEncoder 并用 @register_user_encoder 注册。"
        )
    encoder = _USER_ENCODER_REGISTRY[key](**kwargs)
    if not isinstance(encoder, BaseUserEncoder):
        raise TypeError(f"{key} 不是 BaseUserEncoder 的子类。")
    return encoder
