"""
配置加载模块

按模型名称动态加载 config/<model_name>.py 中的 CONFIG 字典。
模型名与 src/models/ 下的子目录名一致(如 sasrec、tiger)。
"""

import importlib
from pathlib import Path


def load_config(model_name: str) -> dict:
    module_path = f"config.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"模型 '{model_name}' 的配置文件不存在 "
            f"({module_path}.py)。可用模型: {list_models()}"
        ) from e
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"{module_path}.py 必须定义 CONFIG 字典")
    return module.CONFIG


def list_models() -> list:
    config_dir = Path(__file__).parent
    return sorted(
        f.stem for f in config_dir.glob("*.py") if f.stem != "__init__"
    )
