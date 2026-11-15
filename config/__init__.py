"""
配置加载模块

按模型名称动态加载 config/<model_name>.py 中的 CONFIG 字典。
模型名与 src/models/ 下的子目录名一致(如 sasrec、tiger)。

支持快照覆盖: 设置环境变量 CONFIG_SNAPSHOT_<大写模型名>=<py文件路径>
后, load_config 改从该文件加载(文件需定义 CONFIG 字典), 与 config/ 目录
解耦 —— 用于长任务批量评测: 启动时快照 config, 循环内不受后续改动影响。
"""

import importlib.util
import os
from pathlib import Path


def load_config(model_name: str) -> dict:
    snapshot = os.environ.get(f"CONFIG_SNAPSHOT_{model_name.upper()}")
    if snapshot:
        module_path = snapshot
        spec = importlib.util.spec_from_file_location(
            f"config_snapshot_{model_name}", snapshot
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
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
