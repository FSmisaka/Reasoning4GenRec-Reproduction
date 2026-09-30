"""
SIDReasoner Stage 3 checkpoint 合并工具

迁移自 SIDReasoner 官方仓库的 scripts/merge_fsdp_checkpoint.py:
把 verl 产出的多卡 FSDP actor checkpoint 合并为单卡 HuggingFace 模型,
供思考模式评估使用。依赖官方 verl fork(rl.verl_home), 由本脚本
自动注入 sys.path。

用法:
  .venv/bin/python scripts/sidreasoner/merge_ckpt.py \
      --checkpoint checkpoints/RecRL_Reasoning/<Category>_stage3_rl_Qwen3-1.7B/global_step_100/actor \
      --output-dir  checkpoints/RecRL_Reasoning/<Category>_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_verl():
    sys.path.insert(0, os.getcwd())
    from config import load_config

    verl_home = os.environ.get("VERL_HOME") or load_config("sidreasoner")[
        "rl"
    ]["verl_home"]
    verl_home = str(Path(verl_home).expanduser().resolve())
    if not (Path(verl_home) / "verl").is_dir():
        raise SystemExit(
            f"未找到 verl fork({verl_home}/verl)。请克隆 "
            "https://github.com/HappyPointer/SIDReasoner 并在 "
            "config/sidreasoner.py 中设置 rl.verl_home。"
        )
    sys.path.insert(0, verl_home)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FSDP actor checkpoint into single-card HF format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to global_step_xx directory or its actor subfolder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to <checkpoint>/actor_merged.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-cpu-init", action="store_true")
    return parser.parse_args()


def _resolve_actor_dir(checkpoint: Path) -> Path:
    if checkpoint.name == "actor" and checkpoint.is_dir():
        return checkpoint
    actor_dir = checkpoint / "actor"
    if actor_dir.is_dir():
        return actor_dir
    raise FileNotFoundError(f"Could not locate actor directory under {checkpoint}")


def main() -> None:
    args = _parse_args()
    _ensure_verl()
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    actor_dir = _resolve_actor_dir(checkpoint_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = actor_dir.parent / f"{actor_dir.name}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Hugging Face config files under {hf_dir}"
        )

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(output_dir),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=args.trust_remote_code,
        is_value_model=False,
        local_dir=str(actor_dir),
        hf_model_config_path=str(hf_dir),
        use_cpu_initialization=args.use_cpu_init,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()
    print(f"Merged checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
