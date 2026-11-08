"""结果两级目录路由单测(src/eval/result_routing.py + organize_results.py)。"""

import sys
from pathlib import Path

import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.eval.result_routing import (  # noqa: E402
    build_result_path,
    dataset_dir,
    model_step_label,
    result_dir,
    variant_of_model,
)


def test_dataset_dir_mapping():
    assert dataset_dir("Video_Games") == "Games"
    assert dataset_dir("Office_Products") == "Office"
    assert dataset_dir("Industrial_and_Scientific") == "Industrial"
    with pytest.raises(ValueError):
        dataset_dir("Unknown")


def test_model_step_label():
    # step 数从路径中的 global_step_N 提取(MODEL 常指向其下的 actor_merged)
    assert model_step_label(
        "checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam/global_step_1500/actor_merged"
    ) == "step_1500"
    # stage1/stage2 checkpoint
    assert model_step_label(
        "runs/sidreasoner/Video_Games_stage2_activation/final_checkpoint"
    ) == "stage2"
    assert model_step_label(
        "runs/sidreasoner/Office_Products_stage1_sft_own/final_checkpoint"
    ) == "stage1"
    # 识别不了 -> 目录名原样
    assert model_step_label("/some/path/actor_merged") == "actor_merged"


def test_variant_of_model():
    assert variant_of_model(".../Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged") == "steprule"
    assert variant_of_model(".../Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam/global_step_100/actor_merged") == "rolloutbeam"
    assert variant_of_model(".../Video_Games_stage3_rl_Qwen3-1.7B_own_rolloutbeam_div/global_step_100/actor_merged") == "rolloutbeam_div"


def test_result_dir_six_kinds():
    beam = ".../Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam/global_step_1/actor_merged"
    div = ".../Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam_div/global_step_1/actor_merged"
    orig = ".../Video_Games_stage3_rl_Qwen3-1.7B/global_step_1/actor_merged"
    assert result_dir("Video_Games", "nonthink") == "nonthink"                     # 1
    assert result_dir("Video_Games", "short", max_len=3) == "le3"                  # 2
    assert result_dir("Video_Games", "think", model=orig) == "think_steprule"      # 3
    assert result_dir("Video_Games", "think", model=beam) == "think_rolloutbeam"   # 4
    assert result_dir("Video_Games", "think", model=div) == "think_rolloutbeam_div"  # 5
    assert result_dir("Video_Games", "think", model=orig, temperature=1.5) == "think_steprule_temp"  # 6
    # 泛化: beam 模型 + 温度采样
    assert result_dir("Video_Games", "think", model=beam, temperature=1.5) == "think_rolloutbeam_temp"
    assert result_dir("Video_Games", "short", max_len=5) == "le5"


def test_build_result_path_think():
    p = build_result_path(
        "Video_Games", "think",
        "checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam/global_step_900/actor_merged",
    )
    assert p == "results/sidreasoner/Games/think_rolloutbeam/step_900.json"
    # 自建 SID 后缀
    p = build_result_path(
        "Video_Games", "think",
        ".../Video_Games_stage3_rl_Qwen3-1.7B_own_rolloutbeam/global_step_900/actor_merged",
        sid_sfx="_own",
    )
    assert p == "results/sidreasoner/Games/think_rolloutbeam/step_900_own.json"


def test_build_result_path_nonthink_and_short():
    assert build_result_path(
        "Video_Games", "nonthink", "runs/sidreasoner/Video_Games_stage1_sft/final_checkpoint"
    ) == "results/sidreasoner/Games/nonthink/stage1.json"
    assert build_result_path(
        "Video_Games", "short", "stage2", max_len=3, mode_prefix="eval"
    ) == "results/sidreasoner/Games/le3/eval_stage2.json"
    assert build_result_path(
        "Video_Games", "short", "stage2", max_len=3, mode_prefix="think"
    ) == "results/sidreasoner/Games/le3/think_stage2.json"


def test_organize_classify():
    sys.path.insert(0, str(R4GR_ROOT / "scripts" / "sidreasoner"))
    import organize_results as org

    f = org.classify
    # 1) final_result_<C>.<label>.json(非思考, 手动重命名, 点分隔)
    assert f("final_result_Video_Games.stage1.json") == ("Games", "nonthink/stage1.json")
    assert f("final_result_Video_Games.stage3_step800.json") == ("Games", "nonthink/step_800.json")
    # 旧默认命名(无模型信息)
    assert f("final_result_Video_Games.json") == ("Games", "nonthink/legacy.json")
    assert f("final_result_Video_Games_own.json") == ("Games", "nonthink/legacy_own.json")
    # 2) le3 双模式
    assert f("Video_Games_le3_eval_actor_merged.json") == ("Games", "le3/eval_actor_merged.json")
    assert f("Video_Games_le3_think_stage3_step100.json") == ("Games", "le3/think_step_100.json")
    # 3) 温度采样: 单 seed(旧默认) 与手动重命名的 summary
    assert f("Video_Games_think_actor_merged_seed42.json") == ("Games", "think_steprule_temp/actor_merged_seed42.json")
    assert f("Video_Games_think_stage3_temp_step1900.json") == ("Games", "think_steprule_temp/step_1900_summary.json")
    assert f("Video_Games_think_actor_merged_summary.json") == ("Games", "think_steprule_temp/actor_merged_summary.json")
    # 4) 贪心思考模式: stageX_stepN 标签
    assert f("Video_Games_think_stage3_step100.json") == ("Games", "think_steprule/step_100.json")
    assert f("Video_Games_think_stage2.json") == ("Games", "think_steprule/stage2.json")
    # 旧默认贪心命名 + own 后缀
    assert f("Video_Games_think_actor_merged.json") == ("Games", "think_steprule/actor_merged.json")
    assert f("Video_Games_think_actor_merged_own.json") == ("Games", "think_steprule/actor_merged_own.json")
    assert f("Video_Games_think_actor_merged_own_seed42.json") == ("Games", "think_steprule_temp/actor_merged_own_seed42.json")
    # 非思考旧默认命名
    assert f("Office_Products_stage2.json") == ("Office", "nonthink/stage2.json")
    assert f("unrelated.json") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
