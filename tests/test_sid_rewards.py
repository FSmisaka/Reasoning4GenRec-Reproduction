"""Stage 3 reward 单测: 可插拔 SID reward 包 + OneRec-Think Rollout-Beam。

纯 Python(reward 包无 torch/verl 依赖), 直接跑:
    .venv/bin/python -m pytest tests/test_sid_rewards.py -v

beam 条件截断用例需要 SIDReasoner verl fork(默认 ../SIDReasoner,
可用 VERL_HOME 覆盖), 缺失时自动跳过。
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
ENTRY_PATH = R4GR_ROOT / "src" / "reward" / "direct_recommendation_rolloutbeam.py"

sys.path.insert(0, str(R4GR_ROOT))

from src.reward.sid_rewards import (  # noqa: E402
    BEAM_SEP,
    build_sid_reward,
    extract_sid_tokens,
    extract_solution,
    parse_beam_candidates,
)

GT = "<a_1><b_2><c_3>"


# --------------------------------------------------------------------------- #
# 解析工具
# --------------------------------------------------------------------------- #


def test_extract_sid_tokens():
    assert extract_sid_tokens("xx <a_1> yy <b_2><c_3> zz") == ["<a_1>", "<b_2>", "<c_3>"]


def test_extract_solution_after_think():
    resp = "some reasoning </think>\n\n<a_1><b_2><c_3>"
    assert extract_solution(resp) == ["<a_1>", "<b_2>", "<c_3>"]
    assert extract_solution("no think marker <a_1><b_2>") is None


def test_parse_beam_candidates_from_concatenated_string():
    beam_str = "<a_9><b_9><c_9><|beam_sep|>\n<a_1><b_2><c_4><|beam_sep|>\n"
    cands = parse_beam_candidates(beam_str, num_sid=3)
    assert cands == [["<a_9>", "<b_9>", "<c_9>"], ["<a_1>", "<b_2>", "<c_4>"]]


def test_parse_beam_candidates_empty_fragments_dropped():
    beam_str = "<|beam_sep|>\n<|beam_sep|>\n<a_1><b_2><c_3>"
    assert parse_beam_candidates(beam_str, num_sid=3) == [["<a_1>", "<b_2>", "<c_3>"]]
    assert parse_beam_candidates(None) == []


def test_beam_sep_constant():
    assert BEAM_SEP == "<|beam_sep|>"
    # 分隔符必须与 SIDReasoner verl fork 的 rollout 侧一致(若存在该仓库)
    verl_root = Path(os.environ.get("VERL_HOME", R4GR_ROOT.parent / "SIDReasoner"))
    rollout_src = verl_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout_spmd.py"
    if rollout_src.exists():
        src = rollout_src.read_text()
        assert "from verl.utils.reward_score.sid_rewards.base import BEAM_SEP" in src
        assert "sep: str = BEAM_SEP" in src


# --------------------------------------------------------------------------- #
# OneRec-Think Rollout-Beam reward
# --------------------------------------------------------------------------- #


def make_reward(**kwargs):
    return build_sid_reward({"reward_type": "onerec_rollout_beam", **kwargs})


def test_rollout_beam_takes_max_over_beams():
    reward = make_reward()
    extra = {"beam_search_results": "<a_9><b_9><c_9><|beam_sep|>\n<a_1><b_2><c_4><|beam_sep|>\n<a_0><b_0><c_0>"}
    resp = "reasoning </think>\n\n<a_1><b_2><c_4>"
    # 最好的 beam 对了第 1、2 层 -> 奖励 2(论文 Eq.6, 不归一)
    assert reward.compute("rec/test", resp, GT, extra) == 2.0


def test_rollout_beam_full_match_and_normalize():
    extra = {"beam_search_results": "<a_1><b_2><c_3>"}
    resp = "r </think>\n\n<a_1><b_2><c_3>"
    assert make_reward().compute("rec/test", resp, GT, extra) == 3.0
    assert make_reward(normalize=True).compute("rec/test", resp, GT, extra) == 1.0


def test_rollout_beam_prefix_mode():
    # position 模式: 第 3 层独立匹配 -> 1
    assert make_reward(match_mode="position").matched_codes(
        ["<a_9>", "<b_9>", "<c_3>"], ["<a_1>", "<b_2>", "<c_3>"]
    ) == 1
    # prefix 模式: 第一层不匹配即断链 -> 0 / 1
    reward_p = make_reward(match_mode="prefix")
    assert reward_p.matched_codes(["<a_9>", "<b_9>", "<c_3>"], ["<a_1>", "<b_2>", "<c_3>"]) == 0
    assert reward_p.matched_codes(["<a_1>", "<b_9>", "<c_3>"], ["<a_1>", "<b_2>", "<c_3>"]) == 1


def test_rollout_beam_no_think_gets_zero():
    extra = {"beam_search_results": "<a_1><b_2><c_3>"}
    assert make_reward().compute("rec/test", "unfinished reasoning", GT, extra) == 0.0


def test_rollout_beam_empty_beams_get_zero():
    resp = "r </think>\n\n<a_1><b_2><c_4>"
    assert make_reward().compute("rec/test", resp, GT, {"beam_search_results": ""}) == 0.0


def test_rollout_beam_missing_results_raises_with_hint():
    with pytest.raises(RuntimeError, match="sid_beam_size"):
        make_reward().compute("rec/test", "r </think>\n\n<a_1>", GT, {})


def test_rollout_beam_missing_results_can_be_tolerated():
    reward = make_reward(require_beam_results=False)
    assert reward.compute("rec/test", "r </think>\n\n<a_1>", GT, {}) == 0.0


def test_rollout_beam_short_ground_truth_gets_zero():
    extra = {"beam_search_results": "<a_1><b_2><c_3>"}
    assert make_reward().compute("rec/test", "r </think>\n\n<a_1>", "<a_1>", extra) == 0.0


def test_registry_unknown_type_raises():
    with pytest.raises(KeyError, match="reward_type"):
        build_sid_reward({"reward_type": "nope"})


def test_singleton_build():
    assert make_reward() is make_reward()


# --------------------------------------------------------------------------- #
# verl 入口(src/reward/direct_recommendation_rolloutbeam.py)
# --------------------------------------------------------------------------- #


def load_entry_module():
    # 与 verl get_custom_reward_fn 相同的按文件路径加载方式
    spec = importlib.util.spec_from_file_location("custom_module", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entry_point_rule_base_reward():
    module = load_entry_module()
    extra = {"beam_search_results": "<a_1><b_2><c_9><|beam_sep|>\n<a_1><b_9><c_9>"}
    resp = "reasoning </think>\n\n<a_1><b_2><c_9>"
    score = module.rule_base_reward("rec/test", resp, GT, extra, reward_type="onerec_rollout_beam")
    assert score == 2.0


def test_entry_point_default_item_info_path(monkeypatch):
    module = load_entry_module()
    monkeypatch.setenv("DATASET", "games")
    path = module._default_item_info_path()
    assert path.endswith("Video_Games_5_2016-10-2018-11.txt")
    assert Path(path).exists()


def test_entry_point_format_bonus_uses_dataset_info(monkeypatch):
    module = load_entry_module()
    monkeypatch.setenv("DATASET", "games")
    extra = {"beam_search_results": "<a_1><b_2><c_3>"}
    # 采样的 SID 三元组若在 info 前缀树中合法, 附加格式分
    resp = "r </think>\n\n<a_1><b_2><c_3>"
    score = module.rule_base_reward(
        "rec/test", resp, GT, extra,
        reward_type="onerec_rollout_beam", format_reward_weight=0.1,
    )
    assert score >= 3.0  # beam 3 分 + 可能的 0.1 格式分
    assert score in (3.0, 3.1)


# --------------------------------------------------------------------------- #
# beam 条件截断(SIDReasoner verl fork 侧, 缺失该仓库则跳过)
# --------------------------------------------------------------------------- #


def test_truncate_after_think_for_beam():
    verl_root = Path(os.environ.get("VERL_HOME", R4GR_ROOT.parent / "SIDReasoner"))
    rollout_file = verl_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout_spmd.py"
    if not rollout_file.exists():
        pytest.skip(f"未找到 SIDReasoner verl fork: {verl_root}")

    needed = {
        "ray": None, "zmq": None, "filelock": None, "omegaconf": None,
        "tensordict": None, "vllm": None, "vllm.sampling_params": None,
        "vllm.distributed": None, "vllm.distributed.parallel_state": None,
        "vllm.lora": None, "vllm.lora.request": None, "vllm.model_executor": None,
        "vllm.model_executor.sampling_metadata": None, "vllm.worker": None,
        "vllm.worker.worker_base": None, "torch": None, "torch.distributed": None,
        "verl": None, "verl.utils": None, "verl.utils.reward_score": None,
        "verl.workers": None, "verl.workers.rollout": None,
        "verl.workers.rollout.base": None, "verl.utils.profiler": None,
        "verl.utils.torch_functional": None, "verl.single_controller": None,
    }
    for name in needed:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            mod.__getattr__ = lambda attr, _n=name: types.SimpleNamespace()
            sys.modules[name] = mod

    sys.modules["verl"].__path__ = [str(verl_root / "verl")]
    sys.modules["verl"].DataProto = object
    sys.modules["verl.utils"].__path__ = [str(verl_root / "verl" / "utils")]
    sys.modules["verl.utils.reward_score"].__path__ = [str(verl_root / "verl" / "utils" / "reward_score")]
    sys.modules["vllm"].LLM = object
    sys.modules["vllm.sampling_params"].SamplingParams = type("SamplingParams", (), {"__init__": lambda self, **kw: None})
    sys.modules["vllm.sampling_params"].BeamSearchParams = type("BeamSearchParams", (), {"__init__": lambda self, **kw: None})
    sys.modules["vllm.distributed"].parallel_state = sys.modules["vllm.distributed.parallel_state"]
    sys.modules["vllm.lora.request"].LoRARequest = object
    sys.modules["vllm.model_executor.sampling_metadata"].SamplingMetadata = object
    sys.modules["vllm.worker.worker_base"].WorkerWrapperBase = object
    sys.modules["torch"].no_grad = lambda: (lambda f: f)
    sys.modules["verl.workers.rollout.base"].BaseRollout = object
    sys.modules["verl.utils.profiler"].GPUMemoryLogger = lambda **kw: (lambda f: f)
    sys.modules["verl.utils.torch_functional"].get_response_mask = None
    sys.modules["verl.utils.torch_functional"].pad_2d_list_to_length = None

    spec = importlib.util.spec_from_file_location("vllm_rollout_spmd_test", rollout_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"无法在当前环境导入 vllm_rollout_spmd: {e}")

    truncate = module.truncate_after_think_for_beam
    marker = [151668]  # Qwen3 的 </think>
    newlines = {198, 271}  # \n 与 \n\n

    # </think>\n\n 后接 SID: 截到 SID 之前
    out, found = truncate([1, 2, 151668, 271, 101, 102, 103, 0], marker, newlines)
    assert found and out == [1, 2, 151668, 271]

    # 单个换行也成立
    out, found = truncate([1, 151668, 198, 101, 102], marker, newlines)
    assert found and out == [1, 151668, 198]

    # </think> 后无换行
    out, found = truncate([1, 151668, 101], marker, newlines)
    assert found and out == [1, 151668]

    # 推理未闭合: 原样返回
    out, found = truncate([1, 2, 3], marker, newlines)
    assert not found and out == [1, 2, 3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
