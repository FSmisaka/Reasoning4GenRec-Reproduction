"""SIDReasoner 评测结果的两级目录路由。

results/sidreasoner/ 下的产物统一组织为::

    results/sidreasoner/<Dataset>/<variant>/<文件>.json

第一层 <Dataset>: Games / Office / Industrial(按数据集)。

第二层 <variant>(评测类型 x 训练变体):
    nonthink                非思考模式评测
    le3                     短历史(history <= 3)子集评测(eval/think 双模式对照)
    sft                     Stage 1/2 SFT(原始 CE)思考模式评测
    sft_gem                 Stage 1/2 SFT(GEM 损失变体)思考模式评测
    think_steprule          原始 SIDReasoner(StepRule reward)思考模式
    think_rolloutbeam       Rollout-Beam reward(OneRec-Think)思考模式
    think_rolloutbeam_div   Rollout-Beam x OneReason R_div 思考模式
    think_steprule_temp     原始 SIDReasoner 思考模式 + 采样温度(temperature > 0)

    泛化规则(按需自动创建, 不在固定清单内):
    - 思考模式 + temperature > 0 -> <variant>_temp(sft 系同样: sft_temp /
      sft_gem_temp)
    - 短历史阈值非 3 时 -> le<N>

<文件> 不再含数据集名(上层目录已有), 必含 step 数(从模型路径中的
global_step_N 提取; stage1/stage2 checkpoint 用 stage1/stage2; 均无法
识别时退化为模型目录名), 其余按需附加: _own(自建 SID)、_seed<S>(多
seed 采样, 由 think 评测自行追加)、_summary(跨 seed 汇总, 同上)。
"""

import os
import re

# category -> 第一层目录名
DATASET_DIRS = {
    "Video_Games": "Games",
    "Office_Products": "Office",
    "Industrial_and_Scientific": "Industrial",
}

# 结果根目录(相对仓库根)
RESULT_ROOT = os.path.join("results", "sidreasoner")


def dataset_dir(category: str) -> str:
    d = DATASET_DIRS.get(category)
    if d is None:
        raise ValueError(
            f"未知 category: {category!r}, 可选: {sorted(DATASET_DIRS)}"
        )
    return d


def model_step_label(model) -> str:
    """从模型路径提取文件名主体。

    - 路径含 global_step_N(如 .../global_step_1500/actor_merged) -> "step_1500"
    - 路径含 stage1/stage2(runs/sidreasoner/<C>_stage{1,2}_*/final_checkpoint) -> "stage1"/"stage2"
    - 以上都不匹配 -> 模型目录名(如 actor_merged, 保留原样以便人工辨认)
    """
    if model is None:
        return "unknown"
    path = str(model)
    m = re.search(r"global_step_(\d+)", path)
    if m:
        return f"step_{m.group(1)}"
    m = re.search(r"stage([12])", path)
    if m:
        return f"stage{m.group(1)}"
    return os.path.basename(os.path.normpath(path))


def variant_of_model(model) -> str:
    """从模型路径(实验目录名)推断训练变体。

    Stage 1/2 SFT checkpoint(runs/sidreasoner/
    <C>_stage{1_sft,2_activation}[_own][_gem]/final_checkpoint)
    -> "sft" / "sft_gem"(GEM 损失变体);

    Stage 3 RL(checkpoints/RecRL_Reasoning/
    <C>_stage3_rl_Qwen3-1.7B[_own][_rolloutbeam[_div]])
    -> "steprule" / "rolloutbeam" / "rolloutbeam_div"。
    """
    path = str(model or "")
    if re.search(r"stage[12]_(sft|activation)", path):
        return "sft_gem" if "_gem" in path else "sft"
    if "_rolloutbeam_div" in path:
        return "rolloutbeam_div"
    if "_rolloutbeam" in path:
        return "rolloutbeam"
    return "steprule"


def result_dir(category, kind, model=None, temperature=0.0, max_len=None) -> str:
    """第二层目录名。

    kind: "nonthink"(非思考) | "think"(思考) | "short"(短历史子集)
    """
    if kind == "short":
        n = int(max_len) if max_len is not None else 3
        return f"le{n}"
    if kind == "nonthink":
        return "nonthink"
    if kind == "think":
        variant = variant_of_model(model)
        temp_sfx = "_temp" if temperature and float(temperature) > 0 else ""
        if variant in ("sft", "sft_gem"):
            # SFT 阶段思考模式评测: 不带 think_ 前缀(sft / sft_gem)
            return f"{variant}{temp_sfx}"
        return f"think_{variant}{temp_sfx}"
    raise ValueError(f"未知 kind: {kind!r}(可选 nonthink / think / short)")


def build_result_path(
    category,
    kind,
    model,
    sid_sfx: str = "",
    temperature: float = 0.0,
    max_len=None,
    mode_prefix: str = "",
) -> str:
    """构造完整结果路径(基础路径, 不含 _seed/_summary 后缀)。

    Args:
        category: 数据集 category(如 Video_Games)。
        kind: "nonthink" | "think" | "short"。
        model: 模型路径(用于推断 step 数与训练变体)。
        sid_sfx: SID 集后缀(官方为空, 自建为 "_own")。
        temperature: 思考模式采样温度(>0 路由到 *_temp 目录)。
        max_len: 短历史评测阈值(默认 3 -> le3)。
        mode_prefix: 短历史评测下区分双模式的前缀("eval" / "think")。
    """
    d = os.path.join(
        RESULT_ROOT,
        dataset_dir(category),
        result_dir(category, kind, model=model, temperature=temperature, max_len=max_len),
    )
    label = model_step_label(model)
    if sid_sfx and not sid_sfx.startswith("_"):
        sid_sfx = "_" + sid_sfx
    name = f"{mode_prefix}_{label}{sid_sfx}.json" if mode_prefix else f"{label}{sid_sfx}.json"
    return os.path.join(d, name)
