# Reasoning4GenRec 复现

调参直接改 `config/` 下与模型同名的配置文件, 命令行不接受超参;
数据集通过目标名选择(`games` / `office` / `industrial`)。
模型同名文件在 `src/models/`(模型)、`src/train/`(训练入口)、`scripts/`(启动脚本)中一一对应。

# 1. 环境

```bash
# faster
source /thuir/wangyiyao/miniconda3/etc/profile.d/conda.sh
conda create -p /thuir/wangyiyao/conda_envs/r4gr python=3.11
conda activate /thuir/wangyiyao/conda_envs/r4gr

# 112.111.7.80
source /data/jiangchumeng/miniconda3/etc/profile.d/conda.sh
conda create -p /data/jiangchumeng/conda_envs/r4gr
conda activate /data/jiangchumeng/conda_envs/r4gr

python3 -m venv .venv && source .venv/bin/activate
make setup
```

# 2. 基线模型

```bash
make games-sasrec          # SASRec
make games-sasrec-mini     # SASRec(MiniOneRec 实现)
make games-tiger           # TIGER
make games-caser           # Caser(WSDM'18)
make games-gru4rec         # GRU4Rec(RecSys'18)
make smoke                 # 冒烟测试
```

office / industrial 把 `games` 换掉即可。

# 3. SIDReasoner(KDD'26)

论文 Table 2 = 3 阶段后, 以 Stage 3 模型做思考模式评估:

```bash
# 0) 一次性准备
make download-model        # Qwen3-1.7B; 下载后修改 config.sidreasoner 中的 model 为模型的路径
make install-vllm          # 思考模式评估需要

# 1) Stage 1 & 2
make games-sidreasoner-sft             # Stage 1: 增强式 SID-语言对齐 SFT
make games-sidreasoner-activation      # Stage 2: 推理激活 SFT
make games-sidreasoner-sft-gem
make games-sidreasoner-activation-gem  # 默认接 Stage 1 GEM 的 final_checkpoint

# 2) Stage 3
make games-sidreasoner-rl              # Stage 3: GRPO 强化学习(前置见下)
make games-sidreasoner-rl-beam         # Stage 3 变体: OneRec-Think Rollout-Beam reward(见下)
make games-sidreasoner-rl-beam-div     # Stage 3 变体: Rollout-Beam x OneReason R_div(见下)

# 挂 tmux 跑:
make tmux TARGET=games-sidreasoner-sft SESSION=sft-games
make tmux TARGET=games-sidreasoner-sft-gem SESSION=sft-gem-games
make tmux TARGET=games-sidreasoner-activation-gem SESSION=act-gem-games

make tmux TARGET=games-sidreasoner-rl-beam SESSION=rl-beam-games
make tmux TARGET=games-sidreasoner-rl-beam-div SESSION=rl-beam-div-games

tmux ls                                # 1. 列出服务器上仍在跑的会话
tmux attach -t sft                     # 2. 接回对应会话(Ctrl-b d 只退出查看, 不停任务)
tmux kill-session -t sft               # 3. 清理会话

# 3) 合并 Stage 3 的 FSDP 分片 checkpoint
make games-sidreasoner-merge
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam_div

# 4) 思考模式评估 & 计算指标 & 画图
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
make games-sidreasoner-metrics         # 自动取最新 think 结果, 可用 RESULT=xxx.json 指定
make games-sidreasoner-curves          # 默认画全部: NDCG@5/10, HR@5/10
make games-sidreasoner-curves ARGS="--metric NDCG@10"

# 非思考模式: MODEL 取 stage1 / stage2 / 任意 checkpoint 路径
make games-sidreasoner-eval MODEL=stage2
```

## SFT 变体: GEM 损失(Stage 1/2, 可选)

把 Stage 1/2 SFT 的 CE 损失换成 [GEM](https://arxiv.org/abs/2408.16673)(ICLR 2025, [官方代码](https://github.com/liziniu/GEM)): 以 `q = softmax(logits/β)` 为固定目标分布的熵正则逆 KL 损失, 精度接近 CE 的同时保留输出多样性(β→1 退化为 CE, β 越小越保多样; 更多样的冷启动模型对后续 Stage 3 RL 探索有利, 见 GEM 论文)。实现移植自官方 `sft_trainer_v2.py`, **评估/早停仍用 CE eval_loss**(官方行为), 训练日志附带 `ce_loss` 与 `entropy` 诊断量, 产物目录自动加 `_gem` 后缀与基线互不覆盖:

```bash
# 双卡(2x A800-80G): batch/lr 等沿用基线配置, 仅换损失, 梯度累积自动适配卡数
make games-sidreasoner-sft-gem           # Stage 1 GEM -> runs/sidreasoner/Video_Games_stage1_sft_gem
make games-sidreasoner-activation-gem    # Stage 2 GEM(默认接 Stage 1 GEM 的 final_checkpoint)
# 等价手动方式: DATASET=games bash scripts/sidreasoner/sft_gem.sh
# 挂 tmux:     make tmux TARGET=games-sidreasoner-sft-gem SESSION=sft-gem
```

- 超参: `config/sidreasoner.py` 的 `training.sft.gem` / `training.activation.gem`(`beta=0.7`、`h="logsigmoid"` = 官方论文/脚本默认)
- 提速(2x A800-80G): gem 段的 `micro_batch_size`(Stage 1 为 4, Stage 2 为 8)仅对 GEM 运行生效(CE 基线仍为 1/2), 梯度累积自动补偿, 全局 batch 仍 1024、训练等价。注意: accelerate 的 bf16 混合精度会把 forward 输出 logits 转成 fp32(gem_loss 内部已转回 bf16、ce_loss 的 fp32 拷贝已提前释放, 见 `sidreasoner_gem.py`), 全 3072 批次峰值仍约 55-60GB, micro=8 必 OOM; `gradient_checkpointing` 必须保持 True
- 显式配置过 `training.sft.out` 时, Stage 2 GEM 需手动把 `training.activation.base_model` 指到 GEM Stage 1 checkpoint
- 监控: `loss`(GEM)与 `ce_loss` 应同步下降; `entropy` 下降慢于 CE 基线即多样性保留生效; `print_entropy` 有百分之几开销, 极限提速可关
- 评测: `make games-sidreasoner-think MODEL=runs/sidreasoner/Video_Games_stage1_sft_gem/final_checkpoint` -> 结果自动路由到 `results/sidreasoner/Games/sft_gem/stage1.json`(原始 SFT -> `sft/stage1.json`, 见下节路由表)

## 评测结果目录结构(两级路由)

所有评测结果按 **数据集 x 评测类型** 自动路由到
`results/sidreasoner/<Dataset>/<variant>/` 下, 文件名只含 step 数
(从 MODEL 路径的 `global_step_N` 提取; stage1/stage2 checkpoint 用
`stage1`/`stage2`)+ 可选的 `_own`(自建 SID)/`_seed<S>`/`_summary` 后缀:

```
results/sidreasoner/
  Games/  Office/  Industrial/          <- 第一层: 数据集
    nonthink/               step_100.json            # 非思考模式(eval.sh)
    le3/                    eval_stage2.json         # 短历史(<=3)子集,
                            think_stage2.json       #   eval/think 双模式对照
    sft/                    stage1.json              # Stage 1/2 SFT(原始 CE)思考模式
                            stage2.json
    sft_gem/                stage1.json              # Stage 1/2 SFT(GEM 变体)思考模式
                            stage2.json
    think_steprule/         step_100.json            # 原始 Stage3(StepRule)思考模式
    think_rolloutbeam/      step_900.json            # Rollout-Beam reward 思考模式
    think_rolloutbeam_div/  step_1500.json           # + OneReason R_div 思考模式
    think_steprule_temp/    step_100_seed42.json     # 原始思考模式+采样温度(>0)
                            step_100_summary.json    #   多 seed 汇总
```

- **变体自动推断**: 从 MODEL 路径的实验名后缀识别。Stage 1/2 训练目录
  (`*_stage1_sft` / `*_stage2_activation`)归 `sft`, 带 `_gem` 后缀归
  `sft_gem`; Stage 3 按 `_rolloutbeam` / `_rolloutbeam_div` / 无后缀
  =steprule; 温度>0 时自动落到 `<变体>_temp`(SFT 系为 `sft_temp` /
  `sft_gem_temp`)。例: `make games-sidreasoner-think
  MODEL=runs/sidreasoner/Video_Games_stage1_sft_gem/final_checkpoint`
  -> `Games/sft_gem/stage1.json`。
- **存量文件一键归档**: `make sidreasoner-organize`(先 `DRY_RUN=1`
  预览); 旧 think 文件默认归 `think_steprule`, 确知来自 beam 模型的
  请手动挪到对应目录。
- **Stage 3 RL 训练曲线**: 按 step 对比三个变体的思考模式指标, 输出到
  `<Dataset>/figures/<指标>.png`(贪心结果现场重算指标, 采样 summary 取
  mean 并画 std 置信带; 缺 step 的折线自然截断; 叠加官方论文最终结果的
  金色水平参考线, `--no-baseline` 可关):

  ```bash
  .venv/bin/pip install matplotlib          # 一次性(服务器 venv 默认没有)
  make games-sidreasoner-curves             # 默认画全部: NDCG@5/10, HR@5/10
  make games-sidreasoner-curves ARGS="--metric NDCG@10"   # 只画一张
  ```
  新增折线: 在 `src/eval/sidreasoner_rl_curves.py` 的 `DEFAULT_SERIES`
  里加一行(label + 结果子目录), 或调用 `plot_rl_metric(series=...)` 传入。

## 思考模式评估: 采样解码 + 多 seed 指标(默认)

`config/sidreasoner.py` 的 `evaluation` 段控制 CoT 解码方式:

| 配置项 | 默认 | 说明 |
|---|---|---|
| `think_temperature` | `1.5` | CoT 采样温度(OneRec-Think 官方 Stage 评测 `eval_parallel_8gpu_cot.sh` 的 `--think_temperature 1.5 --think_top_p 0.95`); **`0.0` 恢复原贪心解码** |
| `think_top_p` | `0.95` | CoT 采样 top-p(温度>0 时生效) |
| `eval_seeds` | `[42,43,44,45,46]` | 固定 5 个随机种子, 逐 seed 评测(vLLM per-request seed, 可复现) |
| `ci_level` | `0.95` | 置信区间水平 |
| `save_per_seed_results` | `False` | 多 seed 评测是否额外写单 seed 明细文件(默认只写 summary) |

温度>0 时: SID beam 阶段仍为确定性约束束搜索, 随机性仅来自 CoT 采样;
**默认只输出** `results/sidreasoner/<C>_think_<模型>_summary.json`
(每个 NDCG@K / HR@K 的均值、样本标准差、95% 置信区间[t 分布,
n=5 时 t=2.776], 并含逐 seed 明细), 终端同步打印; 不再写单 seed
明细文件(需要其 `predict`/`cot` 字段做多样性分析等导入时, 把
`evaluation.save_per_seed_results` 置 `true`)。

```bash
make games-sidreasoner-metrics   # summary 生成后自动打印聚合表

# 也可手动对多份单 seed 结果聚合(如分开跑/分开传文件):
DATASET=games .venv/bin/python -m src.eval.sidreasoner_metrics aggregate \
    --path results/sidreasoner/Games/think_steprule_temp/step_100_seed42.json,...,seed8888.json
```

## 把玩 (终端对话)

```bash
make games-sidreasoner-chat MODEL=runs/sidreasoner/Video_Games_stage1_sft/final_checkpoint
```

会话内 `/help` 查看全部命令

不想从头训练: 下载[官方 checkpoint](https://huggingface.co/Sober-Clever/SIDReasoner-Models)
后跳过步骤 1)-2), 直接跑 think + metrics。

## Stage 3 前置

```bash
git clone --depth 1 https://github.com/HappyPointer/SIDReasoner ../SIDReasoner   # verl fork
make tmux TARGET=install-rl-env USE_MEGATRON=0 USE_SGLANG=0   # 跳过用不到的 Megatron/sglang
```

## Stage 3 奖励变体: OneRec-Think Rollout-Beam(可选)

默认 Stage 3 用论文原版 StepRule 奖励(对 rollout 采样出的那一个 SID 给
0.25/0.5/1.0 步进分 + 0.1 格式分)。本变体换成 OneRec-Think
(arXiv:2510.11639, Eq. 6-7)的 **Rollout-Beam 奖励**: 每条 rollout 的
思维链 CoT 之后, 由 vLLM rollout worker 以宽度 K 再做一次 SID beam
search, 取 K 个候选 SID 中与 ground truth 匹配 code 数(共 L 层)最多者
作为该 CoT 的奖励:

```text
R_Rollout-Beam = max_{ŝ∈B(τ)} Σ_{l=1..L} 1[ŝ^l = s_gt^l]
```

直观理解: 不按"模型这次恰好采样出的 SID"给分, 而是按"模型沿这条思维链
最多能推荐到多准"给分。采样 SID 绝大多数全错导致 GRPO 组内优势恒为 0
的问题可被缓解, 同时奖励与 beam-search 推理保持一致。

### 前置(一次性): 同步修改版 verl fork

beam search 基建位于 SIDReasoner verl fork 内部(rollout worker 截断
`</think>` 后跑 beam、ray_trainer 把 K 个候选注入 `extra_info`、
`rollout.yaml` 的 `sid_beam_size/sid_length` 配置项), r4gr 只负责入口与
奖励实现。请把本地**修改版 SIDReasoner** 同步到服务器, 替换
`../SIDReasoner`(或按上文方式用 `VERL_HOME` 指向它), 并校验:

```bash
grep -n "sid_beam_size" ../SIDReasoner/verl/trainer/config/rollout/rollout.yaml
grep -n "truncate_after_think_for_beam" \
    ../SIDReasoner/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
grep -n "inject_beam_search_results" ../SIDReasoner/verl/trainer/ppo/ray_trainer.py
```

三条命令都有输出即就绪(普通 Stage 3 不受这些改动影响)。

### 训练 / 合并 / 评估(终端操作)

```bash
# Stage 3(Rollout-Beam): 从同一个 Stage 2 checkpoint 出发, 其余流程与默认一致
make games-sidreasoner-rl-beam
# 长任务同样建议挂 tmux(断开 SSH 不中断):
make tmux TARGET=games-sidreasoner-rl-beam SESSION=rlbeam

# 等价手动方式(docker 镜像内同理, 前缀 PY=python3):
DATASET=games bash scripts/sidreasoner/rl_beam.sh

# 合并 FSDP 分片 checkpoint(默认目录自动加 _rolloutbeam 后缀):
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam

# 思考模式评估(实验目录名多了 _rolloutbeam):
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam/global_step_100/actor_merged

# 评估结果按两级目录自动路由到 Games/think_rolloutbeam/step_100.json,
# 与 steprule / beam-div 的结果天然隔离, 直接算指标即可:
make games-sidreasoner-metrics \
    RESULT=results/sidreasoner/Games/think_rolloutbeam/step_100.json
```

A/B 对照: `make games-sidreasoner-rl`(StepRule)与
`make games-sidreasoner-rl-beam`(Rollout-Beam)产物按后缀隔离
(wandb experiment、checkpoint、merge 目录均带 `_rolloutbeam`; 自建
SID 时叠加为 `_own_rolloutbeam`), 两套可并存直接对比。office /
industrial 把 `games` 换掉即可。

### 奖励参数(统一改 config/sidreasoner.py 的 rl 段)

| 配置项 | 默认 | 说明 |
|---|---|---|
| `reward` | `"steprule"` | `steprule`(论文原版)/ `rollout_beam`; `rl.sh` 据此选择 reward 文件(不改则默认行为不变) |
| `sid_beam_size` | `4` | beam 宽度 K; 必须 >1, 否则 rl.sh 直接报错(防止奖励恒为 0) |
| `sid_length` | `3` | SID 层数 L(`<a_*><b_*><c_*>` 为 3) |
| `reward_kwargs.reward_type` | `"onerec_rollout_beam"` | 注册表中的 reward 实现 |
| `reward_kwargs.match_mode` | `"position"` | `position`=论文式逐层比对; `prefix`=前缀连续匹配(StepRule 式对照实验) |
| `reward_kwargs.normalize` | `False` | 匹配数除以 L, 奖励归一到 [0,1] |
| `reward_kwargs.format_reward_weight` | `0.0` | 采样 SID 合法性的附加格式分(论文无此项; >0 时按 DATASET 自动找 info 文件) |

说明: `make games-sidreasoner-rl-beam` 用环境变量强制
`reward=rollout_beam`; 若直接把配置改成 `"reward": "rollout_beam"`,
普通 `make games-sidreasoner-rl` 也会走 Rollout-Beam。`make
games-sidreasoner-rl` 的默认(配置默认值)仍是 StepRule, 原有行为不变。

### 数据流与开销

```text
rollout 采样(CoT + 采样 SID)
  -> rollout worker 截断到 </think>(含其后换行)
  -> vLLM beam search(宽度 K, 深度仅 L=3 个 token)
  -> non_tensor_batch["beam_search_results"]
  -> ray_trainer 注入每条 rollout 的 extra_info
  -> src/reward/direct_recommendation_rolloutbeam.py 按 Eq.6-7 打分
```

每条 rollout 仅多一次 K 宽 × L token 的增量生成, 相对 1024 token 的
rollout 生成开销可忽略。

### 新增自定义奖励变体(为后续迭代准备)

奖励计算与 trainer 解耦在 `src/reward/sid_rewards/`(纯 Python):

```python
# src/reward/sid_rewards/my_reward.py
from .base import BaseSidReward, parse_beam_candidates
from .registry import register_sid_reward

@register_sid_reward("my_reward")
class MyReward(BaseSidReward):
    def compute(self, data_source, solution_str, ground_truth, extra_info=None):
        gt = self.ground_truth_sids(ground_truth)      # ['<a_1>', '<b_2>', '<c_3>']
        beams = parse_beam_candidates(                 # 开了 beam search 即有 K 个候选
            (extra_info or {}).get("beam_search_results"), num_sid=self.num_sid_levels)
        ...
```

在 `src/reward/sid_rewards/__init__.py` 里 `from . import my_reward` 注册,
保持 `rl.reward` 为 `rollout_beam` 或 `rollout_beam_div`(开 beam 基建并决定
输出后缀), 再把 `config/sidreasoner.py` 的 `rl.reward_kwargs` 里显式加上
`"reward_type": "my_reward"`(显式值优先于 rl.sh 的分支默认)即可,
无需改 rl.sh / trainer。单测:

```bash
python -m pytest tests/test_sid_rewards.py -v   # 纯 Python, 无需 GPU/verl
# venv 里没有 pytest 时: .venv/bin/pip install pytest, 再用 .venv/bin/python -m pytest ...
```

## Stage 3 奖励变体二: Rollout-Beam x OneReason R_div(可选)

在 Rollout-Beam 奖励基础上再乘 OneReason(arXiv:2606.06260, Eq.9)的
**多样性因子 R_div**, 鼓励"一条思维链能带出多个类别的候选":

```text
R_div(CoT) = max(0, m^(1) - 1) / (K - 1)
R(CoT)     = R_RolloutBeam(CoT) x R_div(CoT)
```

其中 K = beam 宽度(该 CoT 的 K 个候选 SID), m^(1) = K 个候选**第 1 层
code**(如 `<a_*>`)去重后的个数。论文动机: 首层 code 粗粒度决定类别
("the first sub-token largely determines the subsequent decoding process
and captures coarse-grained item categories")。R_div 在思维链级别定义
(同一 CoT 的 K 个候选共享同一因子), 乘法组合与论文 Eq.7 一致; 差异在于
论文的基础奖励是 0/1 命中指示、K 条候选为并发生成, 我们按实验设计保留
beam-max 匹配数作为基础奖励。例: K=3, 候选首层为 `<a_1><a_1><a_2>` ->
m=2, R_div=1/2。

### 终端操作(与 Rollout-Beam 完全同构, 只是 target/后缀不同)

```bash
# 训练(产物自动加 _rolloutbeam_div 后缀, 与 steprule/_rolloutbeam 互不覆盖)
make games-sidreasoner-rl-beam-div
make tmux TARGET=games-sidreasoner-rl-beam-div SESSION=rl-beam-div-games

# 合并 + 评估 + 指标
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam_div
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam_div/global_step_100/actor_merged
# 结果自动路由到 Games/think_rolloutbeam_div/step_100.json:
make games-sidreasoner-metrics \
    RESULT=results/sidreasoner/Games/think_rolloutbeam_div/step_100.json

# 等价手动方式: DATASET=games bash scripts/sidreasoner/rl_beam_div.sh
#            或 REWARD=rollout_beam_div DATASET=games bash scripts/sidreasoner/rl.sh
```

前置与 Rollout-Beam 相同(修改版 verl fork + `rl.sid_beam_size > 1`)。
`rl.reward` 的三个取值 `steprule / rollout_beam / rollout_beam_div` 与三个
make target 一一对应; 额外参数只有 `rl.reward_kwargs.div_level`(默认 1 =
首层, 论文设定; 可改 2/3 做层位消融), 其余参数(match_mode/normalize/
format_reward_weight)与 Rollout-Beam 共享。

### 注意事项(R_div)

- **beam 集中可能压低奖励**: beam search 天然倾向集中, 模型较确定时 K 个
  候选首层可能全同(m=1 -> R_div=0), 乘法会把该 CoT 奖励压成 0, 甚至组内
  全零无梯度。对策: 增大 `rl.sid_beam_size`(如 8/16); 用
  `critic/rewards/mean` 与 rollout_beam 的均值对比定位是命中率低还是多样
  性低。
- **三方 A/B**: `rl` / `rl-beam` / `rl-beam-div` 从同一 Stage 2 出发,
  checkpoint、think 结果按后缀(无 / `_rolloutbeam` / `_rolloutbeam_div`)
  完全隔离, 可并存对比。

# 4. 自建 SID(独立训练 RQ-VAE)

## 构建流程

```
item.json (title/brand/... 原始字段)
   │  ① embed    text_fields 拼接 -> BGE(bge-base-en-v1.5) 编码
   │             CLS pooling + L2 归一化, 缓存于 runs/rqvae/<C>/embeddings.npy
   ▼
embeddings.npy (N x 768)
   │  ② train    MLP 编码器 -> 3 层残差 VQ(各 256 code)
   │             EMA + 逐层 k-means 初始化 + 死码重播种
   │             loss = 重建 MSE + commitment; 量化全部 item
   ▼
index.own.json + report.json + checkpoint.pt   (仍在 runs/rqvae/<C>/)
   │  ③ apply    官方原件一次性备份 -> 级联重写 data/raw/Amazon 下
   │             4 处 SID 文件(见下节路由表) -> load_bundle 验证
   ▼
data/raw/Amazon 下的 SID 全部替换为本仓库自建版本
```

```bash
make games-rqvae            # 三步连跑(office / industrial 同名替换)
make games-rqvae-embed      # ① 约 10s(模型已缓存) / 首次约 2.5min
make games-rqvae-train      # ② 约 4min(MPS/CPU 均可)
make games-rqvae-apply      # ③ 备份 + 替换 + 验证
```

- 超参(编码器 / text_fields / 码本结构 / 训练)全部在 `config/rqvae.py`。
- 重跑: 改配置后 embed 有缓存校验(FORCE=1 强制重算); train 每次全量重训。
- 质量报告看 `runs/rqvae/<C>/report.json`(码本利用率 / 碰撞 / 与官方重叠率)。

当前成品(见各 `report.json`):

| 数据集 | items | 唯一 SID | 碰撞(自建/官方) | 码本利用 | 重建 cos |
|---|---|---|---|---|---|
| Video_Games | 3858 | 3643 | 215 / 31 | 256/256/247 | 0.951 |
| Office_Products | 3459 | 3199 | 260 / 15 | 256/256/254 | 0.948 |
| Industrial_and_Scientific | 3686 | 3291 | 395 / 16 | 256/254/255 | 0.939 |

与官方 SID 的已知差异:

- **碰撞更多**(item 共享同一 SID): BGE 对近重复 title(如 Skylanders
  系列, title 仅差一两个词)聚得过紧, 量化后落入同一格。属文本表示的
  固有属性, 码本利用率本身已满格; 消费端(SidTable/trie)原生支持
  一 SID 多 item。实验依据: description 模板文案会把 title 信号淹没,
  全字段编码碰撞 797, 仅 title+brand 降到 215。
- **RL parquet 历史更短**: 重建版基于 CSV(截断 10), 官方版均值 ~15.6,
  属上游已知差异, 与 SID 本身无关。
- `item_enhanced_v2.json` / `integrated_narrative.csv` / `general/`
  不含 SID 字符串, 两套 SID 通用; token 集合恒为 768 个
  (`<a_0>..<c_255>`), tokenizer / trie / 评估代码零改动。

# 5. SID 数据流与切换(官方 vs 自建)

## 路由表: SID 从哪 4 处文件流向哪些程序

| SID 载体 | 消费程序(读取点) |
|---|---|
| `index/<C>.index.json`<br>(item -> SID 表) | `src/data/sid.py::SidTable` <- `bundle.py::load_bundle`(TIGER 训练 / smoke 测试); `sidreasoner_sft.py` / `sidreasoner_activation.py` 的 paths -> `sidreasoner_data.py` 各 Dataset(SID-语言对齐); `models/sidreasoner.py::TokenExtender`(收集 768 token 扩 Qwen3 词表); `eval/sidreasoner_eval_think.py` |
| `{train,valid,test}/<C>_5_*.csv`<br>(`history_item_sid` / `item_sid` 两列) | TIGER(经 load_bundle); Stage 1 SFT(`sidreasoner_data.py`); RL parquet 构建(`sidreasoner_rl_data.py`); 非思考评估 `eval/sidreasoner_eval.py` + `scripts/sidreasoner/eval.sh` 的 TEST_FILE; 思考评估 `eval/sidreasoner_eval_think.py`; MiniOneRec(`sasrec_mini.py`) |
| `info/<C>_5_*.txt`<br>(SID\ttitle\titem_id) | 非思考评估的前缀 trie(`sidreasoner_eval.py`); 思考评估(`sidreasoner_eval_think.py`); **Stage 3 RL 奖励的合法 SID 表**(`reward/direct_recommendation_steprule.py`; Rollout-Beam 变体为 `reward/direct_recommendation_rolloutbeam.py`, 仅格式分开启时用); 指标计算(`sidreasoner_metrics.py`); MiniOneRec |
| `rec_reasoning_verl/<C>/*.parquet`<br>(prompt 内嵌 SID + ground_truth) | Stage 3 GRPO(`scripts/sidreasoner/rl.sh` 的 `data.train_files/val_files` -> verl) |

不受 SID 影响: SASRec / Caser / GRU4Rec(只用 item_id 列), 以及
`item.json` / `item_enhanced_v2.json` / `integrated_narrative.csv` /
`general/sampled_data.arrow`(均不含 SID 字符串)。

## 切换方法

SID 只通过上述 4 处文件进入下游, 代码与词表零耦合, 因此 A/B 对比
= 整组文件互换:

```bash
DATASET=games SID=official make games-sid-switch   # 切到官方 SID
DATASET=games SID=own     make games-sid-switch   # 切回自建 SID
```

- 两套完整快照: `data/raw_official_backup/<C>/`(官方原件) 与
  `data/raw_own_backup/<C>/`(自建, 切换工具自动建档并保持最新)。
- 切换工具按 index.json 内容识别当前生效集, 切换前会把 in-place
  文件存回所属快照, 保证往返无损, 切完自动 `load_bundle` 验证。
- **输出自动隔离**: 训练/评估的输出目录与文件名按当前 SID 集自动
  加后缀(official 无后缀, 与历史路径完全兼容; own -> `_own`),
  两套 SID 的产物互不覆盖:

  | 环节 | official | own |
  |---|---|---|
  | Stage 1/2 | `runs/sidreasoner/<C>_stage{1,2}_*/` | `runs/sidreasoner/<C>_stage{1,2}_*_own/` |
  | Stage 3 | `checkpoints/RecRL_Reasoning/<C>_stage3_rl_Qwen3-1.7B/` | `..._Qwen3-1.7B_own/` |
  | 非思考评估 | `results/sidreasoner/<D>/nonthink/step_N.json` | `step_N_own.json`(同目录) |
  | 思考评估 | `results/sidreasoner/<D>/think_*/step_N.json` | `step_N_own.json`(同目录) |

  (D 为 Games/Office/Industrial; 自建 SID 只体现在文件名 `_own` 后缀,
  目录结构不变。)

  Stage 2 的 base_model、Stage 3 的初始策略、评估的 stage1/stage2
  简写路径均按同一规则联动, 无需手工干预。
- **SID_SET 环境变量**: 快照不齐无法自动判定时(如新机器), 用
  `SID_SET=own|official make ...` 显式声明, 对训练/评估/merge 均生效。
- **checkpoint 与 SID 绑定**: 切换 SID 后, 旧 SID 的 checkpoint 不可
  直接评估, 须按当前 SID 重新训练再评估(上表的后缀机制保证两边
  产物共存, 直接对照)。

# 6. 思维链多样性分析(SIDReasoner)

观察"不同用户的思维链高度雷同"这一多样性问题。分析命令**单文件输入**:
一份思考模式评测结果 JSON(逐条记录、含 `cot` 字段), 结果**镜像路由**到
`results/diversity/` 下的同名子路径, 只产出一个分析 JSON:

```
输入  results/sidreasoner/Games/think_steprule/step_1900.json
输出  results/diversity/Games/think_steprule/step_1900.json   # {"text","raw","meta"}
```

```bash
# 直接运行
.venv/bin/python -m src.analysis.diversity_analysis \
    results/sidreasoner/Games/think_steprule/step_1900.json
# 或经 make(等价, 环境变量传参)
INPUT=results/sidreasoner/Games/think_steprule/step_1900.json \
    PAIR_CAP=2000 DEVICE=cuda make games-diversity-analyze
```

- 统计内容(每份输出含 text/raw 双视角, text=剔除 SID 只看自然语言,
  raw=保留 SID): TF-IDF 词频与 bigram 频率 Top、TTR/distinct-1,2/熵
  (含停用词 `lexical` 与去停用词 `lexical_content` 双口径)、完全重复率、
  成对 TF-IDF 余弦与 3-gram Jaccard(亦有去停用词版)、首句模板集中度
  (`openers` SID 泛化 / `openers_strict` 原样双口径)、gzip 压缩率、
  最相似样本对 Top-3; `meta` 记录输入路径/文档数/闭合率/统计参数。
- 输入要求: 逐条记录的 think 结果(贪心单文件或多 seed 的单 seed 明细,
  后者需置 `evaluation.save_per_seed_results: true`); `_summary.json`
  是跨 seed 聚合、无逐条 cot, 不能作为输入(脚本会明确报错)。
- 性能: 全量 6142 条秒级; 成对指标受 `PAIR_CAP` 限制(默认 600, 0=不限),
  GPU(cuda/mps)自动走 torch 矩阵乘, `DEVICE=cuda` 强制。
- 无中间文件: 不再落盘 corpora/markdown; 旧版三语料对比流程
  (teacher/stage2/stage3 采集 + report.md)已移除。
- 早期结论存档(games, 300 条贪心采样): stage2 相对 teacher TTR 减半
  (0.024 vs 0.045)、成对余弦均值 +50%(0.285 vs 0.189)、首句模板 Top5
  覆盖 4 倍; stage3(GRPO 后)词汇多样性持平但首句模板进一步固化
  (Top1 5.3% vs 1.3%)——多样性坍缩发生在 Stage 2 SFT 且延续到 RL。
