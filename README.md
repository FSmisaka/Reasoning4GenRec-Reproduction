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

# 1) 三阶段训练(4 卡)
make games-sidreasoner-sft             # Stage 1: 增强式 SID-语言对齐 SFT
make games-sidreasoner-activation      # Stage 2: 推理激活 SFT
make games-sidreasoner-rl              # Stage 3: GRPO 强化学习(前置见下)
make games-sidreasoner-rl-beam         # Stage 3 变体: OneRec-Think Rollout-Beam reward(见下)

# 挂 tmux 跑:
make tmux TARGET=games-sidreasoner-sft SESSION=sft
make tmux TARGET=games-sidreasoner-rl-beam SESSION=rl-beam-games
tmux ls                                # 1. 列出服务器上仍在跑的会话
tmux attach -t sft                     # 2. 接回对应会话(Ctrl-b d 只退出查看, 不停任务)
tmux kill-session -t sft               # 3. 清理会话

# 2) 合并 Stage 3 的 FSDP 分片 checkpoint
make games-sidreasoner-merge
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam

# 3) 思考模式评估 + 计算指标
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
make games-sidreasoner-metrics         # 自动取最新 *_think_*.json, 可用 RESULT=xxx.json 指定

# 非思考模式: MODEL 取 stage1 / stage2 / 任意 checkpoint 路径
make games-sidreasoner-eval MODEL=stage2
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

# 评估结果文件名由模型目录名生成, 与 steprule 的结果会同名互相覆盖;
# 出结果后先改名, 再分别算指标:
mv results/sidreasoner/Video_Games_think_actor_merged.json \
   results/sidreasoner/Video_Games_think_beam_step100.json
make games-sidreasoner-metrics \
    RESULT=results/sidreasoner/Video_Games_think_beam_step100.json
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
然后把 `config/sidreasoner.py` 的 `rl.reward_kwargs.reward_type` 改为
`my_reward` 即可, 无需改 rl.sh / trainer。单测:

```bash
python -m pytest tests/test_sid_rewards.py -v   # 纯 Python, 无需 GPU/verl
# venv 里没有 pytest 时: .venv/bin/pip install pytest, 再用 .venv/bin/python -m pytest ...
```

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
  | 非思考评估 | `results/sidreasoner/final_result_<C>.json` | `final_result_<C>_own.json` |
  | 思考评估 | `results/sidreasoner/<C>_think_*.json` | `<C>_think_*_own.json` |

  Stage 2 的 base_model、Stage 3 的初始策略、评估的 stage1/stage2
  简写路径均按同一规则联动, 无需手工干预。
- **SID_SET 环境变量**: 快照不齐无法自动判定时(如新机器), 用
  `SID_SET=own|official make ...` 显式声明, 对训练/评估/merge 均生效。
- **checkpoint 与 SID 绑定**: 切换 SID 后, 旧 SID 的 checkpoint 不可
  直接评估, 须按当前 SID 重新训练再评估(上表的后缀机制保证两边
  产物共存, 直接对照)。

# 6. 思维链多样性分析(SIDReasoner)

观察"不同用户的思维链高度雷同"这一多样性问题, 对三个对象做统计对比:
**teacher**(Stage 2 训练数据中教师模型的思维链)、**stage2** / **stage3**
(对应阶段模型在测试集上输出的思维链)。两个阶段, 产物均在
`results/diversity/<Category>/`(语料 `corpora/*.jsonl` + `analysis.json`
+ `report.md`):

```bash
# ① 采集语料(默认 teacher + 本地 stage2 checkpoint; MPS/CUDA 均可,
#    复刻 think 评测的 prompt, 贪心生成到 </think>, 无需 vllm)
make games-diversity-collect NUM=300
#    stage3 两种方式: 本地生成或导入服务器 think 结果(含全部三个对象)
MODEL_SPECS="stage3=<stage3 checkpoint 路径>" make games-diversity-collect
IMPORT_SPECS="stage3=results/sidreasoner/Video_Games_think_xxx.json" \
    FORCE=1 make games-diversity-collect

# ② 统计报告: TF-IDF 词频 / bigram 频率 / TTR 与 distinct-n / 熵 /
#    完全重复率 / 成对 TF-IDF 余弦与 3-gram Jaccard / 首句模板集中度 /
#    gzip 压缩率 / 最相似样本对示例; text 与 raw(含 SID)双视角;
#    词汇类指标(TTR/distinct/熵/词表)与 gzip/Jaccard 另有"去停用词"
#    对照版(*_content 键 / 报告第三张对比表)
make games-diversity-report
```

- 同一 `NUM`/`SEED` 下各模型的测试用户采样一致, stage2/stage3 可比;
  生成预算默认 512 token(`MAX_NEW_TOKENS=256` 对齐官方 think 评测)。
- 复跑换参数: `FORCE=1 NUM=500 make games-diversity-collect`。

## 服务器全量分析(复用已算好的 think JSON, 无需 GPU/vllm)

若服务器上已跑过思考评测, 直接导入其结果做全量统计(think JSON 的
`cot` 字段即思维链; teacher 从 CSV 取全部约 4.9 万行):

```bash
# ① 采集: 全量 teacher + 导入 stage2/stage3 的 think 结果
DATASET=games .venv/bin/python -m src.analysis.collect_diversity_corpora \
    --teacher --num -1 --force \
    --import stage2=results/sidreasoner/Video_Games_think_stage2.json \
    --import stage3=results/sidreasoner/Video_Games_think_stage3_step1900.json
# 或经 make: IMPORT_SPECS="stage2=results/sidreasoner/Video_Games_think_stage2.json,stage3=results/sidreasoner/Video_Games_think_stage3_step1900.json" \
#   TEACHER=1 NUM=-1 FORCE=1 make games-diversity-collect

# ② 全量统计报告(--full = 不截采样文档; 成对指标仍受 PAIR_CAP 限制,
#    全量语料每语料约 15s + 内存 ~1.6GB, PAIR_CAP=2000 时 Jaccard 约再 +1min)
DATASET=games .venv/bin/python -m src.analysis.diversity_report \
    --full --pair-cap 2000
# 或: FULL=1 PAIR_CAP=2000 make games-diversity-report
```

- **等量化(默认开)**: teacher 全量(4.9 万)与 think 结果(6142)规模不同,
  而 TTR/distinct/熵/gzip 等语料级指标随语料变大单调走低, 直接对比会
  失真。报告默认把所有语料按 seed 采样到与最小语料相同的篇数
  (各语料的采集总量/统计量分别记录在其小节与 `analysis.json` 的
  `meta.n_docs_collected/n_docs_analyzed`);`--no-equalize` 可关闭。
- 导入的语料 `user_id` 为 `#行号`(官方 think JSON 不含该字段)。
- 全量数值与采样 300 的数值不可直接对调着比(语料级指标依赖规模),
  对比请始终在同一份 report 内进行。

## 温度对照实验(检验"teacher 多样性是否来自采样温度")

采集脚本支持采样生成: `SAMPLING=default`(模型 generation_config 自带,
Qwen3 默认 T=0.6, top_p=0.95, top_k=20)或 `SAMPLING=1.0`(指定温度)。
对照语料命名加 `_temp` 后缀, 与贪心语料并存; 统计结果用 `--out`
写到独立文件, 不覆盖贪心版:

```bash
# ① 服务器上(A800, 全量测试集, bf16, 大 batch)生成采样版语料
#    注意 Stage 3 checkpoint 在 checkpoints/ 下(merge 后的 actor_merged),
#    不在 runs/ 下; step 数以实际训练为准(先 ls 确认)
DATASET=games SAMPLING=default NUM=-1 BATCH_SIZE=64 \
    FORCE=1 MODEL_SPECS="stage2_temp=runs/sidreasoner/Video_Games_stage2_activation/final_checkpoint,stage3_temp=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_1900/actor_merged" \
    make games-diversity-collect
# ② 同一套统计, 输出 analysis_temp.json / analysis_temp.md
#    (corpora 全部参与: teacher + 贪心 stage2/stage3 + 采样 *_temp, 等量化对齐)
DATASET=games FULL=1 PAIR_CAP=2000 DEVICE=cuda OUT=analysis_temp.json \
    make games-diversity-report
```

解读: 若 `*_temp` 的首句唯一率/成对余弦明显接近 teacher, 说明先前差距
有相当部分是"贪心 vs 采样"的解码混杂; 若仍与贪心版接近, 则多样性
坍缩是模型分布的属性, 与解码温度无关。

- 配套工具 `src/analysis/dup_decomposition.py`: 把首句重复分解为
  "同用户多余行 vs 跨用户"两种来源(行归属/配对归属双口径),
  用于检验"重复是否来自同一用户的级联测试行"。对贪心与采样语料
  各跑一遍即可对比温度的影响:
  `DATASET=games .venv/bin/python -m src.analysis.dup_decomposition --corpora stage2,stage3,stage2_temp,stage3_temp`
  (经 --import 导入的语料 user_id 是 #行号, 同用户口径恒为 0, 不适用)

- games 上的初步结论(300 条, seed 42, 三语料同测试用户): stage2 相对
  teacher, TTR 减半(0.024 vs 0.045)、distinct-2 减半(0.139 vs 0.296)、
  成对余弦均值 +50%(0.285 vs 0.189)、gzip 率更低、首句模板 Top5 覆盖
  4 倍(6.7% vs 1.7%), 且出现逐字相同的输出(完全重复率 0.7%)——
  SFT 后模型输出显著比教师数据更同质; stage3(GRPO 后)各项词汇多样性
  指标与 stage2 几乎持平(TTR 0.024 / distinct-2 0.143 / 余弦 0.279),
  但首句模板进一步固化(Top1 覆盖 5.3% vs 1.3%, Top5 12.3% vs 6.7%)——
  RL 未修复多样性坍缩, 反而加剧了开头句式的集中。
