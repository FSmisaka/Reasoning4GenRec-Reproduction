# SIDReasoner Stage 3:跨 Query CoT 多样性奖励(DIVER 改造版)

在 SIDReasoner Stage 3 GRPO 的原始推荐 reward 之上,叠加"当前用户 CoT 与
相似用户成功 CoT 的文本差异"作为内在奖励,以鼓励面向不同用户生成更具区分度
的推理路径。设计源自 DIVER(arXiv:2509.26209)的 TD / Potential-based
Reward Shaping / Conditional + Balanced Shaping,比较集合从"同一 Query 的
G 条 rollout"替换为"in-batch 相似用户的代表性 CoT"。

> 注意:跨 Query 设置下 Reference 依赖 batch 内其他用户,原论文的
> 最优策略不变性定理不再自动成立,本设计视为需实验验证的 reward。

**默认关闭。** 关闭时 Stage 3 的 reward / advantage / 训练行为与改造前完全一致。

---

## 1. 最终 Reward

对第 q 个 query 的第 i 条 rollout:

```
d(o_{q,i}) = (1 / K_q) * Σ_{q' ∈ N_K^valid(q)} [1 - BLEU_sym(c_{q,i}, c~_{q'})]

R'_{q,i} = R_{q,i} + λ(t) * I_{q,i} * clip( γ^{T_{q,i}} * d(o_{q,i}), 0, σ )
```

- `R_{q,i}`:正确性 reward(steprule / onerec_rollout_beam /
  onereason_div_rollout_beam,由 `correctness_reward_type` 选择),
  **原样保留**;
- 合并方式 `combine`:`add`(默认, 上式)或 `mul`
  `R' = R * (1 + intrinsic)`(R=0 的 rollout 不获得多样性加成);
- `N_K^valid(q)`:in-batch 余弦 top-K 相似用户中,有"推荐成功"rollout 的
  子集;`K_q = |N_K^valid(q)|`,为 0 时 `d = 0`;
- `c~_{q'}`:邻居 q' 的 G 条 rollout 中随机选一条成功 CoT 作 Reference
  (多个成功时随机,seed 可配);
- `similarity`(配置)可选 `bleu`(默认, 下式)、`ngram_jaccard`、`llmjudge`
  或 `niche`(小众物品多样性, 三个分量的设计见第 8 节):
  `sim(a,b) = |G_n(a)∩G_n(b)| / |G_n(a)∪G_n(b)|`(n-gram 集合 Jaccard,
  `ngram_n` 控阶数), `TD = 1 - sim`。`make <ds>-sidreasoner-rl-ngram`
  即 Jaccard 版。`llmjudge` 由 Qwen3 judge 看双方历史, 对 evidence /
  strategy / profile 三维各评 0-4 相似度(`sim = mean/4`, 提示词见
  `src/reward/interdiv/judge_prompt.txt`), `make <ds>-sidreasoner-rl-beam-llmjudge`。
- `BLEU_sym(a,b) = (BLEU(ref=a,cand=b) + BLEU(ref=b,cand=a)) / 2`,
  口径与 DIVER 官方 `calculate_belu_matrix` 一致:
  whitespace 分词、截断前 1000 字符、5-gram 权重
  `(0.02, 0.1, 0.15, 0.25, 0.38)`(和 0.9,不归一)、无平滑
  (任一阶零精度 → BLEU=0)。自实现与 nltk `sentence_bleu` 300/300
  随机对逐值一致(见测试);
- `I_{q,i}` 由 `success_mode` 决定: `full`(默认, 三级全匹配)/
  `prefix`(首层匹配)/ `none`(仅要求答案可解析)。**不能**用
  "原 reward > 0"判断:StepRule 的格式分 0.1 可在零匹配时给正分;
- `λ(t)`:线性衰减 `λ_init → λ_final`,跨 `decay_steps` 个 train batch
  (内部计数,验证调用不计);
- `γ^{T}`:折扣,`T` = rollout token 数;`γ=1.0`(默认)时恒为 1,不需要
  tokenizer;
- `σ`:intrinsic 裁剪上界(防奖励黑客)。

**仅对 split == 'train' 的 batch 生效**;验证/测试 batch 只算原始 reward。

### 低成功率阶段的 reward 设计开关(默认 = DIVER 严格口径)

Stage 2 起点策略的 SID 三级全匹配成功率仅 ~1%, 严格口径下
`intrinsic = λ·I·clip(TD)` 几乎恒为 0(实测 probe: 93.8% 的 GRPO 组
intrinsic 零方差)。四个独立开关(默认均保持原行为):

| 参数 | 取值 | 作用 |
|---|---|---|
| `success_mode` | `full`(默认)/`prefix`/`none` | I 的口径: 三级全匹配 / 首层 code 匹配 / 不做正确性约束(仅要求答案可解析) |
| `reference_mode` | `full`(默认)/`prefix`/`none` | 邻居 Reference 准入: 三级全匹配 / 首层 code 匹配 / 退回任一可解析 CoT(优先级 full > prefix > any; 旧值 `success`/`any` 兼容) |
| `bleu_smoothing` | `none`(默认)/`method1` | DIVER 无平滑(自然语言 CoT 上 BLEU 常为 0, TD 饱和到 1)/ nltk add-1 平滑 |
| `td_norm`（也可写 `norm`） | `none`（默认）/ `group_rank` / `batch_rank` / `group_minmax` / `batch_minmax` | `group_` 按 GRPO 组归一化，`batch_` 按整个 reward batch 归一化。旧值 `rank` 等同于 `group_rank` |
| `similarity` | `bleu`(默认)/`ngram_jaccard` | 多样性度量: SymBLEU / `1 - \|G_n(a)∩G_n(b)\| / \|G_n(a)∪G_n(b)\|`(n-gram 集合 Jaccard) |
| `ngram_n` | 1 | ngram_jaccard 的 n 阶数 |

n-gram、BLEU、LLM-Judge 和 Niche 共用这些模式。只选 `success=true` 且
`cot_ok=true` 的 rollout。秩归一化把原始 TD 按秩映射到 [0, 1]，并列取
平均秩。min-max 从选中 rollout 的最小值和最大值计算
`(TD - min) / (max - min)`，所有选中分数相等时取 0。只有一条时两种方式都取 0。
没有有效参考时，原始 TD 记为 0，仍按这两项条件判断是否参与。
归一化后的 `td_used` 再进入折扣、裁剪和加权。

在 32 query x 8 rollout 的真实 probe batch 上实测(Video_Games, Stage 2):

| 配置(success/ref/smooth/norm) | success | n_refs | intrinsic 零方差组 | intrinsic 组内 std | intrinsic 均值 |
|---|---|---|---|---|---|
| full/full/none/none(原版) | 1.2% | 0.59 | 93.8% | 0.002 | 0.001 |
| prefix/prefix/method1/none | 11.3% | 3.41 | 56.2% | 0.032 | 0.022 |
| prefix/prefix/method1/rank | 11.3% | 3.41 | 84.4% | 0.011 | 0.008 |
| prefix/full/method1/rank | 11.3% | 0.59 | 84.4% | 0.010 | 0.008 |
| none/none/method1/rank | 98.8% | 8.00 | **0.0%** | 0.066 | 0.099 |

要点: (1) 只去掉成功门而不做秩归一化, TD≈1 的组内常数平移会被 GRPO
advantage 的均值消去, 几乎不产生学习信号; (2) 秩归一化的组内区分度只在
"每组 >=2 条可塑形 rollout"时成立: prefix 门(约 11% rollout 通过)下大量
组只有 0~1 条通过, 零方差组占比反而高于不做 rank; (3) `none/none/rank`
让每个组都有区分度, 代价是多样性项不再与正确性挂钩(需保证 λ·σ 小于
相邻 base 奖励档差, 如 steprule 的 0.25, 避免排序反转; λ=0.2, σ=1.0 时
最大差 0.2 < 0.25)。reference 门只影响 TD 覆盖率(见 n_refs 列), 不影响
塑形覆盖; 塑形覆盖由 success_mode 决定。

---

## 2. 实现流程与模块调用关系

```
离线(一次性):
  parquet(train/valid/test) ──parse prompt 历史 SID──> 物品 id(0 起始)
        │                                              │
        └── extra_info.index 对齐校验                   ▼
                                    UserEncoder(sasrec | bge_mean)
                                              │ L2-normalize
                                              ▼
        runs/sidreasoner/user_embed/<Category>_<encoder>.npz
        (split, index, embeddings, history_items, user_ids, meta)

训练(每个 step,verl driver 端):
  BatchRewardManager(batch) 把 B query × G rollout 一次性传入
        ▼
  direct_recommendation_interdiv.rule_base_reward()
        ├─ 正确性 reward:逐行复用 correctness_reward_type 选中的实现
        │  (steprule / onerec_rollout_beam / sid_rewards 注册表;未改动)
        └─ InterQueryDiversityAddon.compute_batch()
              ├─ 按 (split,index) 分组到 query(拒绝跨 query 混组)
              ├─ UserEmbeddingStore 查用户向量/历史/user_id
              ├─ build_same_user_mask:同 user_id 精确排除 + Jaccard≥0.8 兜底
              ├─ topk_neighbors:U·Uᵀ 余弦 + top-K(GPU tensor op)
              ├─ 每个邻居随机选一条成功 CoT 作 Reference
              ├─ 逐 rollout:d = mean(1 - BLEU_sym)(独立计算)
              └─ shape_rewards:λ(t) 衰减 + I 条件 + γ^T + clip σ
        ▼
  R' = R + intrinsic  ──> verl GRPO advantage(未改动)
```

分布式说明:verl 的 reward 计算发生在 driver(或单个 async Ray actor)上,
拿到的是 repeat 后的**全局逻辑 batch**(`ray_trainer.py:1274`),因此
in-batch 邻居检索跨 DP rank 天然一致,无需跨设备同步。

---

## 3. 离线 User Embedding

### 提取命令

```bash
# 默认 encoder = config 的 user_encoder_type(sasrec)
DATASET=games make games-sidreasoner-user-embed
# 等价: DATASET=games bash scripts/sidreasoner/user_embed.sh
#       可选 ENCODER=bge_mean OUT=... BATCH_SIZE=... DEVICE=cuda SASREC_CKPT=...
```

- SASRec encoder 复用 `src/models/sasrec.py` 与已训练 checkpoint
  (默认 `runs/<Category>_sasrec_bce1/best.pt`;Office/Industrial 无 SASRec
  checkpoint 时可用 `ENCODER=bge_mean`,复用 `runs/rqvae/<Category>/embeddings.npy`);
- 用户向量 = 最后一个有效交互位置 hidden state;历史 > max_seq_len 只留最近
  N 条;左 padding;空历史置零;L2 归一化;
- 提取以 **Stage 3 实际使用的 parquet** 为源,解析 prompt 里的历史 SID →
  物品 id(index.json 反查,码本碰撞取最小 id)。**不用 user_id 映射样本**,
  也不含目标物品/未来交互;
- 分批 GPU 推理(`embed_batch_size`,默认 256),`np.savez` 持久化;
- Video_Games 实跑:55,275 条(train 49,133 + test 6,142),dim=32,全部
  L2 归一化,`user_id_aligned=True`。

### 缓存格式(npz)

| 字段 | 形状 | 说明 |
|---|---|---|
| `split` | (N,) str | train / valid / test |
| `index` | (N,) int64 | parquet 行号 = 训练时 `extra_info["index"]` |
| `embeddings` | (N, D) f32 | 已 L2 归一化 |
| `history_items` | (N,) object | 每项 np.int64 数组(0 起始物品 id) |
| `user_ids` | (N,) str | 精确同用户排除用;空串=未知(旧缓存兼容) |
| `meta` | str(json) | encoder/维度/数据目录/时间/user_id_aligned/digest |

对齐保证:提取时校验 `extra_info.index == 行号`,否则报错拒绝产出;reward 端
按 `(split, extra_info.index)` 查表,缺键报错并提示重建。user_id 按行号
join 自 split CSV(抽样 500/500 行验证 parquet↔CSV 行对齐);CSV 缺失或行数
不一致时自动降级为空(user_id 仅用于排除,GMM:不同用户历史 Jaccard p95=0.000,
Jaccard≥0.8 兜底不会误伤)。

### 配置(config/sidreasoner.py → `rl.inter_query_diversity`)

| 参数 | 默认 | 含义 |
|---|---|---|
| `enable` | False | 总开关(等价 `INTER_DIV=1`) |
| `user_encoder_type` | sasrec | `sasrec` \| `bge_mean`(注册表可扩展) |
| `user_embedding_path` | None | 默认 `runs/sidreasoner/user_embed/<Category>_<encoder>.npz` |
| `top_k_neighbors` | 8 | 相似用户数 K |
| `same_user_jaccard` | 0.8 | 同用户兜底阈值 |
| `lambda_init` / `lambda_final` | 0.2 / 0.0 | 多样性权重线性衰减 |
| `decay_steps` | 300 | 衰减步数(train batch 数) |
| `gamma` | 1.0 | 折扣因子(<1 时用 `model_path` tokenizer 计 token 数) |
| `sigma` | 0.5 | intrinsic 裁剪上界 |
| `bleu_div_len` | 1000 | BLEU 参与比较的 CoT 前缀字符数 |
| `seed` | 0 | Reference CoT 随机选择种子 |
| `metrics_path` | None | 默认 embedding 同目录 `interdiv_metrics.jsonl` |
| `log_every` | 1 | 摘要打印间隔 |
| `embed_batch_size` / `sasrec_checkpoint` / `bge_item_embeddings` | - | 仅离线提取用 |
| `niche_*`(dict/lexstats/sif 路径, summary_method, embed_*) | 见第 8 节 | 仅 `similarity="niche"` 使用 |

### 日志

每个 train batch 追加一行 JSON 到 `logs/<实验名>_interdiv_metrics.jsonl`
(rl.sh 自动按实验名隔离),并打印摘要。reward 组成三统计(命名对齐 verl
critic/*,供 `make <ds>-sidreasoner-reward-curves` 画图):

```
[interdiv] step=0 lam=0.2000 base=0.2750 aug=0.2798 success=0.250 td=0.1251
           td_ok=0.0963 intr=0.0048 nbrs=8.00 sim=0.6590 (0.1s)
```

JSONL 逐 step 含:

| 字段 | 含义 |
|---|---|
| `critic/correctness_rewards/{mean,max,min}` | 正确性 reward(StepRule / Rollout-Beam) |
| `critic/diversity_rewards/{mean,max,min}` | 多样性 reward(BLEU / n-gram) |
| `critic/total_rewards/{mean,max,min}` | 总 reward(= verl 的 critic/rewards) |
| `lambda` / `success_rate` / `td_mean` / `valid_neighbors_mean` / ... | 塑形与诊断量 |

reward 组成曲线(样式与 `sidreasoner_rl_curves.py` 一致, 展示哪些曲线改
`src/eval/sidreasoner_reward_curves.py` 的 `DEFAULT_SERIES`):

```bash
make games-sidreasoner-reward-curves        # -> results/sidreasoner/Games/figures/rewards.png
make games-sidreasoner-reward-curves ARGS="--metrics logs/<实验名>_interdiv_metrics.jsonl"
```

字段:原始 reward 均值、增广 reward 均值、推荐成功率、全部/成功 rollout 的
平均 TD、平均裁剪后 intrinsic、当前 λ、有效邻居数、邻居相似度均值。

---

## 4. 文件清单

**新增**

| 文件 | 作用 |
|---|---|
| `src/reward/user_represent/{__init__,registry,base,sasrec_encoder,bge_mean_encoder}.py` | 用户表征编码器接口 + 注册表(SASRec / BGE 均值) |
| `src/data/sidreasoner_user_embed.py` | 离线 User Embedding 提取脚本(parquet → npz) |
| `src/reward/interdiv/{__init__,bleu,llmjudge,similarity,cot_utils,user_store,neighbor,shaping,reward}.py` + `judge_prompt.txt` | BLEU / n-gram Jaccard / LLM-Judge 语义相似度 / CoT 与成功判定 / 缓存查表 / Top-K 检索 / 塑形(add·mul) / 编排 |
| `src/reward/interdiv/niche.py` + `src/data/sidreasoner_niche_preprocess.py` | Niche 小众物品多样性 scorer(c1/c2/c3)与离线预处理(第 8 节) |
| `src/reward/direct_recommendation_interdiv.py` | verl BatchRewardManager 入口(正确性 reward x 多样性 reward;`correctness_reward_type` 旧名 `base_reward_type` 兼容) |
| `scripts/sidreasoner/user_embed.sh` + `rl_bleu.sh` / `rl_ngram.sh` / `rl_beam_bleu.sh` / `rl_beam_ngram.sh` / `rl_beam_llmjudge.sh` / `rl_beam_niche.sh` / `rl_div.sh` | 提取 / 训练命令包装(正确性 x 多样性矩阵) |
| `tests/test_interdiv.py` | 单测(含入口兼容性、combine、门控/度量矩阵) |
| `tests/test_niche.py` | niche 单测(预处理/三分量/总结段边界/集成) |
| `tests/test_user_embed.py` | 提取/编码器/泄漏测试 |
| `docs/sidreasoner_inter_query_diversity.md` | 本文档 |

**修改**

| 文件 | 改动 |
|---|---|
| `config/sidreasoner.py` | 新增 `rl.inter_query_diversity` 配置块(默认关闭) |
| `scripts/sidreasoner/rl.sh` | 新增 `INTER_DIV` 分支(关闭时零改动);新增 `${INTERDIV_FLAGS}` 占位 |
| `Makefile` | 新增 `<dataset>-sidreasoner-user-embed`、`<dataset>-sidreasoner-rl-bleu` |
| `src/eval/result_routing.py` | 评测结果路由识别 `_<corr>_<div>` 后缀 -> `think_<corr>_<div>/`(旧 `_bleu`/`_ngram` 兼容) |
| `tests/test_sid_rewards.py` | 修复历史遗留的 `torch.no_grad` 全局污染(导入后恢复),使全量测试顺序无关 |

**未改动**:StepRule / Rollout-Beam reward、Stage 1/2、模型结构、GRPO
advantage、verl fork(仅使用其自带 `batch` reward manager)。

---

## 5. 训练命令(服务器)

正确性 reward(StepRule / Rollout-Beam)与多样性 reward(BLEU / n-gram)解耦,
`COMBINE=add|mul` 控制合并;通用入口 `rl-div` 也支持两者组合:

```bash
# 0) 前置: Stage 2 checkpoint 就绪;SASRec checkpoint 存在(或 ENCODER=bge_mean)

# 1) 离线 User Embedding(每个数据集一次;parquet 更新后需重跑)
make games-sidreasoner-user-embed

# 1b) Niche 多样性的离线预处理(仅 similarity="niche" 需要, 见第 8 节)
make games-sidreasoner-niche-data

# 2) 正确性 x 多样性(StepRule 底座, 加法合并)
make games-sidreasoner-rl-bleu        # 产物 ..._Qwen3-1.7B_steprule_bleu
make games-sidreasoner-rl-ngram       # 产物 ..._Qwen3-1.7B_steprule_ngram

# 3) Rollout-Beam 底座(需修改版 verl fork)
make games-sidreasoner-rl-beam-bleu   # 产物 ..._Qwen3-1.7B_rolloutbeam_bleu
make games-sidreasoner-rl-beam-ngram  # 产物 ..._Qwen3-1.7B_rolloutbeam_ngram
make games-sidreasoner-rl-beam-llmjudge  # LLM-Judge 语义多样性(先起 vllm serve, 见 README)
make games-sidreasoner-rl-beam-niche     # Niche 小众物品多样性(先跑 1b)

# 4) 通用入口(任意组合; COMBINE=mul 为乘法: R' = R*(1+intrinsic))
make games-sidreasoner-rl-div CORR=beam DIV=ngram COMBINE=mul
make games-sidreasoner-rl-div CORR=beam-div DIV=bleu

# 5) 合并 / 评估(think 目录按 <corr>_<div> 路由)
make games-sidreasoner-merge RL_SUFFIX=_steprule_bleu
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_steprule_bleu/global_step_100/actor_merged
# -> results/sidreasoner/Games/think_steprule_bleu/step_100.json
# 多样性分析镜像 -> results/diversity/Games/think_steprule_bleu/step_100.json
make games-diversity-analyze INPUT=results/sidreasoner/Games/think_steprule_bleu/step_100.json

# 6) 关闭(默认):与原 Stage 3 完全一致,不需要 embedding 文件
make games-sidreasoner-rl
```

产物路由(`<corr>` ∈ steprule/rolloutbeam/rolloutbeam_div,
`<div>` ∈ bleu/ngram/llmjudge/niche):

| 环节 | 路径 |
|---|---|
| checkpoint | `checkpoints/RecRL_Reasoning/<C>_stage3_rl_Qwen3-1.7B[_own][_<corr>][_<div>]/` |
| 训练指标(JSONL) | `logs/<同名实验>_interdiv_metrics.jsonl` |
| 思考评测 | `results/sidreasoner/<D>/think_<corr>[_<div>]/step_N.json` |
| 多样性分析 | `results/diversity/<D>/think_<corr>[_<div>]/step_N.json`(镜像输入) |
| reward 曲线 | `results/sidreasoner/<D>/figures/rewards.png` |

注意: 启用多样性后实验名统一为 `_<corr>_<div>`(如 `_steprule_bleu`),
与改造前的 `_bleu`/`_ngram` 不同;训练数学与超参未变。旧后缀 checkpoint
的评测路由仍兼容(`think_bleu` / `think_ngram`)。

---

## 6. 测试与兼容性

` .venv/bin/python -m pytest tests/ -v` → **97 passed**(含既有测试)。

覆盖点:

1. SASRec 向量:形状/归一化/超长历史截断/空历史置零/确定性;
2. 无泄漏:历史仅取自 prompt,损坏 `extra_info.answer` 后产物不变;
3. Top-K:排除自身;精确 user_id 排除(历史完全不同的同用户也排除);
   Jaccard 兜底;K 超过可用邻居时截断;单 query 边界;
4. BLEU:同文≈1、无交集=0、部分重叠∈(0,1)、双向对称、截断生效、
   与 nltk `sentence_bleu` 逐值一致(300/300);
5. 分组:不同 query 的 CoT 不会并入同一组;验证 batch 不塑形;
6. 无有效邻居:`K_q=0 → intrinsic=0`;
7. 失败 rollout 无多样性奖励(格式分 0.1 > 0 也不发放);
8. 塑形:λ 线性衰减、γ^T 折扣、σ 裁剪、条件指示;
9. 兼容性:关闭开关时逐行 score 与原 StepRule reward **完全一致**;
   关闭时 `rl.sh` 生成的命令与原脚本逐字节一致(实测无 `interdiv`/`batch`
   相关 flag)。

真实数据实证(Video_Games):

- 同用户行(CSV A4433 的 8 条样本):user_id 排除开启后互为邻居 **0 次**,
  关闭时 20 次(证明精确排除必要);
- 不同用户历史 Jaccard:p95=0.000,max=0.167(0.8 阈值无误伤);
- 模拟 batch(64 query × 4 rollout):成功率 0.25,avg sim 0.66,
  8 个有效邻居,base 0.275 → aug 0.280,耗时 ~0.1s/batch。

已知限制:

- λ(t) 用进程内计数,重启后从 0 重计(不随 checkpoint 恢复);
- 邻居多样性依赖 batch 组成(B=256 时每个 query 的可用邻居充足);
- gamma<1 时需要 `model_path` tokenizer(已在启用分支自动传入 Stage-2
  checkpoint 路径)。

---

## 7. 离线 reward 探针

先在 GPU 上生成一次 Stage 2 测试 rollout 缓存，再复用缓存计算不同奖励。
生成使用 Stage 3 测试 parquet 和训练式采样，每个 query 采样 16 条。
缓存保留原样本键、训练 batch 口径的分组、完整回复、按顺序排列的 SID beam 候选、
真实 SID 和输入历史（从 parquet prompt 取 SID，从物品目录取 title）。

```bash
make games-sidreasoner-rollout-cache
make games-sidreasoner-reward-probe ARGS="--diversity ngram --correctness rollout-beam --ngram-n 2"
make games-sidreasoner-reward-probe ARGS="--diversity ngram --correctness rollout-beam --ngram-n 3"
```

默认缓存路径为 `data/sidreasoner_rollouts/<Category>_stage2_test.jsonl`，
报告写入 `results/reward_probe/`，文件名含正确性奖励（若指定）和 n-gram 阶数。
可用 `--out` 指定两种输出位置，用探针的 `--cache` 指定输入缓存。
生成时可用 `--n-queries 8 --out <路径>` 做小规模试跑，每个 query 仍采 16 条。
跨用户奖励需要同一 parquet 的用户向量缓存，缺失时执行
`make games-sidreasoner-user-embed`。

探针要求同时指定 `--diversity`（`ngram`、`bleu`、`llmjudge`、`niche`、`custom`）
和 `--correctness`（`steprule`、`rollout-beam`、`rollout-beam-div`）。
最终奖励为正确性奖励加上经过门控、塑形和加权的多样性奖励。
可用 `--reward-kwargs k=v` 覆盖奖励参数，
用 `--shaping-step` 固定塑形步数（默认 0）。多个 N 用同一份缓存和相同的
参考抽样种子比较。

例如，`--reward-kwargs norm=batch_minmax` 按缓存中的每个 batch 做 min-max 归一化，
`--reward-kwargs norm=group_minmax` 则按 query 归一化。比较不同模式时，可用
`--out` 分别保存报告。

JSON 在 `summary.diversity`、`summary.correctness`、`summary.total` 下报告各自的指标。

- **组内均值和标准差**（`within_group_mean`、`within_group_std`）分别先在每个组的
  全部 rollout 上计算均值和总体标准差，再对所有 GRPO 组等权平均。
  `zero_variance_rate` 是对应分数组内标准差为零的组占比。
- **有效比较覆盖率**以全部 query 为分母，统计至少有一对可比较 rollout 的 query。
  多样性比较要求两条 CoT 完整、通过门控、有有效参考，且正确性分数相同。
  正确性比较要求多样性分数相同，总奖励比较使用全部组内对。
  多样性部分额外记录有至少一条合格 rollout 的 query 数和有参考诊断的 query 数。
- **真实物品排名一致率**沿用相应分数的配对条件。自定义多样性奖励未提供参考诊断时，
  多样性覆盖率为 `null`，排名一致率考虑所有正确性分数相同的组内对。真实 SID 未命中视为
  末尾之后，只取排名不同的对，reward 与排名同向记 1，反向记 0，reward 并列记 0.5。
  先对 query 内的成对结果求均值，再跨 query 平均。无比较样本时为 `null`。
- **n-gram 零重叠率**统计符合案例条件的 rollout 与所有有效参考 CoT 的逐对比较，
  用距离等于 1 的比较数除以总比较数。分母为零时为 `null`。
- **n-gram 合格 rollout 统计**（`eligible_within_group_mean`、`eligible_within_group_std`）
  只用 `success=true`、`cot_ok=true` 且 `n_refs >= 1` 的多样性分数。
  先逐组计算均值和总体标准差，再对有合格 rollout 的组取平均。
  只有一条合格 rollout 的组的标准差为零。
  `eligible_queries` 是参与聚合的组数。没有合格组时两个指标均为 `null`。
- **案例**以一条符合条件的候选 CoT 及其全部实际参考 CoT 为一组。若超过 100 组，
  按 `--case-seed` 等概率抽样 100 组。JSON 保留双方各自的 TD 和最终 reward、
  双方历史、参考的用户相似度与逐对 TD（该 reward 支持时）。

`--diversity custom --reward-entry <入口.py>` 支持原 verl 逐 batch API，
逐条 API 使用 `--per-row`。返回浮点数或含 `score` 的字典，分数为已加权的
多样性奖励，探针会将它与所选的正确性奖励相加。
启用归一化时，探针对自定义入口返回的 `score` 进行归一化，并沿用未入选行的分数。
缺少 `cot_ok` 或 `success` 的时候，探针根据 CoT 和配置的 `success_mode` 补齐。
跨用户覆盖率与案例还需返回 `n_refs` 和 `references`。使用 `norm=none` 时，
由自定义入口提供 `cot_ok` 和 `success`。每个参考的 `row_index` 指向同一 batch 的 rollout，此外可返回
`user_similarity` 和逐对 `td`。

---

## 8. Niche(小众物品)多样性奖励

`similarity="niche"`(`make <ds>-sidreasoner-rl-beam-niche`,
`src/reward/interdiv/niche.py`)。塑形/邻居检索/Reference 选择与
BLEU/ngram/llmjudge 完全一致, 只替换 d(o) 的计算:

```
d(o_{q,i}) = ( c1 + c2 + c3 ) / 3
```

**niche 物品定义**(离线统计): train split 的 dedup (user_id, item) 对计频
(物品的独立交互用户数), 物品按频次降序排序后**物品数的后 80%** 为 niche
(频次并列按物品 id 升序截断; index.json 全量物品参与, 零频物品自动 niche;
Video_Games 实测: 3858 物品中 3086 niche, 边界 17 用户, niche 覆盖 41.6%
交互量)。与官方/自建 SID 集无关(仅依赖 item id), 但 sid_to_item 取自当前
index.json, **切换 SID 集后需重跑预处理**。

三个分量:

| 分量 | 定义 | 无法计算时 |
|---|---|---|
| c1 信息提取比例 | \|niche(CoT SID) ∩ niche(history)\| / \|niche(history)\|, 集合语义去重; 分子限定历史内(堵死编造冷门 SID 的 hack), 幻觉 SID 丢弃 | 历史无 niche 物品 -> 0 |
| c2 小众大众相似度 | 含 SID 的句子按"句内物品全 niche/全大众"分成两个去重集合(**mixed 句两侧剔除**, 避免与自身算 sim=1), BGE 句向量全对余弦均值 sim, c2 = 1 - sim | 任一侧为空 -> 0 |
| c3 总结句相似度 | 总结段 = 最后一个 SID 所在句之后的文本; 与 ≤8 个相似用户 Reference CoT(选择逻辑与 BLEU/ngram/llmjudge 一致, reference_mode 口径)的总结段算余弦均值 sim, c3 = 1 - sim | 无有效 Reference 或总结段为空 -> 0 |

**c3 的度量二选一**(`niche_summary_method`, 训练时可切换):

- `sif`(默认): Arora et al. ICLR 2017 平滑逆频率句向量 —— 词向量按
  `a/(a+p(w))` 加权平均后减去语料公共成分(第一右奇异向量)再归一。
  词向量 = BGE(词) 的 CLS 输出(离线预计算静态表), p(w) = 离线语料词频;
  词表外词跳过。频次加权 + 语义向量, 对短文本的改写鲁棒(TF-IDF 的
  同义词盲会低估相似度导致 reward 饱和);
- `tfidf`: BM25 加权稀疏向量余弦(概率化 idf `ln(1+(N-df+0.5)/(df+0.5))`
  \+ tf 饱和 `tf(k1+1)/(tf+k1)`, b=0)。idf 查离线表, **未登录词取最大
  idf**(当最稀有词; 注意与 diversity_analysis 的 `idf.get(t,1.0)` 方向相反)。

停用词 = 仓库通用英文表(diversity_analysis.STOPWORDS) ∪ **自动生成的域内
停用词**(离线语料 df >= 50% 的内容词, 如 user/game/prefer, 数据集无关)。

### 离线预处理与命令

```bash
# 0) 一次性预处理(约 1 分钟; SIF 词向量需下载 BGE, 服务器上跑)
make games-sidreasoner-niche-data
#    -> data/raw/Amazon/niche/Video_Games/{popularity.json, cot_lexstats.json, sif_words.npz}
#    (语料 = index/Video_Games.integrated_narrative.csv 的 reasoning_path 全量 49133 条)

# 1) 训练(前置同 rl-bleu: user-embed + 修改版 verl fork)
make games-sidreasoner-rl-beam-niche    # 产物 ..._Qwen3-1.7B_rolloutbeam_niche
#    切换 c3 度量: config rl.inter_query_diversity.niche_summary_method = tfidf|sif

# 2) reward 探针(验证组内区分度; 需先完成预处理)
make games-sidreasoner-reward-probe ARGS="--diversity niche --correctness rollout-beam"
```

配置(`config/sidreasoner.py` 的 `rl.inter_query_diversity`, **全部参数走
config, 不需要 export 环境变量**): `niche_dict_path` / `niche_lexstats_path`
/ `niche_sif_path`(默认 `<root>/niche/<Category>/...`)、
`niche_summary_method`(sif|tfidf)、`niche_embed_model`(**直接填本地模型目录
即可离线**, 与预处理共用)、`niche_embed_device`(默认 cpu, reward 在 driver
端执行)、`niche_embed_batch_size`(句向量批大小)、
`niche_embed_num_threads`(CPU 推理 torch 线程数; Ray 会给 driver 设
`OMP_NUM_THREADS=1`, 不覆盖时 BGE 会慢 ~10 倍, 建议≈物理核数)、
`niche_only_success`(默认 True: 只对能拿到 intrinsic 的成功 rollout 计算
c1/c2/c3 —— 失败行 intrinsic 恒 0, 训练数学不变, 低成功率阶段可省 ~90%
的 BGE 计算; 关掉则失败行也计算, td 诊断覆盖全部 rollout)。

### 性能说明

单 batch 的主要成本是 c2 的 BGE 句向量编码(CPU, 批内去重 + 进程级缓存)。
实测首步曾达 ~700s, 原因是 Ray driver 的 `OMP_NUM_THREADS=1`;
`niche_embed_num_threads` 设为物理核数后可降一个数量级。另两个开关:
`niche_only_success`(默认开, 成功率 ~10% 时只编码约 1/10 的 rollout)与
`niche_embed_device="cuda"`(更快, 但 driver 与训练共用 GPU, 需留意显存)。
c1/c3 均为纯 Python/numpy, 开销可忽略。

### 诊断指标

JSONL 逐 batch 追加 `niche/c1_mean|c2_mean|c3_mean`(分量均值)、
`niche/c1_defined_frac|c2_defined_frac|c3_defined_frac`(**可计算率**,
区分结构性 0 与真实低分)、`niche/hist_niche_mean`、`niche/n_scored`;
逐 rollout 的 reward_extra_info 含 `niche_c1/c2/c3` 与 defined 标志。
注意开着 `niche_only_success`(默认)时, 失败行的 `td` 与 `niche_*`
恒为 0(它们本来也拿不到 intrinsic), 看诊断请用 `td_ok`(`td_success_mean`)。

实测口径(Video_Games SFT CoT + tfidf 变体, 200 行抽样): c1 可计算 73%
(历史含 niche 物品的行占比), c2 可计算 ~19%(mixed 句剔除后同时有两种
句子的 CoT), c3 可计算 77%。注意 **c1/c2 不依赖邻居 Reference**(K_q=0 时
只有 c3 归 0, c1/c2 照常计算), 这与 BLEU/ngram/llmjudge 的 "K_q=0 -> d=0"
不同, 是设计内行为。

### 已知限制与风险

- c2 奖励"同一 CoT 内两种句风的反差", 策略可能学出固定的冷门物品话术
  (跨用户模板收敛, 与用户间多样性目标偏离); lambda 衰减 + sigma clip +
  三项平均可限制影响, 建议监控不同用户 c2 分布是否塌缩;
- SIF 的词表外词(离线语料未见的词)被跳过, RL 中出现的新词不产生 c3 信号;
- c2 句向量在 reward 进程内默认 CPU BGE; 已用 `niche_only_success` +
  `niche_embed_num_threads` + batch 内去重/进程级缓存控制成本, 仍嫌慢可
  `niche_embed_device="cuda"`(注意 driver 端显存)或改用 tfidf 变体
  (c2 仍需句向量)。
