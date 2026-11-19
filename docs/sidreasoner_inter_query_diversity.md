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

- `R_{q,i}`:原始推荐 reward(steprule 或 rollout_beam 族),**原样保留**;
- `N_K^valid(q)`:in-batch 余弦 top-K 相似用户中,有"推荐成功"rollout 的
  子集;`K_q = |N_K^valid(q)|`,为 0 时 `d = 0`;
- `c~_{q'}`:邻居 q' 的 G 条 rollout 中随机选一条成功 CoT 作 Reference
  (多个成功时随机,seed 可配);
- `similarity`(配置)可选 `bleu`(默认, 下式)或 `ngram_jaccard`:
  `sim(a,b) = |G_n(a)∩G_n(b)| / |G_n(a)∪G_n(b)|`(n-gram 集合 Jaccard,
  `ngram_n` 控阶数), `TD = 1 - sim`。`make <ds>-sidreasoner-rl-ngram`
  即后者, 产物后缀 `_ngram`。
- `BLEU_sym(a,b) = (BLEU(ref=a,cand=b) + BLEU(ref=b,cand=a)) / 2`,
  口径与 DIVER 官方 `calculate_belu_matrix` 一致:
  whitespace 分词、截断前 1000 字符、5-gram 权重
  `(0.02, 0.1, 0.15, 0.25, 0.38)`(和 0.9,不归一)、无平滑
  (任一阶零精度 → BLEU=0)。自实现与 nltk `sentence_bleu` 300/300
  随机对逐值一致(见测试);
- `I_{q,i}` = 采样 SID 三元组与 ground truth **三级全匹配**。
  **不能**用"原 reward > 0"判断:StepRule 的格式分 0.1 可在零匹配时给正分;
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
| `td_norm` | `none`(默认)/`rank` | 无 / 组内秩归一化: TD 按秩映射到 [0,1] 均布(仅 I=1 的 rollout 参与), 保证 GRPO 组内区分度由 λ·σ 直接控制; rank 模式建议 `sigma=1.0` |
| `similarity` | `bleu`(默认)/`ngram_jaccard` | 多样性度量: SymBLEU / `1 - \|G_n(a)∩G_n(b)\| / \|G_n(a)∪G_n(b)\|`(n-gram 集合 Jaccard) |
| `ngram_n` | 1 | ngram_jaccard 的 n 阶数 |

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
        ├─ base reward:逐行复用 StepRule / sid_rewards(未改动)
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
| `src/reward/interdiv/{__init__,bleu,cot_utils,user_store,neighbor,shaping,reward}.py` | BLEU / CoT 提取与成功判定 / 缓存查表 / Top-K 检索 / 塑形 / 编排 |
| `src/reward/direct_recommendation_interdiv.py` | verl BatchRewardManager 入口(组合原 reward + 多样性) |
| `scripts/sidreasoner/user_embed.sh` / `scripts/sidreasoner/rl_bleu.sh` | 提取 / 训练命令包装 |
| `tests/test_interdiv.py` | 38 项单测(含入口兼容性) |
| `tests/test_user_embed.py` | 13 项提取/编码器/泄漏测试 |
| `docs/sidreasoner_inter_query_diversity.md` | 本文档 |

**修改**

| 文件 | 改动 |
|---|---|
| `config/sidreasoner.py` | 新增 `rl.inter_query_diversity` 配置块(默认关闭) |
| `scripts/sidreasoner/rl.sh` | 新增 `INTER_DIV` 分支(关闭时零改动);新增 `${INTERDIV_FLAGS}` 占位 |
| `Makefile` | 新增 `<dataset>-sidreasoner-user-embed`、`<dataset>-sidreasoner-rl-bleu` |
| `src/eval/result_routing.py` | 评测结果路由识别 `_bleu` 后缀 -> `think_bleu/` |
| `tests/test_sid_rewards.py` | 修复历史遗留的 `torch.no_grad` 全局污染(导入后恢复),使全量测试顺序无关 |

**未改动**:StepRule / Rollout-Beam reward、Stage 1/2、模型结构、GRPO
advantage、verl fork(仅使用其自带 `batch` reward manager)。

---

## 5. 训练命令(服务器)

```bash
# 0) 前置: Stage 2 checkpoint 已就绪;SASRec checkpoint 存在(或 ENCODER=bge_mean)

# 1) 离线 User Embedding(每个数据集一次;parquet 更新后需重跑)
make games-sidreasoner-user-embed

# 2) 启用跨 Query 多样性(steprule 基础奖励;产物自动加 _bleu 后缀)
make games-sidreasoner-rl-bleu

# 3) 与 Rollout-Beam 组合
make games-sidreasoner-rl-bleu REWARD=rollout_beam_div

# 3b) n-gram Jaccard 度量(产物自动加 _ngram 后缀)
make games-sidreasoner-rl-ngram

# 4) 合并 / 评估(结果路由到 Games/think_bleu/)
make games-sidreasoner-merge RL_SUFFIX=_bleu
make games-sidreasoner-merge RL_SUFFIX=_ngram
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_bleu/global_step_100/actor_merged

# 5) 关闭(默认):与原 Stage 3 完全一致,不需要 embedding 文件
make games-sidreasoner-rl
```

调参全部走 `config/sidreasoner.py` 的 `rl.inter_query_diversity`(或在
config 里直接 `enable: True`);无需改任何脚本。

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

## 7. 组内区分度探针(reward 通用验证工具)

`src/eval/sidreasoner_reward_probe.py`: 采样一个 batch(B query x G rollout,
vLLM 温度采样, 默认策略 = Stage 2 checkpoint), 调用任意 reward 入口, 逐组
报告每条 rollout 的 score / base_reward / success / td / intrinsic 与组内
std、零方差组占比、组内 CoT 成对 BLEU 相似度(CoT 坍缩诊断)。

```bash
make games-sidreasoner-reward-probe                      # 默认 --reward bleu
make games-sidreasoner-reward-probe ARGS="--reward ngram"   # n-gram Jaccard 度量
make games-sidreasoner-reward-probe ARGS="--reward steprule"
make games-sidreasoner-reward-probe ARGS="--batch-json <旧探针输出>"   # 复用 batch 换 reward 重测
make games-sidreasoner-reward-probe ARGS="--reward custom --reward-entry <入口.py> --reward-kwargs k=v"  # 新 reward
```

输出 `results/reward_probe/<Category>_stage3_rl_<label>_probe.json`(命名对齐
Stage 3 实验名后缀), 内含 `batch.rows` 可直接被 `--batch-json` 复用。
局限: 依赖 rollout 侧 beam search 的 reward(rollout_beam 族)无法用本工具
生成, 需提供带 `beam_search_results` 的 batch。
