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
make games-sidreasoner-user-embed      # 跨 Query 多样性奖励的离线 User Embedding(一次性)
make games-sidreasoner-rl-bleu         # Stage 3 变体: 跨 Query CoT 多样性奖励(DIVER 改造版, 见下)

# 挂 tmux 跑:
make tmux TARGET=games-sidreasoner-sft SESSION=sft-games
make tmux TARGET=games-sidreasoner-sft-gem SESSION=sft-gem-games
make tmux TARGET=games-sidreasoner-activation-gem SESSION=act-gem-games

make tmux TARGET=games-sidreasoner-rl-beam SESSION=rl-beam-games
make tmux TARGET=games-sidreasoner-rl-beam-div SESSION=rl-beam-div-games
make tmux TARGET=games-sidreasoner-rl-bleu SESSION=rl-bleu-games

tmux ls                                # 1. 列出服务器上仍在跑的会话
tmux attach -t sft                     # 2. 接回对应会话(Ctrl-b d 只退出查看, 不停任务)
tmux kill-session -t sft               # 3. 清理会话

# 3) 合并 Stage 3 的 FSDP 分片 checkpoint
make games-sidreasoner-merge
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam
make games-sidreasoner-merge RL_SUFFIX=_rolloutbeam_div
make games-sidreasoner-merge RL_SUFFIX=_bleu

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

Stage 1/2 的 CE 损失换成 [GEM](https://arxiv.org/abs/2408.16673) 熵正则逆 KL
(更多样的冷启动对 Stage 3 RL 探索有利); 评估/早停仍用 CE eval_loss,
产物目录自动加 `_gem` 后缀, 超参在 `config/sidreasoner.py` 的
`training.sft.gem` / `training.activation.gem`:

```bash
make games-sidreasoner-sft-gem           # Stage 1 GEM -> runs/sidreasoner/Video_Games_stage1_sft_gem
make games-sidreasoner-activation-gem    # Stage 2 GEM(默认接 Stage 1 GEM 的 final_checkpoint)
make games-sidreasoner-think MODEL=runs/sidreasoner/Video_Games_stage1_sft_gem/final_checkpoint
#      -> 结果路由到 Games/sft_gem/stage1.json
```

## 评测结果目录结构(两级路由)

结果按 数据集 x 训练变体 自动路由到 `results/sidreasoner/<Dataset>/<variant>/`,
变体从 MODEL 路径后缀推断(`_gem` / `_rolloutbeam` / `_rolloutbeam_div` /
`_bleu` / 无=steprule; 温度>0 加 `_temp`; 自建 SID 加 `_own` 文件后缀):

```bash
make sidreasoner-organize        # 旧版平铺结果一键归档(DRY_RUN=1 预览)
make games-sidreasoner-curves    # 训练曲线: 默认 5 条线, _temp 为 95% CI 带, 金色水平线=论文值
```

## 思考模式评估: 采样解码 + 多 seed 指标(默认)

CoT 采样解码(top_p/温度/seeds 在 `config/sidreasoner.py` 的 `evaluation`
段, `think_temperature=0.0` 恢复贪心), 默认 5 seed 输出
`step_N_summary.json`(均值/标准差/95% CI):

```bash
make games-sidreasoner-metrics                                        # summary 自动打印聚合表
DATASET=games .venv/bin/python -m src.eval.sidreasoner_metrics aggregate \
    --path <单seed文件1>.json,<单seed文件2>.json                       # 手动聚合
# 需要 predict/cot 明细做多样性分析时: evaluation.save_per_seed_results=true
```

## 把玩 (终端对话)

```bash
make games-sidreasoner-chat MODEL=runs/sidreasoner/Video_Games_stage1_sft/final_checkpoint   # /help 查看命令
```

不想从头训练: 下载[官方 checkpoint](https://huggingface.co/SIDReasoner-Models)
后跳过 Stage 1-3, 直接跑 think + metrics。

## Stage 3 前置

```bash
git clone --depth 1 https://github.com/HappyPointer/SIDReasoner ../SIDReasoner   # verl fork
make tmux TARGET=install-rl-env USE_MEGATRON=0 USE_SGLANG=0   # 跳过用不到的 Megatron/sglang
```

## Stage 3 奖励变体

三个变体从同一 Stage 2 checkpoint 出发, 产物按实验名后缀完全隔离
(无 / `_rolloutbeam` / `_rolloutbeam_div` / `_bleu`), 可并存 A/B:

```bash
make games-sidreasoner-rl              # StepRule(论文原版): 采样 SID 0.25/0.5/1.0 步进分 + 0.1 格式分
make games-sidreasoner-rl-beam         # Rollout-Beam: CoT 后 beam K 个候选的匹配 code 数最大值
make games-sidreasoner-rl-beam-div     # 再乘 OneReason R_div(候选首层 code 去重度)
make games-sidreasoner-user-embed      # (bleu 前置)SASRec 用户向量离线缓存 -> runs/sidreasoner/user_embed/
make games-sidreasoner-rl-bleu         # 跨 Query CoT 多样性: R + lambda*I*clip(gamma^T*TD, 0, sigma)
make games-sidreasoner-reward-probe    # reward 组内区分度探针: 采样一个 batch, 逐组报告 reward 分解
                                       # ARGS="--n-queries 32 --rollout-n 8 --reward bleu|steprule|custom ..."

# 评估(目录名对应 think_rolloutbeam / think_rolloutbeam_div / think_bleu)
make games-sidreasoner-think MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B_rolloutbeam/global_step_100/actor_merged
```

- Rollout-Beam 系需要修改版 verl fork(rollout 侧 beam search 基建), 校验:
  `grep -n sid_beam_size ../SIDReasoner/verl/trainer/config/rollout/rollout.yaml`
  等三条 grep 有输出即就绪, 普通 Stage 3 不受影响。
- beam 参数: `rl.sid_beam_size`(K, >1)/ `sid_length`(L); beam-div 另有
  `reward_kwargs.div_level`(默认 1 = 首层); beam 集中导致 R_div 全零时增大 K。
- bleu 变体参数: `rl.inter_query_diversity`(K 邻居/lambda 衰减/gamma/sigma/
  BLEU 截断等), 设计与实现详见 `docs/sidreasoner_inter_query_diversity.md`;
  默认关闭, 不影响其他 target。
- 自定义奖励: 继承 `BaseSidReward` 注册到 `src/reward/sid_rewards/`(见
  `tests/test_sid_rewards.py`), `reward_kwargs.reward_type` 显式指定即可,
  无需改 trainer。单测: `.venv/bin/python -m pytest tests/ -v`。

# 4. 自建 SID(独立训练 RQ-VAE)

```bash
make games-rqvae            # embed(BGE) -> train(RQ-VAE) -> apply(替换官方 SID) 三步连跑
make games-rqvae-embed      # ① item 文本 -> BGE 向量, 缓存 runs/rqvae/<C>/embeddings.npy
make games-rqvae-train      # ② 3 层残差 VQ(各 256 code), 产物 index.own.json
make games-rqvae-apply      # ③ 备份官方原件并替换 data/raw/Amazon 下 4 处 SID 文件
```

- 超参与质量报告: `config/rqvae.py`、`runs/rqvae/<C>/report.json`
  (码本利用率/碰撞率/与官方重叠)。
- 已知差异: 自建碰撞多于官方(BGE 对近重复 title 聚类紧, 消费端原生支持
  一 SID 多 item); 重建 RL parquet 历史更短(基于截断 10 的 CSV)。

# 5. SID 数据流与切换(官方 vs 自建)

SID 经 4 处文件进入下游(代码零耦合): `index/<C>.index.json`(item->SID 表,
SidTable/词表扩展/评估)、`{train,valid,test}/<C>_5_*.csv`(SFT/RL parquet/
评估)、`info/<C>_5_*.txt`(前缀 trie、Stage 3 reward 合法 SID 表)、
`rec_reasoning_verl/<C>/*.parquet`(Stage 3 GRPO)。SASRec/Caser/GRU4Rec
只用 item_id, 不受影响。

```bash
DATASET=games SID=official make games-sid-switch   # 切到官方 SID
DATASET=games SID=own     make games-sid-switch   # 切回自建 SID
```

- 两套快照: `data/raw_official_backup/` / `data/raw_own_backup/`, 往返无损,
  切完自动 `load_bundle` 验证; 快照不齐时 `SID_SET=own|official` 显式声明。
- 输出自动隔离: own 时训练目录/结果文件名加 `_own` 后缀, 与 official 互不覆盖。
- 旧 SID 的 checkpoint 不可跨 SID 集评估, 需按当前 SID 重训。

# 6. 思维链多样性分析(SIDReasoner)

输入一份思考模式评测 JSON(逐条记录含 `cot` 字段; `_summary.json` 不可用,
需 `save_per_seed_results=true`), 输出镜像路由的分析 JSON:
`results/sidreasoner/Games/think_steprule/step_100.json`
-> `results/diversity/Games/think_steprule/step_100.json`:

```bash
.venv/bin/python -m src.analysis.diversity_analysis results/sidreasoner/Games/think_steprule/step_1900.json
INPUT=<同上路径> PAIR_CAP=2000 DEVICE=cuda make games-diversity-analyze   # 等价
```

统计: TF-IDF/bigram 频率、TTR/熵(含/去停用词)、完全重复率、成对余弦与
Jaccard、首句模板集中度、gzip 压缩率、最相似样本对 Top-3(text=剔除 SID /
raw=保留 SID 双视角)。早期结论: 多样性坍缩发生在 Stage 2 SFT 并延续到 RL。
