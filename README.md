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

以 games 为例(office / industrial 同名替换), 论文 Table 2 的最终结果
= 三个阶段全部完成后, 以 Stage 3 模型做思考模式评估:

```bash
# 0) 一次性准备
make download-model        # Qwen3-1.7B; 下载后修改 config.sidreasoner 中的 model 为模型的路径
make install-vllm          # 思考模式评估需要

# 1) 三阶段训练(4 卡)
make games-sidreasoner-sft             # Stage 1: 增强式 SID-语言对齐 SFT
make games-sidreasoner-activation      # Stage 2: 推理激活 SFT
make games-sidreasoner-rl              # Stage 3: GRPO 强化学习(前置见下)

# 三个阶段都是长任务(Stage 1 数天), 建议挂 tmux 跑(SSH 断开后服务器上继续,
# 输出同时落盘 logs/<会话名>.log; 不指定卡则自动挑选至多 4 张空闲卡):
make tmux TARGET=games-sidreasoner-sft SESSION=sft
tmux attach -t sft                     # 接回实时查看; 退出查看按 Ctrl-b 再按 d
# 断线重连后找回任务:
tmux ls                                # 1. 列出服务器上仍在跑的会话
tmux attach -t sft                     # 2. 接回对应会话(Ctrl-b d 只退出查看, 不停任务)
tail -f logs/sft.log                   #    不接会话也可直接读落盘日志

# 2) 合并 Stage 3 的 FSDP 分片 checkpoint
make games-sidreasoner-merge

# 3) 思考模式评估 + 计算指标
make games-sidreasoner-think \
    MODEL=checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
make games-sidreasoner-metrics         # 自动取最新 *_think_*.json, 可用 RESULT=xxx.json 指定

# 中间对照(非思考模式): MODEL 取 stage1 / stage2 / 任意 checkpoint 路径
make games-sidreasoner-eval MODEL=stage2
```

## 本地把玩三个阶段的 checkpoint(终端对话, Mac 可跑)

`src/eval/sidreasoner_chat.py` 提供网页 AI 式的终端交互, 直观感受三个阶段
各自学会了什么(Stage 1 认 SID、会翻译; Stage 2 先思考再作答但内容较套路;
Stage 3 推理更自主、SID 更准)。三个 checkpoint 都是自包含 HF 目录(模型 +
含 SID token 的 tokenizer), 从服务器整目录拷回即可(每个约 3.5GB);
Stage 3 需先在服务器 `make games-sidreasoner-merge` 再拷 `actor_merged`:

```bash
# 服务器上打包
tar czf sidreasoner_ckpt.tgz \
  runs/sidreasoner/Video_Games_stage1_sft/final_checkpoint \
  runs/sidreasoner/Video_Games_stage2_activation/final_checkpoint \
  checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
# 拷回本地, 在仓库根目录解压(tar xzf)后:
make games-sidreasoner-chat MODEL=runs/sidreasoner/Video_Games_stage1_sft/final_checkpoint
```

office / industrial 把 `games` 换掉即可。会话内 `/help` 查看全部命令:

- **Stage 1**: 进会话先 `/think off`(它学的是空思考壳后直答), 再 `/t1`
  `/t2` 验证 SID<->标题互译, `/t3` 观察直答 SID(无推理)。
- **Stage 2/3**: 保持 `/think on`(默认), `/t3` 出推荐题, 观察先 `<think>`
  推理再输出 SID; `/score <a_x><b_y><c_z>` 按论文奖励规则
  (0.25/0.5/1.0 步进 + 0.1 格式)给上一条回答打分, `/t2` 的标准答案会
  自动记录、随后直接 `/score` 即可。三个阶段各开一次会话玩同样的题,
  对比就是最直观的 case 分析。
- 默认每条输入独立单轮(贴近训练分布), `/multi on` 切多轮; `/temp 0` 贪心。
- 纯本地只依赖 torch + transformers(无需 vllm/verl); Mac 自动用
  MPS + float32(24GB 内存足够), 服务器上可
  `--device cuda --dtype bfloat16` 加速。

不想从头训练: 下载[官方 checkpoint](https://huggingface.co/Sober-Clever/SIDReasoner-Models)
后跳过步骤 1)-2), 直接跑 think + metrics。

## Stage 3 前置(一次性)

```bash
git clone --depth 1 https://github.com/HappyPointer/SIDReasoner ../SIDReasoner   # verl fork
make install-rl-env        # 或直接用官方 docker 镜像(推荐):
                           # hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0
```

网络慢导致安装耗时长时, 同样可挂 tmux 跑(断开 SSH 不中断, 上述三个变量会转发):

```bash
make tmux TARGET=install-rl-env USE_MEGATRON=0 USE_SGLANG=0   # 跳过用不到的 Megatron/sglang
tmux attach -t r4g_install_rl_env                             # 接回查看(Ctrl-b d 退出查看)
```

- RL 的初始策略固定为 Stage 2 输出, 脚本自动引用, 无需配置。
- **HF 包版本已整族钉死**: `make setup` 与 `make install-rl-env` 会安装官方脚本
  (`install_vllm_sglang_mcore.sh`, 2025-04, 与 vllm 0.8.5/torch 2.6 配套)时代的组合——
  transformers 4.51.3 / tokenizers 0.21.1 / huggingface-hub 0.30.2 / datasets 3.5.1 /
  accelerate 1.6.0 / peft 0.15.0。verl fork 与训练代码依赖 transformers 4.x,
  **请勿手动升级这几个包**; 若已升级坏, 见文末常见问题。
- verl fork 位置默认 `../SIDReasoner`, 可用 config 的 `rl.verl_home` 或 `VERL_HOME=... make games-sidreasoner-rl` 覆盖。
- docker 镜像内运行: `PY=python3 bash scripts/sidreasoner/rl.sh`。
- 不想用 wandb: 把 config 的 `rl.use_wandb`、`training.*.report_to` 关掉。

## 论文参考值(Table 2, R@5 / N@5 / R@10 / N@10)

| 数据集 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| games | 0.0710 | 0.0460 | 0.1031 | 0.0563 |
| office | 0.1373 | 0.1119 | 0.1648 | 0.1208 |
| industrial | 0.1109 | 0.0905 | 0.1438 | 0.1010 |

# 4. 自建 SID(独立训练 RQ-VAE)

官方 SID 是上游预计算产物(3 层 x 每层 256 code, 即 `<a_N><b_N><c_N>`),
本仓库不含其生成代码。以下流程在 item.json 文本上独立训练一个
RQ-VAE, 产出一份自己的 SID 并级联替换全部消费文件。

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
| `info/<C>_5_*.txt`<br>(SID\ttitle\titem_id) | 非思考评估的前缀 trie(`sidreasoner_eval.py`); 思考评估(`sidreasoner_eval_think.py`); **Stage 3 RL 奖励的合法 SID 表**(`reward/direct_recommendation_steprule.py`); 指标计算(`sidreasoner_metrics.py`); MiniOneRec |
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

- games 上的初步结论(300 条, seed 42, 三语料同测试用户): stage2 相对
  teacher, TTR 减半(0.024 vs 0.045)、distinct-2 减半(0.139 vs 0.296)、
  成对余弦均值 +50%(0.285 vs 0.189)、gzip 率更低、首句模板 Top5 覆盖
  4 倍(6.7% vs 1.7%), 且出现逐字相同的输出(完全重复率 0.7%)——
  SFT 后模型输出显著比教师数据更同质; stage3(GRPO 后)各项词汇多样性
  指标与 stage2 几乎持平(TTR 0.024 / distinct-2 0.143 / 余弦 0.279),
  但首句模板进一步固化(Top1 覆盖 5.3% vs 1.3%, Top5 12.3% vs 6.7%)——
  RL 未修复多样性坍缩, 反而加剧了开头句式的集中。

# 7. 备忘

```bash
make games-sidreasoner-rl-data                          # 从当前 CSV+index 重建 RL parquet(切 SID 后自动生效, 无需手动)
make games-sidreasoner-merge EVAL_INTERVAL=100          # 只合并特定步数间隔
make games-sidreasoner-sft CUDA_VISIBLE_DEVICES=0,1,2,3 # 指定训练用卡(不指定则自动选空闲卡)
make tmux TARGET=games-sidreasoner-sft  # 挂 tmux 后台跑长任务(断开 SSH 不中断; 卡自动挑选)
make games-sidreasoner-eval CUDA_LIST="0 1"             # 指定评估用卡(默认 0 1)
DATASET=games bash scripts/sidreasoner/sft.sh           # 等价的手动运行方式
DATASET=games .venv/bin/python -m src.train.tiger
```

- 训练数据已随仓库就位(`data/raw/Amazon/`), 无需下载。
- Stage 1/2 输出在 `runs/sidreasoner/`, Stage 3 在 `checkpoints/RecRL_Reasoning/`, 评估结果在 `results/sidreasoner/`。
- 8 种对齐任务的 prompt 模板与 GPT-4o-mini 语料增强 prompt 见 `docs/sidreasoner_prompts.md`。

## 常见问题(SIDReasoner)

- **无外网服务器启动即报 `Errno 97 ... huggingface.co`**: `model.base_model`
  为 HF id 时启动会尝试联网下载。解决: 经镜像下载到本地并改 config 指向
  本地路径(`export HF_ENDPOINT=https://hf-mirror.com` 后执行
  `huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B`,
  再把 `model.base_model` 改为 `"./models/Qwen3-1.7B"`);
  可再加 `export HF_HUB_OFFLINE=1` 杜绝残余联网探测。
- **wandb 连不上**: 同因无外网。把 config 中 sft/activation 的
  `report_to` 改为 `"none"`, 或 `export WANDB_MODE=offline`。
- **torchrun 开头 `socket.cpp:759 Address family not supported`**:
  本机 IPv6 探测警告, 自动回落 IPv4, 无害可忽略。
- **Stage 3 报 `cannot import name 'AutoModelForVision2Seq'`(或 `TrainingArguments` 没有
  `group_by_length` 参数)**: transformers 被依赖解析装成了 5.x, 而官方 verl fork 与训练
  代码需要 4.x。**必须整族同代降级**(只降 transformers 会连锁触发 huggingface-hub /
  datasets 的 `ImportError`):
  ```bash
  .venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
      "transformers==4.51.3" "tokenizers==0.21.1" "huggingface-hub==0.30.2" \
      "datasets==3.5.1" "accelerate==1.6.0" "peft==0.15.0"
  ```
  之后用 `python -c "from transformers import AutoModelForVision2Seq; import datasets,
  transformers, accelerate, peft, vllm; print('all ok')"` 验证。
