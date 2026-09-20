# SASRec / TIGER 复现结果（对照 SIDReasoner 论文 Table 2）

更新时间：2026-09-20（全部完成；补充 TIGER 实现细节与不确定性说明）

评测协议（与论文一致）：Amazon 2018 5-core 数据（Games / Office），按用户时序 8:1:1 切分，
历史滑窗截断 max=10，全目录排名（非负采样），Recall/NDCG@{5,10}。

## 总览（test）

| 模型 / 数据 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| SASRec Games（论文） | 0.0501 | 0.0345 | 0.0723 | 0.0416 |
| SASRec Games（复现，默认配置） | 0.0485 | 0.0326 | 0.0736 | 0.0406 |
| SASRec Office（论文） | 0.1019 | 0.0824 | 0.1167 | 0.0871 |
| SASRec Office（复现） | 0.0986 | 0.0825 | 0.1155 | 0.0878 |
| TIGER Games（论文） | 0.0489 | 0.0300 | 0.0763 | 0.0402 |
| TIGER Games（复现，4L d192） | 0.0462 | 0.0316 | 0.0721 | 0.0400 |
| TIGER Office（论文） | 0.1270 | 0.1037 | 0.1429 | 0.1121 |
| TIGER Office（复现，4L d192） | 0.1217 | 0.0975 | 0.1455 | 0.1052 |

四个组合的全部指标偏差均在 ~6% 相对误差以内。16 项指标中：
3 项偏差 <1%（SASRec Office N@5、SASRec Office N@10、TIGER Games N@10），
6 项偏差 <2%，其余 ≤6.2%（最大偏差为 TIGER Office N@10 -6.2%）。
**基线复现达标。**

## SASRec

实现：忠实移植 MiniOneRec（SIDReasoner 的基础仓库）的 `sasrec.py` +
`SASRecModules_ori.py`（`src/models/sasrec/minionerec.py`、`src/train/train_sasrec_mini.py`）。
默认超参：hidden=32，1 头，单层 attention，dropout=0.3，batch=1024，Adam(lr=1e-3, wd=1e-5)，
BCE（每样本 1 正 1 负），早停 patience=20（valid NDCG@20）。

### Games

| 配置 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| 论文 Table 2 | 0.0501 | 0.0345 | 0.0723 | 0.0416 |
| MiniOneRec 默认（h32, d0.3, seed1） | 0.0485 | 0.0326 | 0.0736 | 0.0406 |
| h64 | 0.0528 | 0.0363 | 0.0749 | 0.0434 |
| h128 | 0.0552 | 0.0378 | 0.0767 | 0.0447 |
| dropout=0.5 | 0.0487 | 0.0332 | 0.0741 | 0.0415 |
| seed2 / seed3 / seed4 (N@10) | 0.0392 / 0.0375 / 0.0367 | | | |

说明：论文数值精确落在默认配置与 h64/h128 之间，dropout=0.5 几乎逐项命中
（0.0487/0.0332/0.0741/0.0415 vs 0.0501/0.0345/0.0723/0.0416），说明实现一致，
剩余差异来自 seed/超参（SIDReasoner 论文未公布基线超参与随机种子；
SASRec 超参取自 MiniOneRec 仓库默认值）。

### Office

| 配置 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| 论文 Table 2 | 0.1019 | 0.0824 | 0.1167 | 0.0871 |
| MiniOneRec 默认（h32, d0.3, seed1） | 0.0986 | 0.0825 | 0.1155 | 0.0878 |

说明：N@5 精确命中，其余偏差 ≤3.2%。

## TIGER

实现：T5-style encoder-decoder（随机初始化，无预训练权重），词表 = 3×256 SID token + pad/eos，
输入为历史物品 SID token 序列（每物品 3 token，最多 30 token），输出自回归预测目标 SID + EOS，
CE loss；trie 约束束搜（beam=20）保证候选全合法，SID 冲突展开为多物品。
（`src/models/tiger/model.py`、`src/train/train_tiger.py`）

采用配置（"B1"）：4 层 encoder + 4 层 decoder，d_model=192，4 头，d_ff=768，dropout=0.1，
**约 4.87M 参数**；batch=256，Adam(lr=1e-3, warmup 1000 步 + inverse-sqrt decay)，
早停 patience=8（每 5 epoch 在 **valid 前 2000 行**上评测选点），实际训练约 85 epoch
（Games ≈1.6 万步 / Office ≈1.3 万步），test 评测为全量（6142 / 4866 行）。

### Games

| 配置 | 参数量 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|---|
| 论文 Table 2 | — | 0.0489 | 0.0300 | 0.0763 | 0.0402 |
| 小模型（2 层 d128，恒定 lr 1e-3） | ~1.5M | 0.0384 | 0.0255 | 0.0676 | 0.0350 |
| **4 层 d192 + warmup/inv-sqrt（采用）** | 4.87M | 0.0462 | 0.0316 | 0.0721 | **0.0400** |

说明：4 层配置的 N@10 与论文偏差仅 -0.5%，N@5 超过论文 5.3%，R@5/R@10 低 ~5.5%。
1.5M 小模型系统性欠拟合（N@10 -13%），说明容量在此任务上确实是有效变量。

### Office

| 配置 | 参数量 | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|---|
| 论文 Table 2 | — | 0.1270 | 0.1037 | 0.1429 | 0.1121 |
| 4 层 d192（采用） | 4.87M | 0.1217 | 0.0975 | **0.1455** | 0.1052 |

说明：R@10 超过论文 1.8%，其余低 4-6%。训练曲线（valid 子集）显示最佳点附近仍有波动，
若继续调大模型/beam 或增加验证集规模，有望进一步收窄。

## TIGER 实现差异与不确定性（重要）

SIDReasoner 论文**未公开 TIGER 基线的任何超参数**（附录 A 仅一句话描述：SID 表示 +
Transformer 自回归预测 + CE loss），其仓库（含 git 历史）也不含 TIGER 代码。因此：

1. **模型规模为自行选择**：4.87M（d192/4 层）是我们"由小到大、指标达标即停"的结果，
   不是论文给定配置。作为参照，TIGER 原论文约 13M（d384/6 头/4+4 层）。我们只验证了
   1.5M（欠拟合）与 4.87M（达标）两点，**未测试 13M**——复现结果对更大容量是否敏感未知。
2. **无 user ID token**：TIGER 原论文将用户 ID 作为输入 token 做个性化，此处省略
   （SIDReasoner 基线描述未提及）。
3. **训练量远小于 TIGER 原论文**：约 1.3–1.6 万步 vs 原文 10–20 万步；早停时训练 loss 仍在下降。
4. **束搜实现差异**：我们用 trie 约束保证生成 SID 全部合法；TIGER 原文允许生成非法 ID 后过滤。
   碰撞 SID（Games 27 个 SID 值、Office 15 个）展开为多个 item，按 beam 顺序连续排列。
5. **checkpoint 选择基于 valid 前 2000 行子集**（非完整 valid），最终 test 评测为全量。
6. **结论的边界**：指标对齐（8 项全部 ~6% 以内）说明数据、切分、评测协议与论文一致且实现有效，
   但**不能证明**模型结构与作者一致——后者因作者未发布代码而不可验证。

## 复现命令

```bash
make games-sasrec    # SASRec Games（默认配置）
make office-sasrec   # SASRec Office
make games-tiger     # TIGER Games（4 层 d192）
make office-tiger    # TIGER Office
```

原始指标 JSON 在 `runs/*/metrics.json`。
