"""跨 Query CoT 多样性奖励编排(In-batch Top-K 相似用户 + TD + DIVER 塑形)。

流程(对一次 reward 调用收到的整个 batch, B 个 query x G 个 rollout):
1. 按 (split, index) 把 rollout 分组到 query, 拒绝跨 query 混组;
2. 从离线缓存查每个 query 的用户向量与历史物品集合;
3. 排除自身与同用户(历史物品集合 Jaccard >= same_user_jaccard),
   余弦相似度 top-K 选出相似用户 N_K(q);
4. 每个邻居从其 G 个 rollout 中随机选一个"推荐成功"(SID 三级全匹配)
   的 CoT 作为 Reference, 无成功 rollout 的邻居不作为有效 Reference;
5. 对当前 query 的每个 rollout 独立计算
   d(o_{q,i}) = mean_{q' in valid} (1 - sim_BLEU(c_{q,i}, c~_{q'})),
   K_q = 0 时 d = 0;
6. DIVER 风格塑形:
   R' = R + lambda(t) * I * clip(gamma^T * d, 0, sigma),
   仅对 split == 'train' 的 batch 生效(验证集只算原始 reward)。

lambda(t) 用的训练步数由本模块内部计数(每次 train batch 调用 +1,
验证调用不计数); 进程重启后从 0 重新计数(无 checkpoint 语义, 已知
限制, 见 README/交付说明)。
"""

import json
import os
import random
import time

import numpy as np

from .bleu import DiverBleu
from .cot_utils import extract_cot, recommendation_match
from .neighbor import build_same_user_mask, topk_neighbors
from .shaping import lambda_at, rank_normalize, shape_rewards
from .similarity import NgramJaccard
from .user_store import UserEmbeddingStore

#: 参与比较的 CoT 最少 whitespace token 数(低于此值 td 记 0,
#: 防止用退化空 CoT 刷多样性; DIVER 的 5-gram BLEU 对过短文本本就
#: 无意义)
_MIN_COT_TOKENS = 5


class InterQueryDiversityAddon:
    """可插在任意逐行 base reward 之上的跨用户多样性奖励。"""

    def __init__(
        self,
        user_embedding_path: str,
        top_k_neighbors: int = 8,
        same_user_jaccard: float = 0.8,
        bleu_div_len: int = 1000,
        lambda_init: float = 0.2,
        lambda_final: float = 0.0,
        decay_steps: int = 300,
        gamma: float = 1.0,
        sigma: float = 0.5,
        model_path: str = None,
        metrics_path: str = None,
        log_every: int = 1,
        seed: int = 0,
        device: str = "cpu",
        success_mode: str = "full",
        reference_mode: str = "success",
        bleu_smoothing: str = "none",
        td_norm: str = "none",
        similarity: str = "bleu",
        ngram_n: int = 1,
    ):
        """device 仅用于 BxB 邻居矩阵乘, 默认 cpu: reward 在 verl driver
        进程执行, 默认用 cuda 会在 colocated GPU 上建 context 挤占显存。

        success_mode: 条件塑形指示 I 的口径("full" 三级全匹配 /
            "prefix" 首层匹配 / "none" 不设正确性约束)。
        reference_mode: 邻居 Reference 的准入档位("full" 仅三级全匹配 /
            "prefix" 首层 code 匹配即可 / "none" 退回任一可解析 CoT,
            优先级 full > prefix > any)。低成功率阶段(如 Stage 2 起点)
            默认口径会让 intrinsic 恒 0, 可逐级放宽。
        bleu_smoothing: "none"(DIVER 官方) / "method1"(add-1 平滑,
            5-gram 不再几乎恒为 0, TD 有真实区分度)。
        td_norm: "none" / "rank"。rank = 把每个 GRPO 组内(仅 I=1 的
            rollout)的 TD 按秩映射到 [0, 1] 均布后再塑形。TD 的组内
            绝对差通常极小(~0.01), 直接相加对 GRPO 组内排序几乎无影响;
            秩归一化保证组内区分度由 λ·σ 直接控制(均值固定, 无常数平移)。
        similarity: 多样性度量。"bleu" = SymBLEU(DIVER 官方口径, 含
            bleu_smoothing 选项); "ngram_jaccard" = 1 - n-gram 集合
            Jaccard(|G_n(a)∩G_n(b)| / |G_n(a)∪G_n(b)|)。
        ngram_n: ngram_jaccard 的 n(默认 1)。
        """
        self.store = UserEmbeddingStore(user_embedding_path)
        self.top_k = int(top_k_neighbors)
        self.same_user_jaccard = float(same_user_jaccard)
        self.similarity = str(similarity)
        if self.similarity == "bleu":
            self.scorer = DiverBleu(
                div_len=int(bleu_div_len), smoothing=bleu_smoothing
            )
        elif self.similarity == "ngram_jaccard":
            self.scorer = NgramJaccard(n=int(ngram_n), div_len=int(bleu_div_len))
        else:
            raise ValueError(
                f"未知 similarity: {similarity!r}(bleu | ngram_jaccard)"
            )
        self.success_mode = str(success_mode)
        if self.success_mode not in ("full", "prefix", "none"):
            raise ValueError(
                f"未知 success_mode: {success_mode!r}(full | prefix | none)"
            )
        # 三档: full=邻居三级全匹配 / prefix=邻居首层匹配 / none=退回任一
        # 可解析 CoT(优先级 full > prefix > any); 兼容旧别名 success/any
        ref_alias = {"success": "full", "any": "none"}
        self.reference_mode = ref_alias.get(str(reference_mode), str(reference_mode))
        if self.reference_mode not in ("full", "prefix", "none"):
            raise ValueError(
                f"未知 reference_mode: {reference_mode!r}"
                "(full | prefix | none; 旧别名 success/any 亦兼容)"
            )
        self.td_norm = str(td_norm)
        if self.td_norm not in ("none", "rank"):
            raise ValueError(f"未知 td_norm: {td_norm!r}(none | rank)")
        self.lambda_init = float(lambda_init)
        self.lambda_final = float(lambda_final)
        self.decay_steps = int(decay_steps)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.model_path = model_path
        self.device = device
        self.log_every = max(int(log_every), 1)
        self.rng = random.Random(seed)

        if metrics_path is None:
            metrics_path = os.path.join(
                os.path.dirname(os.path.abspath(user_embedding_path)),
                "interdiv_metrics.jsonl",
            )
        self.metrics_path = metrics_path
        self._train_steps = 0
        self._tokenizer = None

    # ------------------------------------------------------------------ #

    def _tokenizer_token_count(self, text: str) -> int:
        """gamma < 1 时才需要; 从 Stage-2 checkpoint 懒加载 tokenizer。"""
        if self.gamma == 1.0:
            return 1
        if self._tokenizer is None:
            if not self.model_path:
                raise ValueError(
                    "gamma < 1 需要逐 rollout token 数, 请在 reward_kwargs 里"
                    "传 model_path(Stage-2 checkpoint)或保持 gamma=1.0。"
                )
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _query_key(extra_info: dict):
        if not isinstance(extra_info, dict):
            return None
        if "index" not in extra_info or "split" not in extra_info:
            return None
        return (str(extra_info["split"]), int(extra_info["index"]))

    # ------------------------------------------------------------------ #

    def compute_batch(self, rows: list, base_rewards: list) -> list:
        """对整个 batch 计算增广 reward。

        Args:
            rows: 每条 rollout 的 {data_source, solution_str, ground_truth,
                extra_info}(来自 verl BatchRewardManager)。
            base_rewards: 逐行原始推荐 reward(保持不变)。

        Returns:
            长度 == len(rows) 的 list[dict]: {"score": 增广 reward,
            "base_reward", "success", "td", "intrinsic", ...}
        """
        n = len(rows)
        results = [
            {
                "score": float(base_rewards[i]),
                "base_reward": float(base_rewards[i]),
                "success": 0,
                "td": 0.0,
                "intrinsic": 0.0,
                "n_refs": 0,     # 该 query 实际可用的邻居 Reference 数 K_q
                "cot_ok": 0,     # 本 rollout 的 CoT 是否可解析(</think> 且 >=5 token)
            }
            for i in range(n)
        ]
        if n == 0:
            return results

        groups: dict = {}
        for i, row in enumerate(rows):
            key = self._query_key(row.get("extra_info"))
            if key is None:
                raise ValueError(
                    f"第 {i} 条 rollout 的 extra_info 缺少 split/index, 无法对齐"
                    " user embedding 缓存(parquet 需带 extra_info)。"
                )
            groups.setdefault(key, []).append(i)

        train_keys = [k for k in groups if k[0] == "train"]
        is_train_batch = len(train_keys) == len(groups) and len(train_keys) > 0
        if not is_train_batch or len(train_keys) < 2:
            # 验证/测试 batch 或 query 数不足以构成邻居: 只回原始 reward
            return results

        t_start = time.time()
        step = self._train_steps
        lam = lambda_at(step, self.lambda_init, self.lambda_final, self.decay_steps)

        # ---- 用户表征与邻居检索(按去重后的 query) ----
        embeddings, histories, user_ids = self.store.lookup(train_keys)
        import torch

        user_emb = torch.as_tensor(embeddings, dtype=torch.float32)
        history_sets = [frozenset(int(x) for x in h) for h in histories]
        exclude = build_same_user_mask(
            history_sets, self.same_user_jaccard, user_ids=user_ids
        )
        neighbor_idx, neighbor_sim = topk_neighbors(
            user_emb, self.top_k, exclude=exclude, device=self.device
        )

        # ---- 逐 query: 匹配度 / 成功指示 / reference CoT ----
        parsed_flags: dict = {}
        prefix_lens: dict = {}
        success_flags: dict = {}
        cots: dict = {}
        for q_i, key in enumerate(train_keys):
            for r in groups[key]:
                row = rows[r]
                parsed, plen = recommendation_match(
                    row["solution_str"], row["ground_truth"]
                )
                parsed_flags[r] = parsed
                prefix_lens[r] = plen
                if self.success_mode == "none":
                    ok = parsed
                elif self.success_mode == "prefix":
                    ok = parsed and plen >= 1
                else:  # full
                    ok = parsed and plen >= 3
                results[r]["success"] = int(ok)
                results[r]["prefix_len"] = int(plen)
                success_flags[r] = ok
                cots[r] = extract_cot(row["solution_str"])
                results[r]["cot_ok"] = int(
                    cots[r] is not None
                    and len(cots[r].split()) >= _MIN_COT_TOKENS
                )

        def _usable(r) -> bool:
            return results[r]["cot_ok"] == 1

        def pick_reference(row_ids: list):
            """从某 query 的 rollouts 里挑一个参考 CoT。

            reference_mode=full:   只挑三级全匹配的 CoT;
            reference_mode=prefix: 首层 code 匹配即可;
            reference_mode=none:   退回任一可解析 CoT(优先级 full > prefix > any)。
            """
            full_pool = [cots[r] for r in row_ids if _usable(r) and prefix_lens[r] >= 3]
            prefix_pool = [
                cots[r] for r in row_ids if _usable(r) and prefix_lens[r] >= 1
            ]
            any_pool = [cots[r] for r in row_ids if _usable(r)]
            if self.reference_mode == "full":
                pool = full_pool
            elif self.reference_mode == "prefix":
                pool = prefix_pool
            else:  # none: 逐级退回
                pool = full_pool or prefix_pool or any_pool
            return self.rng.choice(pool) if pool else None

        references = {
            q_i: pick_reference(groups[key]) for q_i, key in enumerate(train_keys)
        }

        # ---- 逐 query 逐 rollout 计算跨用户 TD ----
        sim_np = neighbor_sim.numpy()
        sims_finite = sim_np[np.isfinite(sim_np)]
        td_values = np.zeros(n, dtype=np.float64)
        valid_neighbor_counts = []
        for q_i in range(len(train_keys)):
            valid_refs: list = []
            for slot in range(neighbor_idx.shape[1]):
                nb = int(neighbor_idx[q_i, slot])
                if nb < 0:
                    continue
                ref = references.get(nb)
                if ref is not None:
                    valid_refs.append(ref)
            valid_neighbor_counts.append(len(valid_refs))
            for r in groups[train_keys[q_i]]:
                results[r]["n_refs"] = len(valid_refs)
            if not valid_refs:
                continue
            for r in groups[train_keys[q_i]]:
                cot = cots[r]
                if cot is None or len(cot.split()) < _MIN_COT_TOKENS:
                    continue
                d = float(
                    np.mean(
                        [self.scorer.diversity(cot, ref) for ref in valid_refs]
                    )
                )
                td_values[r] = d

        # 保留原始 TD 供回报/诊断(intrinsic 用可能归一化后的值)
        td_raw = td_values.copy()

        # ---- 可选: 组内秩归一化(仅重排"可塑形"的 rollout, 保证 GRPO 组内
        # 区分度; 均值固定为 0.5 分布, 不引入组内常数平移) ----
        if self.td_norm == "rank":
            for key in train_keys:
                rows_q = groups[key]
                eligible = [
                    r for r in rows_q
                    if success_flags[r] and results[r].get("cot_ok", 0)
                ]
                m = len(eligible)
                if m < 2:
                    for r in eligible:
                        td_values[r] = 0.0
                    continue
                vals = [td_values[r] for r in eligible]
                for r, v in zip(eligible, rank_normalize(vals)):
                    td_values[r] = float(v)

        # ---- 塑形(只作用于本 batch 的 train 行) ----
        train_rows = [r for key in train_keys for r in groups[key]]
        base_arr = np.array([base_rewards[r] for r in train_rows], dtype=np.float64)
        success_arr = np.array([success_flags[r] for r in train_rows], dtype=bool)
        td_arr = td_values[train_rows]
        if self.gamma == 1.0:
            token_counts = None
        else:
            token_counts = [
                self._tokenizer_token_count(rows[r]["solution_str"]) for r in train_rows
            ]
        intrinsic, augmented = shape_rewards(
            base_arr,
            success_arr,
            td_arr,
            token_counts=token_counts,
            gamma=self.gamma,
            sigma=self.sigma,
            lam=lam,
        )
        for j, r in enumerate(train_rows):
            results[r]["td"] = float(td_raw[train_rows][j])       # 原始 TD(诊断)
            results[r]["td_used"] = float(td_arr[j])              # 塑形实际用值
            results[r]["intrinsic"] = float(intrinsic[j])
            results[r]["score"] = float(augmented[j])

        self._train_steps += 1
        self._log_metrics(
            step=step,
            lam=lam,
            n_queries=len(train_keys),
            n_rollouts=len(train_rows),
            base_rewards=base_arr,
            success_arr=success_arr,
            td_arr=td_raw[train_rows],
            intrinsic=intrinsic,
            augmented=augmented,
            neighbor_sim_mean=float(np.nanmean(sims_finite)) if sims_finite.size else 0.0,
            valid_neighbors_mean=float(np.mean(valid_neighbor_counts))
            if valid_neighbor_counts
            else 0.0,
            elapsed=time.time() - t_start,
        )
        return results

    # ------------------------------------------------------------------ #

    def _log_metrics(self, **m):
        success = m["success_arr"]
        td = m["td_arr"]
        record = {
            "step": m["step"],
            "lambda": m["lam"],
            "n_queries": m["n_queries"],
            "n_rollouts": m["n_rollouts"],
            "success_mode": self.success_mode,
            "reference_mode": self.reference_mode,
            "similarity": self.similarity,
            "bleu_smoothing": getattr(self.scorer, "smoothing", None),
            "ngram_n": getattr(self.scorer, "n", None),
            "td_norm": self.td_norm,
            "base_reward_mean": float(np.mean(m["base_rewards"])),
            "augmented_reward_mean": float(np.mean(m["augmented"])),
            "success_rate": float(np.mean(success)) if len(success) else 0.0,
            "td_mean": float(np.mean(td)) if len(td) else 0.0,
            "td_success_mean": float(np.mean(td[success])) if success.any() else 0.0,
            "intrinsic_mean": float(np.mean(m["intrinsic"])) if len(m["intrinsic"]) else 0.0,
            "neighbor_sim_mean": m["neighbor_sim_mean"],
            "valid_neighbors_mean": m["valid_neighbors_mean"],
            "elapsed_s": round(m["elapsed"], 3),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.metrics_path)) or ".", exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if record["step"] % self.log_every == 0:
            print(
                "[interdiv] "
                f"step={record['step']} lam={record['lambda']:.4f} "
                f"base={record['base_reward_mean']:.4f} "
                f"aug={record['augmented_reward_mean']:.4f} "
                f"success={record['success_rate']:.3f} "
                f"td={record['td_mean']:.4f} "
                f"td_ok={record['td_success_mean']:.4f} "
                f"intr={record['intrinsic_mean']:.4f} "
                f"nbrs={record['valid_neighbors_mean']:.2f} "
                f"sim={record['neighbor_sim_mean']:.4f} "
                f"({record['elapsed_s']:.1f}s)",
                flush=True,
            )
