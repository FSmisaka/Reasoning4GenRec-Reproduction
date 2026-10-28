"""
SIDReasoner 终端把玩入口(Mac MPS / CPU / 服务器 GPU 均可)

像网页 AI 一样与三个 Stage 的 checkpoint 对话, 直观感受各阶段能力:
- Stage 1: /think off 后直答 SID(/t1 /t2 验证 SID<->标题翻译)
- Stage 2: /think on 先 <think> 推理再答 SID(教师腔, 内容较套路)
- Stage 3: 同格式, 推理更自主、SID 更准(/score 按论文规则逐级打分)

推荐任务(/t3 及 /sys rec 下的自由输入)默认走"评测复刻"解码, 与
make *-sidreasoner-eval / *-sidreasoner-think 等价:
- /think off: 空思考块硬拼进 token 序列 + trie 约束束搜索
  (每步只允许合法 SID 前缀的下一 token, 思维链在解码层面不可能出现)
- /think on : 先自由贪心生成思维链, 截断到 </think> 拼回 prompt,
  再做同样的约束束搜索(对应思考评测的两阶段)
/constrain off 可退回旧的自由生成模式。

启动(checkpoint 目录需就位, 拷贝方法见 README 本地把玩一节):
  DATASET=games MODEL=<checkpoint路径> .venv/bin/python -m src.eval.sidreasoner_chat
  make games-sidreasoner-chat MODEL=<checkpoint路径>   # 等价
会话内输入 /help 查看全部命令; --prompt "..." 可单轮非交互运行(冒烟测试用)。
"""

import argparse
import os
import random
import sys
import threading
import time

import torch

from src.reward.direct_recommendation_steprule import (
    calculate_format_reward,
    calculate_reward,
    extract_sid_tokens,
)

REC_SYSTEM = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\nCan you recommend the next item for the user "
    "based on their interaction history?\n"
)
SID_SYSTEM = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request. \nAnswer the question about item identification."
)
# 非思考评测(EvalSidDataset)的 system, 逐字迁移(含 "request. " 后的空格)
PRED_SYSTEM = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request. \nCan you predict the next possible item that "
    "the user may expect?\n"
)

HELP = """\
命令(直接输入文本则作为 user 消息发给模型):
  /t1             随机真实商品: 问 SID -> 标题(自由生成, 验证 Stage 1 翻译)
  /t2             随机真实商品: 问标题 -> SID(标准答案自动记录, 之后可 /score)
  /t3 [s1,s2,...] 推荐任务: 给定(缺省随机)SID 历史 -> 评测复刻解码输出 top-N SID
  /score [SID]    按论文规则给上一条回答打分; top-N 输出时报命中排名
  /sid [n]        随机展示 n 条真实 SID 与标题(自己出题用)
  /think on|off   评测复刻模式的两档: off=空思考块+约束束搜索直答 SID
                  (思维链被解码器禁止); on=自由思维链 -> </think> -> 约束束搜索
  /constrain on|off  评测复刻解码开关(默认 on; off 退回自由生成,
                     此时 /think off 不再能强制阻断思维链)
  /beams N        束宽(默认取 config evaluation.num_beams=10, 与评测一致)
  /sys rec|id|pred 自由输入的 system 提示: rec=Stage2/3 训练与思考评测,
                  pred=非思考评测(Case 5 措辞, 配合手动输入其 user 模板),
                  两者均走评测复刻解码; id=物品识别(自由生成)
  /multi on|off   多轮对话开关(默认 off: 每条输入独立单轮, 贴近训练分布)
  /temp T         自由生成的采样温度(0 = 贪心; 评测复刻路径恒为贪心)
  /reset          清空多轮历史
  /help /exit
"""

NO_DATA_HINT = (
    "未设置 DATASET 环境变量, 此命令不可用; "
    "用 DATASET=games|office|industrial 重新启动"
)


def resolve_info_path():
    if not os.environ.get("DATASET"):
        return None
    try:
        from src.reward.direct_recommendation_steprule import _resolve_info_path

        path = _resolve_info_path()
        return path if os.path.exists(path) else None
    except Exception:
        return None


def load_info_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and len(extract_sid_tokens(parts[0])) == 3:
                items.append((parts[0].strip(), parts[1].strip()))
    return items


def get_hash(x):
    x = [str(_) for _ in x]
    return "-".join(x)


def find_last_sublist(lst, sub):
    if not sub:
        return None
    n, m = len(lst), len(sub)
    for start in range(n - m, -1, -1):
        if lst[start : start + m] == sub:
            return start
    return None


def build_sid_trie(tokenizer, info_path):
    """trie 约束(迁移自 sidreasoner_eval(_think).py):
    对 "</think>\\n\\n" 之后已生成的 token 序列, 建立合法下一 token 的前缀树。"""
    prefix_prompt = "</think>\n\n"
    with open(info_path, "r", encoding="utf-8") as f:
        semantic_ids = [line.split("\t")[0].strip() for line in f]
    sep_ids = tokenizer(prefix_prompt, add_special_tokens=False)["input_ids"]
    prefix_index = len(sep_ids)
    eos = tokenizer.eos_token_id
    hash_dict = {}
    for sid in semantic_ids:
        ID = tokenizer(f"{prefix_prompt}{sid}\n").input_ids + [eos]
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                key = get_hash(ID[:i])
            else:
                key = get_hash(ID[prefix_index:i])
            hash_dict.setdefault(key, set()).add(ID[i])
    return {k: list(v) for k, v in hash_dict.items()}, sep_ids


def load_eval_cfg():
    defaults = {"num_beams": 10, "max_new_tokens": 256, "length_penalty": 0.0}
    try:
        from config import load_config

        evaluation = load_config("sidreasoner")["evaluation"]
        defaults.update(
            {k: evaluation[k] for k in defaults if k in evaluation}
        )
    except Exception:
        pass
    return defaults


class Session:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = args.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if args.dtype == "auto":
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            dtype = getattr(torch, args.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=dtype, low_cpu_mem_usage=True
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=dtype, low_cpu_mem_usage=True
            )
        self.model.to(device).eval()

        self.device, self.dtype = device, dtype
        self.args = args
        self.think = args.think == "on"
        self.multi = False
        self.default_system = REC_SYSTEM
        self.temperature = args.temperature
        self.history = []
        self.last_reply = None
        self.last_gt = None
        self.last_reply_list = None
        self.eval_cfg = load_eval_cfg()
        self.constrain = True
        self.trie = None
        self.sep_ids = None
        self.sid2title = {}
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model, "generation_config"):
            for k in ("temperature", "top_p"):
                setattr(self.model.generation_config, k, None)

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            chat_template_kwargs={"enable_thinking": self.think},
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if self.args.max_new_tokens > 0:
            budget = self.args.max_new_tokens
        else:
            limits = [
                m
                for m in (
                    getattr(self.model.config, "max_position_embeddings", None),
                    getattr(self.tokenizer, "model_max_length", None),
                )
                if isinstance(m, int) and 0 < m < 10**9
            ]
            cap = min(limits) if limits else 8192
            budget = max(cap - enc.input_ids.shape[1], 1)
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **enc,
            max_new_tokens=budget,
            streamer=streamer,
            do_sample=self.temperature > 0,
        )
        if self.temperature > 0:
            gen_kwargs.update(temperature=self.temperature, top_p=self.args.top_p)
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        chunks = []
        start = time.time()
        for chunk in streamer:
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        thread.join()
        print()
        reply = "".join(chunks)
        dt = time.time() - start
        n_new = len(self.tokenizer(reply, add_special_tokens=False)["input_ids"])
        print(
            f"[{n_new} tokens, {dt:.1f}s, {n_new / max(dt, 1e-6):.1f} tok/s]"
        )
        self.last_reply = reply
        self.last_reply_list = None
        return reply

    def send(self, user_text, system=None):
        if system is None:
            system = self.default_system
        if self.multi and self.history:
            messages = self.history + [{"role": "user", "content": user_text}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        if self.constrain and self.trie is not None and system != SID_SYSTEM:
            reply = self.send_eval(messages)
        else:
            reply = self.generate(messages)
        if self.multi:
            self.history = messages + [{"role": "assistant", "content": reply}]
        return reply

    def setup_eval(self, info_path, items):
        """构建评测复刻所需的 trie 约束与 SID->标题表。"""
        self.sid2title = {sid: title for sid, title in items}
        self.trie, self.sep_ids = build_sid_trie(self.tokenizer, info_path)
        print(
            f"eval 复刻就绪: trie 约束({len(self.sid2title)} 条 SID, "
            f"num_beams={self.eval_cfg['num_beams']})"
        )

    def _prefix_allowed_tokens(self, batch_id, input_ids):
        """约束回调(迁移自 sidreasoner_eval.py): 只允许延续合法 SID
        前缀的 token, 其余 token 在每一步解码时被屏蔽。"""
        ids = input_ids.tolist()
        pos = find_last_sublist(ids, self.sep_ids)
        if pos is None:
            return [self.tokenizer.eos_token_id]
        after = ids[pos + len(self.sep_ids) :]
        key = get_hash(after) if after else get_hash(self.sep_ids)
        return self.trie.get(key, [self.tokenizer.eos_token_id])

    def beam_search_sids(self, prompt_ids):
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        t0 = time.time()
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.eval_cfg["max_new_tokens"],
                num_beams=self.eval_cfg["num_beams"],
                num_return_sequences=self.eval_cfg["num_beams"],
                early_stopping=True,
                length_penalty=self.eval_cfg["length_penalty"],
                prefix_allowed_tokens_fn=self._prefix_allowed_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        outs = self.tokenizer.batch_decode(
            gen[:, input_ids.shape[1] :], skip_special_tokens=True
        )
        outs = [o.split("Response:\n")[-1].strip() for o in outs]
        return outs, time.time() - t0

    def send_eval(self, messages):
        """评测复刻路径(与 make *-sidreasoner-eval / -think 等价的解码):
        - /think off: "<think>\\n</think>\\n\\n" 硬拼进 token 序列后直接束搜索
          (对应非思考评测, 思维链不可能出现)
        - /think on : 自由贪心生链 -> 截到 </think> 拼回 -> 束搜索 SID
          (对应思考评测的两阶段)
        """
        base_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        base_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self.think:
            print("— 自由思维链(贪心, 对应 think 评测第一阶段)— ")
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=False
            )
            enc = torch.tensor([base_ids], device=self.device)
            thread = threading.Thread(
                target=self.model.generate,
                kwargs=dict(
                    input_ids=enc,
                    attention_mask=torch.ones_like(enc),
                    max_new_tokens=self.eval_cfg["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                ),
            )
            thread.start()
            chunks = []
            for chunk in streamer:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            thread.join()
            print()
            # 与 sidreasoner_eval_think.py 一致: 拼回后截到 </think>
            # (无 </think> 时丢末字符再强制闭合, 保留官方行为)
            text = base_text + "".join(chunks).strip()
            idx = text.find("</think>")
            text = (text[:idx] if idx != -1 else text[:-1]).strip()
            text += "\n</think>\n\n"
            prompt_ids = self.tokenizer(
                text, truncation=True, max_length=2560
            )["input_ids"]
        else:
            prompt_ids = list(base_ids) + self.tokenizer.encode(
                "<think>\n</think>\n\n"
            )
        print(
            f"— trie 约束束搜索(num_beams={self.eval_cfg['num_beams']}, "
            "贪心确定性)— "
        )
        outs, dt = self.beam_search_sids(prompt_ids)
        for rank, sid in enumerate(outs, 1):
            print(f"  {rank:>2}. {sid}  {self.sid2title.get(sid, '')}")
        print(f"[束搜索 {dt:.1f}s]")
        self.last_reply = outs[0] if outs else ""
        self.last_reply_list = outs
        return self.last_reply

    def score(self, gt_sid, prefix_map):
        if not self.last_reply:
            print("还没有生成过回答, 先发一条消息或 /t1 /t2 /t3")
            return
        gt = extract_sid_tokens(gt_sid)[:3]
        if self.last_reply_list:
            # 评测复刻的 top-N 输出: 报 top-1 步进分 + 命中排名(同 HR@K 口径)
            top1 = extract_sid_tokens(self.last_reply_list[0])[:3]
            rank = next(
                (
                    i
                    for i, sid in enumerate(self.last_reply_list, 1)
                    if extract_sid_tokens(sid)[:3] == gt
                ),
                None,
            )
            parts = []
            if len(top1) == 3:
                step = calculate_reward(top1, gt)
                parts.append(f"top-1 {top1} 步进分 {step:.2f}")
                if prefix_map is not None:
                    fmt = 0.1 * float(calculate_format_reward(top1, prefix_map))
                    parts.append(f"格式分 {fmt:.2f}")
            else:
                parts.append(f"top-1 无法解析: {self.last_reply_list[0]!r}")
            if rank is not None:
                parts.append(
                    f"标准 {gt} 命中排名 {rank}/{len(self.last_reply_list)}"
                )
            else:
                parts.append(
                    f"标准 {gt} 未进 top-{len(self.last_reply_list)}"
                )
            print("; ".join(parts))
            return
        tail = self.last_reply.rsplit("</think>", 1)[-1]
        ans = extract_sid_tokens(tail)[:3]
        if len(ans) < 3 or len(gt) < 3:
            print(f"无法解析: 回答中提取到 {ans}, 标准答案 {gt}")
            return
        step = calculate_reward(ans, gt)
        parts = [f"回答 {ans} | 标准 {gt}", f"步进分 {step:.2f}"]
        if prefix_map is not None:
            fmt = 0.1 * float(calculate_format_reward(ans, prefix_map))
            parts.append(f"格式分 {fmt:.2f}, 总分 {step + fmt:.2f}")
        else:
            parts.append(f"总分 {step:.2f} (未设 DATASET, 跳过格式检查)")
        print("; ".join(parts))


def task_sid2title(session, items):
    sid, title = random.choice(items)
    user = f'What is the title of item "{sid}"?'
    print(f"\n[题目] {user}\n[期望] {title}")
    session.last_gt = None
    session.send(user, SID_SYSTEM)


def task_title2sid(session, items):
    sid, title = random.choice(items)
    user = f"Which item has the title: {title}?"
    print(f"\n[题目] {user}\n[期望] {sid} (已记录, 可直接 /score)")
    session.last_gt = sid
    session.send(user, SID_SYSTEM)


def task_recommend(session, items, history_arg):
    if history_arg:
        sids = [
            s.strip()
            for s in history_arg.replace(" ", ",").split(",")
            if s.strip()
        ]
    else:
        sids = [sid for sid, _ in random.sample(items, random.randint(1, 3))]
    user = (
        f"The user has sequentially interacted with items {', '.join(sids)}. "
        "Can you recommend the next item for him? Let's think step by step "
        "before making recommendation. Directly output the item SID after "
        "thinking."
    )
    print(f"\n[题目] {user}\n(有标准答案的话, /score <a_x><b_y><c_z> 打分)")
    session.last_gt = None
    session.send(user, REC_SYSTEM)


def main():
    ap = argparse.ArgumentParser(
        description="SIDReasoner 三阶段 checkpoint 终端交互"
    )
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--tokenizer", default=None, help="缺省与 --model 同目录")
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="生成上限; 0 = 不设上限(仅受模型上下文长度约束, 约 32k)",
    )
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--think", default="on", choices=["on", "off"])
    ap.add_argument("--prompt", default=None, help="单轮生成后退出(测试用)")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit(
            "未指定模型: --model <checkpoint路径> 或 MODEL=... 环境变量"
        )
    if args.tokenizer is None:
        args.tokenizer = args.model
    if not os.path.isdir(args.model):
        raise SystemExit(f"模型目录不存在: {args.model}")

    session = Session(args)
    info_path = resolve_info_path()
    items = load_info_items(info_path) if info_path else None
    prefix_map = None
    if info_path:
        from src.reward.direct_recommendation_steprule import (
            construct_prefix_allowed_hashmap,
        )

        prefix_map = construct_prefix_allowed_hashmap(info_path)
        try:
            session.setup_eval(info_path, items)
        except Exception as e:
            print(f"eval 复刻初始化失败, 退回自由生成: {e}")

    print(
        f"模型 {args.model}\n设备 {session.device} | dtype {session.dtype} | "
        f"思考 {'on' if session.think else 'off'} | "
        f"{'多轮' if session.multi else '单轮(每条独立)'} | "
        f"temp {session.temperature} | 评测复刻"
        f"{'开' if (session.constrain and session.trie) else '关'}"
        f"(beams={session.eval_cfg['num_beams']})\n数据集 "
        f"{os.environ.get('DATASET', '未设(示例题目与格式检查不可用)')}"
    )
    print(HELP)

    if args.prompt is not None:
        session.send(args.prompt)
        return

    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd == "/exit":
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/reset":
                session.history = []
                print("已清空多轮历史")
            elif cmd == "/think":
                if rest in ("on", "off"):
                    session.think = rest == "on"
                    print(f"思考模式 {rest}")
                else:
                    print("用法: /think on|off")
            elif cmd == "/constrain":
                if rest in ("on", "off"):
                    session.constrain = rest == "on"
                    note = "" if session.trie else "(trie 未就绪, 实际不生效)"
                    print(f"评测复刻解码 {rest}{note}")
                else:
                    print("用法: /constrain on|off")
            elif cmd == "/beams":
                if rest.isdigit() and int(rest) > 0:
                    session.eval_cfg["num_beams"] = int(rest)
                    print(f"束宽 {rest}")
                else:
                    print("用法: /beams N (正整数)")
            elif cmd == "/sys":
                if rest in ("rec", "id", "pred"):
                    session.default_system = {
                        "rec": REC_SYSTEM,
                        "id": SID_SYSTEM,
                        "pred": PRED_SYSTEM,
                    }[rest]
                    print(f"自由输入 system 提示: {rest}")
                else:
                    print("用法: /sys rec|id|pred")
            elif cmd == "/multi":
                if rest in ("on", "off"):
                    session.multi = rest == "on"
                    print(f"多轮对话 {rest}")
                else:
                    print("用法: /multi on|off")
            elif cmd == "/temp":
                session.temperature = float(rest)
                print(
                    f"温度 {session.temperature}"
                    + (" (贪心)" if session.temperature <= 0 else "")
                )
            elif cmd == "/sid":
                if items is None:
                    print(NO_DATA_HINT)
                    continue
                for sid, title in random.sample(
                    items, min(int(rest or 5), len(items))
                ):
                    print(f"  {sid}  {title}")
            elif cmd == "/t1":
                if items is None:
                    print(NO_DATA_HINT)
                    continue
                task_sid2title(session, items)
            elif cmd == "/t2":
                if items is None:
                    print(NO_DATA_HINT)
                    continue
                task_title2sid(session, items)
            elif cmd == "/t3":
                if items is None:
                    print(NO_DATA_HINT)
                    continue
                task_recommend(session, items, rest)
            elif cmd == "/score":
                gt = rest or session.last_gt
                if not gt:
                    print("用法: /score <a_x><b_y><c_z> (或先 /t2 自动记录)")
                    continue
                session.score(gt, prefix_map)
            else:
                session.send(line)
        except KeyboardInterrupt:
            print("\n(本轮已中断; /exit 退出)")


if __name__ == "__main__":
    main()
