"""
SIDReasoner 终端把玩入口(Mac MPS / CPU / 服务器 GPU 均可)

像网页 AI 一样与三个 Stage 的 checkpoint 对话, 直观感受各阶段能力:
- Stage 1: /think off 后直答 SID(/t1 /t2 验证 SID<->标题翻译)
- Stage 2: /think on 先 <think> 推理再答 SID(教师腔, 内容较套路)
- Stage 3: 同格式, 推理更自主、SID 更准(/score 按论文规则逐级打分)

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

HELP = """\
命令(直接输入文本则作为 user 消息发给模型):
  /t1             随机真实商品: 问 SID -> 标题(验证 Stage 1 翻译能力)
  /t2             随机真实商品: 问标题 -> SID(标准答案自动记录, 之后可 /score)
  /t3 [s1,s2,...] 推荐任务: 给定(缺省随机)SID 历史 -> 思考后输出 SID
  /score [SID]    按论文规则给上一条回答打分(0.25/0.5/1.0 步进 + 0.1 格式)
  /sid [n]        随机展示 n 条真实 SID 与标题(自己出题用)
  /think on|off   思考模式开关(Stage 1 建议 off, Stage 2/3 保持 on)
  /sys rec|id     自由输入的 system 提示: 推荐任务(rec, 默认)/物品识别(id)
  /multi on|off   多轮对话开关(默认 off: 每条输入独立单轮, 贴近训练分布)
  /temp T         采样温度(0 = 贪心; 默认 0.6)
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
        reply = self.generate(messages)
        if self.multi:
            self.history = messages + [{"role": "assistant", "content": reply}]
        return reply

    def score(self, gt_sid, prefix_map):
        if not self.last_reply:
            print("还没有生成过回答, 先发一条消息或 /t1 /t2 /t3")
            return
        gt = extract_sid_tokens(gt_sid)[:3]
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

    print(
        f"模型 {args.model}\n设备 {session.device} | dtype {session.dtype} | "
        f"思考 {'on' if session.think else 'off'} | "
        f"{'多轮' if session.multi else '单轮(每条独立)'} | "
        f"temp {session.temperature}\n数据集 "
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
            elif cmd == "/sys":
                if rest in ("rec", "id"):
                    session.default_system = (
                        REC_SYSTEM if rest == "rec" else SID_SYSTEM
                    )
                    print(f"自由输入 system 提示: {rest}")
                else:
                    print("用法: /sys rec|id (rec=推荐任务, id=物品识别)")
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
