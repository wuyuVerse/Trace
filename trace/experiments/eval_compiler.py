"""Task 2 / E4 —— LLM 编译器批量准确率（M1 门禁）。

回应 A0.5「编译器误差是主风险面」：P0 用确定性编译隔离变量得高分；这里量化**真实 LLM
编译器**接进全链路后的端到端准确率，及相对「确定性上界」的保留率。

流程（对每题）：
  证据 turns → [LLM 编译器] → 转移 → [State VM 重放/投影] → 当前状态 → 判定是否含 gold 答案
对照：确定性上界 = 直接取「时间最晚的含值证据」（等价于完美编译+折叠）。

指标：
  end2end_acc  = LLM 编译链路答对率
  upper_acc    = 确定性上界答对率
  retention    = end2end_acc / upper_acc   ← 门禁：强模型 ≥ 0.90

LLM 调用带磁盘缓存（trace/RESULTS/.compile_cache/），re-run 免费、可复现。

集群提交（勿本地跑批量）：
  python3 trace/cluster/submit.py \
    --cmd "PYTHONPATH=. python3 trace/experiments/eval_compiler.py --model qwen3.7-max-ali \
           --out trace/RESULTS/compile_qwen.json" --name srg-compile-qwen --pool spot

Run(小样本本地冒烟):  PYTHONPATH=. python3 trace/experiments/eval_compiler.py --model qwen3.7-max-ali --limit 5
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import re
from pathlib import Path

from trace.compile.llm_compiler import compile_episode
from trace.core import execute, Principal

LME = os.getenv("TRACE_LONGMEMEVAL_DATA", "data/LongMemEval/longmemeval_oracle.json")
CACHE_DIR = Path(os.getenv("TRACE_COMPILE_CACHE", ".trace_cache/compile"))


def evidence_stream(item):
    out = []
    dates = item.get("haystack_dates", [])
    for si, sess in enumerate(item["haystack_sessions"]):
        date = dates[si] if si < len(dates) else ""
        for t in sess:
            if t.get("has_answer") and t.get("role") == "user":
                out.append((date, t.get("content", "")))
    out.sort(key=lambda x: x[0])
    return out


# 英文数词 ↔ 阿拉伯数字（0-20 覆盖 LongMemEval 计数类答案）
_NUMWORD = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6",
            "seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12",
            "thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17",
            "eighteen":"18","nineteen":"19","twenty":"20"}
_WORDNUM = {v: k for k, v in _NUMWORD.items()}


def _norm_num_variants(token: str) -> set[str]:
    """一个数字/数词 token 的等价变体集合（'4'↔'four'）。"""
    t = token.lower()
    out = {t}
    if t in _NUMWORD:
        out.add(_NUMWORD[t])
    if t in _WORDNUM:
        out.add(_WORDNUM[t])
    return out


def _ans_hit(answer, text) -> bool:
    ans = str(answer).lower()
    low = text.lower()
    # 数值 span（时间/多位数）：直接子串匹配
    spans = re.findall(r"\d{1,2}:\d{2}|\d[\d,\.]+", ans)
    if spans:
        return any(s in low for s in spans)
    # 计数类：答案里的单个数字/英文数词，命中任一等价变体即可
    small = re.findall(r"\b(?:\d{1,2}|" + "|".join(_NUMWORD) + r")\b", ans)
    if small:
        low_tokens = set(re.findall(r"[a-z0-9]+", low))
        for tok in small:
            if _norm_num_variants(tok) & low_tokens:
                return True
        return False
    # 纯文字答案：命中过半关键词（≥5 字符实词）
    words = [w for w in re.findall(r"[a-z0-9]+", ans) if len(w) >= 5]
    return (sum(1 for w in words if w in low) >= max(1, len(words) // 2)) if words else (ans in low)


def _candidate_values(content):
    return re.findall(r"\d{1,2}:\d{2}|\d[\d,\.]+", content)


def upper_bound_answer(stream) -> str:
    """确定性上界：取时间最晚的含值证据（完美编译+折叠的等价结果）。"""
    if not stream:
        return ""
    for date, content in reversed(stream):
        if _candidate_values(content):
            return content
    return stream[-1][1]


def _cached_compile(text, t_event, t_ingest, model):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{model}|{t_ingest}|{text}".encode()).hexdigest()[:16]
    cf = CACHE_DIR / f"{key}.json"
    if cf.exists():
        data = json.loads(cf.read_text())
        from trace.core import Transition
        return [Transition(**d) for d in data]
    trs = compile_episode(text, t_event=t_event, t_ingest=t_ingest, model=model)
    cf.write_text(json.dumps([t.__dict__ for t in trs]))
    return trs


def eval_model(data, model, limit=0):
    p = Principal(allow_sensitive=True)
    n = e2e = upper = 0
    for item in data:
        if item.get("question_type") != "knowledge-update":
            continue
        if limit and n >= limit:
            break
        n += 1
        stream = evidence_stream(item)
        # 确定性上界
        if _ans_hit(item["answer"], upper_bound_answer(stream)):
            upper += 1
        # LLM 编译链路
        ledger = []
        for clock, (te, text) in enumerate(stream, start=1):
            ledger.extend(_cached_compile(text, te, clock, model))
        state = execute(ledger, None, principal=p)
        state_text = " ".join(pp.value for pp in state)
        if _ans_hit(item["answer"], state_text):
            e2e += 1
    e2e_acc = e2e / n if n else 0.0
    upper_acc = upper / n if n else 0.0
    retention = e2e_acc / upper_acc if upper_acc else 0.0
    return {"model": model, "n": n, "end2end_acc": round(e2e_acc, 4),
            "upper_acc": round(upper_acc, 4), "retention": round(retention, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.7-max-ali")
    ap.add_argument("--limit", type=int, default=0, help="限题数(本地冒烟用);0=全部78题")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = json.load(open(LME))
    res = eval_model(data, args.model, args.limit)
    print("=" * 60)
    print(f"编译器准确率 — {args.model}")
    print("=" * 60)
    print(f"  题数(knowledge-update)      : {res['n']}")
    print(f"  确定性上界准确率 upper_acc  : {res['upper_acc']}")
    print(f"  LLM编译端到端 end2end_acc   : {res['end2end_acc']}")
    print(f"  保留率 retention            : {res['retention']}  (门禁 ≥0.90)")
    print(f"  {'✅ 过门禁' if res['retention'] >= 0.90 else '⚠️ 未过门禁'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"  写入 {args.out}")


if __name__ == "__main__":
    main()
