"""P1 可行性验证：真实 LLM 编译器（Qwen3.7-Max）+ 已验证的 State VM，端到端。

取真实 LongMemEval knowledge-update 案例（5K 成绩 27:12 → 25:50），让 Qwen 把两条
证据 turn 编译成状态转移，跑 State VM（时间折叠），验证它重建出的当前值是 25:50（新值）
而非 27:12（旧值）。这检验的正是 SRG 唯一需要 LLM 的一步：编译准确性。

Run:  PYTHONPATH=. python3 trace/feasibility_p1.py
"""

from __future__ import annotations

import os

import json

from trace.compile import compile_stream
from trace.runtime.llm_client import DEFAULT_MODEL
from trace.core import execute, Principal

LME = os.getenv("TRACE_LONGMEMEVAL_DATA", "data/LongMemEval/longmemeval_oracle.json")


def evidence_stream(item):
    """取 has_answer 证据 turn，按日期升序 -> [(date, text)]。"""
    out = []
    dates = item.get("haystack_dates", [])
    for si, sess in enumerate(item["haystack_sessions"]):
        date = dates[si] if si < len(dates) else ""
        for t in sess:
            if t.get("has_answer") and t.get("role") == "user":
                out.append((date, t.get("content", "")))
    out.sort(key=lambda x: x[0])
    return out


def show_state(state):
    for pp in state:
        print(f"    [{pp.subject} . {pp.attribute}] = {pp.value!r}  (t_event={pp.t_event})")


def run_case(item, model=DEFAULT_MODEL):
    print("=" * 74)
    print("Q:", item["question"])
    print("A(gold):", item["answer"])
    stream = evidence_stream(item)
    print(f"\n证据 turns（{len(stream)} 条，按时间升序）：")
    for d, t in stream:
        print(f"  [{d}] {t[:110]}")

    print(f"\n>>> 用 {model} 编译成状态转移 ...")
    ledger = compile_stream(stream, model=model)
    print(f"编译出 {len(ledger)} 条转移：")
    for tr in ledger:
        print(f"  {tr.op:9} [{tr.subject} . {tr.attribute}] = {tr.content!r}  (t_event={tr.t_event})")

    print("\n>>> 执行引擎 execute -> 投影出【当前状态】S(t)：")
    state = execute(ledger, t_query=None, principal=Principal(allow_sensitive=True))
    show_state(state)

    # 判定：当前状态里，5K 相关槽位的值应含新值 25:50，且不应是旧值 27:12
    vals = " ".join(pp.value for pp in state).lower()
    ans = str(item["answer"]).lower()
    import re
    ans_nums = re.findall(r"\d{1,2}:\d{2}|\d[\d,\.]+", ans)
    hit_new = any(n in vals for n in ans_nums) if ans_nums else (ans in vals)
    stale = "27:12" in vals
    print("\n>>> 结果：")
    print(f"    当前状态含正确新值({ans_nums or ans})? {'✅' if hit_new else '❌'}   仍残留旧值(27:12)? {'❌残留' if stale else '✅无'}")
    return hit_new and not stale


def main():
    data = json.load(open(LME))
    ku = [x for x in data if x["question_type"] == "knowledge-update"]
    # 用第一个含数值更新的案例（27:12 -> 25:50）
    target = None
    for x in ku:
        stream = evidence_stream(x)
        joined = " ".join(t for _, t in stream)
        if "27:12" in joined and "25:50" in joined:
            target = x
            break
    target = target or ku[0]
    ok = run_case(target)
    print("\n" + "=" * 74)
    print(f"P1 端到端可行性（真实 LLM 编译器 + State VM）：{'✅ 通过' if ok else '❌ 未通过'}")


if __name__ == "__main__":
    main()
