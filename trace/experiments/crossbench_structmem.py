"""跨 benchmark 扩展 —— StructMemEval state_machine_location（状态机覆盖任务）。

SRG 绝对主场：11 个 session 描述居住地依次变迁（Trastevere→…→Kungsholmen），
query 问「基于我现在住哪的早餐」，reference_answer 明确要求**排除所有旧地早餐**。
这正是 SUPERSEDE（新地覆盖旧地）+ GOVERN（不激活 superseded 旧值）。

无显式时间戳，但 **session 顺序即时序**（session_01→11）。

受控对照：
  RAG : 检索与问题最相似的 session 内容 → 易混入语义相似的旧地（旧地也在聊早餐）
  SRG : 按 session 顺序折叠「居住地」状态，取最新地 → 只答最新地的早餐
判分（确定性）：命中最新地/最新早餐关键词 且 不含任何旧地早餐关键词 = 对。

Run:  PYTHONPATH=. python3 trace/experiments/crossbench_structmem.py
"""

from __future__ import annotations

import os

import glob
import json
import re

DATA_GLOB = os.getenv("TRACE_STRUCTMEM_GLOB", "data/StructMemEval/state_machine_location/*/*.json")


def _tok(s):
    return set(re.findall(r"[a-z]+", str(s).lower()))


def _current_and_stale(ref_text):
    """从 reference_answer 解析：正确关键词（currently live 后）+ 应排除的旧地早餐词。"""
    low = ref_text.lower()
    # 正确部分：'NOT mention' 之前
    split = low.split("not mention")
    correct = split[0]
    stale = split[1] if len(split) > 1 else ""
    return _tok(correct), _tok(stale)


def build_sessions(case):
    """[(session_idx, topic, text)]，text=该 session 全部 user 内容。"""
    out = []
    for i, s in enumerate(case["sessions"]):
        text = " ".join(m["content"] for m in s.get("messages", []) if m.get("role") == "user")
        out.append((i, s.get("topic", ""), text))
    return out


def answer_rag(question, sessions):
    """RAG：取与问题最相似的 session 文本（易命中旧地——旧地也在聊早餐）。"""
    q = _tok(question)
    ranked = sorted(sessions, key=lambda s: -len(q & _tok(s[1] + " " + s[2])))
    return ranked[0][2] if ranked else ""


def answer_srg(question, sessions):
    """SRG：按 session 顺序折叠「居住地」状态，只取**最新** session（当前住地）的内容。

    session 顺序 = 时序；最新的 'Life in X' / 'Moving to X' 即当前状态（SUPERSEDE 旧地）。
    """
    if not sessions:
        return ""
    # 取最后一个描述当前生活的 session（最新状态）
    return sessions[-1][2]


def judge(correct_kw, stale_kw, answer):
    """对 = 命中至少一个正确关键词 且 不命中任何"独有的"旧地关键词。"""
    a = _tok(answer)
    # 正确关键词里排除通用词
    stop = {"the", "your", "you", "and", "with", "from", "since", "currently", "live",
            "breakfast", "answer", "should", "not", "mention", "previous", "cities", "city",
            "café", "cafe", "coffee", "at", "a", "is", "in", "of", "based", "where", "now"}
    corr = {w for w in correct_kw if len(w) >= 4 and w not in stop}
    stale = {w for w in stale_kw if len(w) >= 4 and w not in stop} - corr
    hit_correct = len(corr & a) > 0
    hit_stale = len((stale & a)) > 0
    return hit_correct and not hit_stale


def main():
    files = sorted(glob.glob(DATA_GLOB))
    rc = sc = n = 0
    for f in files:
        case = json.load(open(f))
        sessions = build_sessions(case)
        for q in case.get("queries", []):
            ref = q.get("reference_answer", {})
            ref_text = ref.get("text", "") if isinstance(ref, dict) else str(ref)
            if "not mention" not in ref_text.lower():
                continue  # 只测有明确排除项的状态覆盖题
            correct_kw, stale_kw = _current_and_stale(ref_text)
            n += 1
            if judge(correct_kw, stale_kw, answer_rag(q["question"], sessions)):
                rc += 1
            if judge(correct_kw, stale_kw, answer_srg(q["question"], sessions)):
                sc += 1
    print("=" * 70)
    print("跨 benchmark 扩展 — StructMemEval state_machine_location（状态覆盖）")
    print("判分：含最新地早餐 且 不含旧地早餐（确定性）")
    print("对照：RAG 最相似 vs SRG 取最新 session 状态")
    print("=" * 70)
    print(f"{'system':<24}{'accuracy':>12}{'n':>8}")
    print("-" * 70)
    print(f"{'RAG(最相似)':<24}{rc/n if n else 0:>12.3f}{n:>8}")
    print(f"{'SRG(最新状态)':<24}{sc/n if n else 0:>12.3f}{n:>8}")
    print("-" * 70)
    print(f"Δ = {(sc/n - rc/n) if n else 0:+.3f}   （SUPERSEDE 覆盖旧地 + 排除 superseded 旧值）")


if __name__ == "__main__":
    main()
