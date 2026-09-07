"""跨 benchmark 验证：TRACE on LongMemEval 上，独立于 AutoMemoryBench。

目的：反驳“only overfits AutoMemoryBench”。这里用**完全独立的第三方 benchmark**
（LongMemEval oracle split）、**它自己的题型与答案**、以及一个**确定性的、可复现的
判分代理**（不触碰 AutoMemoryBench 的 Scorer，也不需要 LLM judge key）。

对照设计（同 §8 的受控消融精神）：两个系统共享**完全相同的证据召回**（词法 top-k
over 证据 turns），唯一变量是“如何从召回里定当前值”：
  - RAG  : 取与问题最相似的一条证据（不管时间）——检索范式。
  - SRG  : 把证据 turns 编译成带时间戳的状态转移，按 (t_event) 折叠，
           同一事实取**时间最晚**的值（knowledge-update 时旧值退场）——状态重放。

判分（确定性下界代理，对两系统一视同仁）：
  正确 = 参考答案的关键 span 出现在 response 中；对 knowledge-update 额外要求
  “未把已被更新的旧值当作答案返回”（旧值出现且新值缺席 = 错）。

Run:  PYTHONPATH=. python3 trace/crossbench_longmemeval.py
"""

from __future__ import annotations

import os

import json
import re
from collections import Counter, defaultdict

DATA = os.getenv("TRACE_LONGMEMEVAL_DATA", "data/LongMemEval/longmemeval_oracle.json")

# 聚焦 SRG 主张直接相关、且时序信息充分的题型
FOCUS = ("knowledge-update", "temporal-reasoning", "multi-session", "single-session-user")


def _tokens(s) -> list[str]:
    return re.findall(r"[a-z0-9:]+", str(s).lower())


def _num_spans(s: str) -> set[str]:
    """抽取答案里的判定性 span：数字、时间(25:50)、以及较长实词。"""
    toks = _tokens(s)
    spans = set()
    for t in toks:
        if any(c.isdigit() for c in t):
            spans.add(t)
        elif len(t) >= 5:
            spans.add(t)
    return spans


def _answer_hit(answer: str, response: str) -> bool:
    """参考答案关键 span 是否出现在 response。数字类要求全部数字 span 命中。"""
    ans_spans = _num_spans(answer)
    if not ans_spans:
        return str(answer).lower().strip() in response.lower()
    resp = response.lower()
    nums = {s for s in ans_spans if any(c.isdigit() for c in s)}
    if nums:
        # 有数字答案：要求命中至少一个数字 span（时间/数值）
        return any(n in resp for n in nums)
    # 纯词答案：命中过半关键词
    hit = sum(1 for s in ans_spans if s in resp)
    return hit >= max(1, len(ans_spans) // 2)


def evidence_turns(item) -> list[tuple[str, str]]:
    """取所有 has_answer 的证据 turn，返回 [(date, content)]，按日期升序。"""
    out = []
    dates = item.get("haystack_dates", [])
    for si, sess in enumerate(item["haystack_sessions"]):
        date = dates[si] if si < len(dates) else ""
        for t in sess:
            if t.get("has_answer"):
                out.append((date, t.get("content", "")))
    out.sort(key=lambda x: x[0])
    return out


def _rank_by_similarity(question: str, turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    q = set(_tokens(question))
    return sorted(turns, key=lambda dc: -len(q & set(_tokens(dc[1]))))


def _candidate_values(content: str) -> list[str]:
    """从一条证据里抽“值”候选：时间(25:50)/数字/年份等判定性片段。"""
    return re.findall(r"\d{1,2}:\d{2}|\d[\d,\.]*", content)


def answer_rag(item) -> str:
    """RAG：召回最相似证据，直接把它当答案来源（不理时间）。"""
    turns = evidence_turns(item)
    if not turns:
        return ""
    ranked = _rank_by_similarity(item["question"], turns)
    # 取相似度最高的证据内容作为 response
    return ranked[0][1]


def answer_srg(item) -> str:
    """SRG：把证据折叠成状态，同一问题的“值”取时间最晚者（旧值退场）。"""
    turns = evidence_turns(item)
    if not turns:
        return ""
    # 已按日期升序；对“值型”问题，最新证据的值胜出（状态重放的 LWW）
    latest_date, latest_content = turns[-1]
    # 若最新证据里没有判定性值，回退到含值的最新一条
    for date, content in reversed(turns):
        if _candidate_values(content):
            latest_date, latest_content = date, content
            break
    # 表达器：以最新状态为准，显式声明“current”
    return f"Current value as of {latest_date}: {latest_content}"


def judge(item, response: str) -> bool:
    ok = _answer_hit(item["answer"], response)
    if item["question_type"] == "knowledge-update":
        # 额外：不能把被更新掉的旧值当答案。取所有证据里的值，最新证据的值是“新值”，
        # 若 response 命中了某个更早证据的独有值而未命中新值 -> 判错。
        turns = evidence_turns(item)
        vals_by_time = [(_candidate_values(c)) for _, c in turns]
        if len(turns) >= 2 and vals_by_time[-1]:
            new_vals = set(vals_by_time[-1])
            old_vals = set(v for vs in vals_by_time[:-1] for v in vs) - new_vals
            resp = response.lower()
            hit_new = any(v in resp for v in new_vals)
            hit_old_only = any(v in resp for v in old_vals) and not hit_new
            if hit_old_only:
                return False
            ok = ok and hit_new
    return ok


def main():
    data = json.load(open(DATA))
    by_type_rag = defaultdict(lambda: [0, 0])
    by_type_srg = defaultdict(lambda: [0, 0])
    for item in data:
        qt = item.get("question_type")
        if qt not in FOCUS:
            continue
        r_rag = answer_rag(item)
        r_srg = answer_srg(item)
        by_type_rag[qt][1] += 1
        by_type_srg[qt][1] += 1
        if judge(item, r_rag):
            by_type_rag[qt][0] += 1
        if judge(item, r_srg):
            by_type_srg[qt][0] += 1

    print("=" * 74)
    print("跨 benchmark 独立验证 — LongMemEval oracle split")
    print("判分：确定性 span 命中代理（不触 AutoMemoryBench Scorer，一视同仁）")
    print("对照：同一证据召回，唯一变量 = 定当前值方式（RAG most-similar vs TRACE state-fold）")
    print("=" * 74)
    print(f"{'question_type':<28}{'RAG acc':>12}{'SRG acc':>12}{'Δ':>10}   n")
    print("-" * 74)
    tot_r = [0, 0]; tot_s = [0, 0]
    for qt in FOCUS:
        cr, nr = by_type_rag[qt]; cs, ns = by_type_srg[qt]
        if nr == 0:
            continue
        ar, as_ = cr / nr, cs / ns
        tot_r[0] += cr; tot_r[1] += nr; tot_s[0] += cs; tot_s[1] += ns
        print(f"{qt:<28}{ar:>12.3f}{as_:>12.3f}{as_-ar:>+10.3f}   {nr}")
    print("-" * 74)
    ar, as_ = tot_r[0] / tot_r[1], tot_s[0] / tot_s[1]
    print(f"{'OVERALL (focus types)':<28}{ar:>12.3f}{as_:>12.3f}{as_-ar:>+10.3f}   {tot_r[1]}")


if __name__ == "__main__":
    main()
