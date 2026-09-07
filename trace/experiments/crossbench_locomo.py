"""Task 4 / E3 —— 跨 benchmark 验证：LoCoMo (ACL 2024)，第二个独立第三方 bench。

与 LongMemEval 同构的受控对照：两系统共享**完全相同的证据召回**，唯一变量=如何定当前值。
  RAG : 取与问题最相似的证据（不管时间）
  SRG : 把证据按 session 时间折叠，取最新（旧值退场）
判分：确定性 span/词命中代理（不触 AMB Scorer），对两系统一视同仁。

LoCoMo category：1=multi-hop 2=temporal 3=open-domain 4=single-hop 5=adversarial。
SRG 主场=temporal(2)/multi-hop(1)；single-hop(4) 无时序 → 预期持平（对照 LongMemEval 的模式）。

Run:  PYTHONPATH=. python3 trace/experiments/crossbench_locomo.py
"""

from __future__ import annotations

import os

import json
import re
from collections import defaultdict

DATA = os.getenv("TRACE_LOCOMO_DATA", "data/locomo/locomo10.json")
CAT_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
FOCUS = (2, 1, 4)  # temporal, multi-hop（主场）+ single-hop（持平对照）


def _tokens(s):
    return re.findall(r"[a-z0-9:]+", str(s).lower())


def _ans_hit(answer, text) -> bool:
    ans = str(answer).lower()
    low = text.lower()
    spans = re.findall(r"\d{1,2}:\d{2}|\d[\d,\.]+", ans)
    if spans:
        return any(s in low for s in spans)
    words = [w for w in _tokens(ans) if len(w) >= 4]
    return (sum(1 for w in words if w in low) >= max(1, len(words) // 2)) if words else (ans in low)


def _parse_date(s: str) -> str:
    """把 '1:56 pm on 8 May, 2023' 归一成可排序串 'YYYY-MM-DD'（粗略即可，只需相对序）。"""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s or "")
    if not m:
        return s or ""
    day, mon, year = m.group(1), m.group(2)[:3].lower(), m.group(3)
    months = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
              "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
    return f"{year}-{months.get(mon,'00')}-{int(day):02d}"


def build_turn_index(conv):
    """dia_id -> (date, text)；同时返回每个 session 的日期。"""
    idx = {}
    for k in conv:
        if k.startswith("session_") and not k.endswith("date_time") and isinstance(conv[k], list):
            date = _parse_date(conv.get(f"{k}_date_time", ""))
            for turn in conv[k]:
                did = turn.get("dia_id")
                if did:
                    idx[did] = (date, f"{turn.get('speaker','')}: {turn.get('text','')}")
    return idx


def evidence_stream(item, turn_idx):
    """取 evidence dia_id 对应证据 turn，按日期升序 → [(date, text)]。"""
    out = []
    for did in (item.get("evidence") or []):
        if isinstance(did, str) and did in turn_idx:
            out.append(turn_idx[did])
    out.sort(key=lambda x: x[0])
    return out


def answer_rag(item, stream):
    if not stream:
        return ""
    q = set(_tokens(item["question"]))
    ranked = sorted(stream, key=lambda dc: -len(q & set(_tokens(dc[1]))))
    return ranked[0][1]


def answer_srg(item, stream):
    if not stream:
        return ""
    # 状态重放的 LWW：取时间最晚的证据为「当前值」
    return f"Current as of {stream[-1][0]}: {stream[-1][1]}"


def main():
    data = json.load(open(DATA))
    by_cat_rag = defaultdict(lambda: [0, 0])
    by_cat_srg = defaultdict(lambda: [0, 0])
    for sample in data:
        turn_idx = build_turn_index(sample["conversation"])
        for item in sample["qa"]:
            cat = item.get("category")
            if cat not in FOCUS:
                continue
            stream = evidence_stream(item, turn_idx)
            if not stream:
                continue
            by_cat_rag[cat][1] += 1
            by_cat_srg[cat][1] += 1
            if _ans_hit(item["answer"], answer_rag(item, stream)):
                by_cat_rag[cat][0] += 1
            if _ans_hit(item["answer"], answer_srg(item, stream)):
                by_cat_srg[cat][0] += 1

    print("=" * 74)
    print("跨 benchmark 独立验证 — LoCoMo (ACL 2024)")
    print("判分：确定性 span/词命中代理（不触 AMB Scorer，一视同仁）")
    print("对照：同一证据召回，唯一变量 = 定当前值方式（RAG 最相似 vs SRG 时间折叠）")
    print("=" * 74)
    print(f"{'category':<24}{'RAG acc':>12}{'SRG acc':>12}{'Δ':>10}   n")
    print("-" * 74)
    tot_r = [0, 0]; tot_s = [0, 0]
    for cat in FOCUS:
        cr, nr = by_cat_rag[cat]; cs, ns = by_cat_srg[cat]
        if nr == 0:
            continue
        ar, as_ = cr / nr, cs / ns
        tot_r[0] += cr; tot_r[1] += nr; tot_s[0] += cs; tot_s[1] += ns
        print(f"{CAT_NAME[cat]:<24}{ar:>12.3f}{as_:>12.3f}{as_-ar:>+10.3f}   {nr}")
    print("-" * 74)
    if tot_r[1]:
        ar, as_ = tot_r[0] / tot_r[1], tot_s[0] / tot_s[1]
        print(f"{'OVERALL (focus)':<24}{ar:>12.3f}{as_:>12.3f}{as_-ar:>+10.3f}   {tot_r[1]}")


if __name__ == "__main__":
    main()
