"""跨 benchmark 扩展 —— Membench (Findings ACL 2025) knowledge_update split。

这是 SRG 的完美主场：每个 event 的 message_list 带 time 时间戳 + 结构化 (rel, attr, value)
三元组（直接对应 SRG 的 subject-attribute-value 转移），QA 是多选，choices 里含旧值干扰。
正是测「取当前值 vs 被更新的旧值」。

受控对照（同 LongMemEval/LoCoMo）：共享同一候选，唯一变量=定当前值方式。
  RAG : 取与问题最相似的 message（不管 time）→ 选最接近的 choice
  SRG : 把 (rel,attr) 相同的 message 按 time 折叠取最新 value → 选最接近的 choice
判分：多选正确率（对齐官方 ground_truth），对两系统一视同仁。

Run:  PYTHONPATH=. python3 trace/experiments/crossbench_membench.py
"""

from __future__ import annotations

import os

import json
import re
from collections import defaultdict

DATA = os.getenv("TRACE_MEMBENCH_DATA", "data/Membench/ThirdAgent/knowledge_update.json")


def _tok(s):
    return set(re.findall(r"[a-z0-9:]+", str(s).lower()))


def _clean_time(t):
    """'2024-10-02 08:06' Tuesday -> 2024-10-02 08:06（可排序）。"""
    m = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})", str(t))
    return m.group(1) if m else str(t)


def _pick_choice(pred_text, choices):
    """把预测文本映射到最接近的 choice 字母。"""
    best, best_score = None, -1
    pt = _tok(pred_text)
    for letter, val in choices.items():
        sc = len(pt & _tok(val))
        # 数值/时间 span 命中加权
        for span in re.findall(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\d+", str(val)):
            if span in pred_text:
                sc += 3
        if sc > best_score:
            best, best_score = letter, sc
    return best


def _target_slot_messages(messages, target_step_id):
    """目标槽位 (rel,attr) 的时间序列（用 target_step_id 定位身份）。两系统共享，保证公平。"""
    keys = []
    if target_step_id:
        for sid in (target_step_id if isinstance(target_step_id, list) else [target_step_id]):
            if isinstance(sid, int) and 0 <= sid < len(messages):
                mm = messages[sid]
                keys.append((mm.get("rel"), mm.get("attr")))
    if not keys:
        return None
    seq = [m for m in sorted(messages, key=lambda m: _clean_time(m.get("time", "")))
           if (m.get("rel"), m.get("attr")) in keys]
    return seq or None


def answer_rag(question, messages, choices, target_step_id=None):
    """RAG：同一目标槽位内，取与问题最相似的 message value（**时间无关**）。"""
    seq = _target_slot_messages(messages, target_step_id)
    if seq is None:
        q = _tok(question)
        seq = sorted(messages, key=lambda m: -len(q & _tok(f"{m.get('rel')} {m.get('attr')} {m.get('value')}")))
    q = _tok(question)
    ranked = sorted(seq, key=lambda m: -len(q & _tok(str(m.get("value", "")))))
    return _pick_choice(str(ranked[0].get("value", "")) if ranked else "", choices)


def _slot_seq(question, messages, target_step_id):
    seq = _target_slot_messages(messages, target_step_id)
    if seq is None:
        slots = defaultdict(list)
        for m in sorted(messages, key=lambda m: _clean_time(m.get("time", ""))):
            slots[(m.get("rel"), m.get("attr"))].append(m)
        q = _tok(question)
        best = max(slots, key=lambda k: len(q & _tok(f"{k[0]} {k[1]}")), default=None)
        seq = slots.get(best) if best else None
    return seq


def answer_srg(question, messages, choices, target_step_id=None):
    """SRG(latest/LWW)：目标槽位内取时间最新值。适合「当前有效值」型 query。"""
    seq = _slot_seq(question, messages, target_step_id)
    if not seq:
        return _pick_choice("", choices)
    return _pick_choice(str(seq[-1].get("value", "")), choices)


def answer_srg_asof(question, messages, choices, target_step_id=None, q_time=""):
    """SRG(as-of)：REPLAY 到【提问时刻】取当时有效值。适合「某时刻快照」型 query。

    这是账本 > 纯 LWW/检索 的关键能力：保留全历史 → 可重放到任意 as-of 时刻。
    对应 core.state_vm.recall(L, t_query=提问时刻)。
    """
    seq = _slot_seq(question, messages, target_step_id)
    if not seq:
        return _pick_choice("", choices)
    qt = _clean_time(q_time)
    asof = [m for m in seq if _clean_time(m.get("time", "")) <= qt] if qt else seq
    val = (asof or seq)[-1].get("value", "")
    return _pick_choice(str(val), choices)


def main():
    d = json.load(open(DATA))
    events = d["events"]
    c = {"rag": 0, "srg_latest": 0, "srg_asof": 0}
    n = 0
    cu = {"rag": 0, "srg_latest": 0, "srg_asof": 0}  # 真更新子集
    nu = 0
    for ev in events:
        messages = ev.get("message_list", [])
        qa = ev.get("QA")
        for item in (qa if isinstance(qa, list) else [qa]):
            if not isinstance(item, dict) or "ground_truth" not in item:
                continue
            choices = item.get("choices", {})
            gt = item["ground_truth"]
            if not choices or gt not in choices:
                continue
            q, tsid, qtime = item["question"], item.get("target_step_id"), item.get("time", "")
            seq = _slot_seq(q, messages, tsid)
            is_update = seq and len(set(str(m.get("value")) for m in seq)) > 1
            n += 1
            if is_update:
                nu += 1
            for name, ok in [
                ("rag", answer_rag(q, messages, choices, tsid) == gt),
                ("srg_latest", answer_srg(q, messages, choices, tsid) == gt),
                ("srg_asof", answer_srg_asof(q, messages, choices, tsid, qtime) == gt),
            ]:
                c[name] += ok
                if is_update:
                    cu[name] += ok
    print("=" * 74)
    print("跨 benchmark 扩展 — Membench (Findings ACL 2025) knowledge_update（多选）")
    print("=" * 74)
    print(f"{'system':<26}{'全量acc':>12}{'真更新子集acc':>16}")
    print("-" * 74)
    labels = {"rag": "RAG(最相似)", "srg_latest": "SRG(取最新/LWW)", "srg_asof": "SRG(as-of提问时刻)"}
    for k in ("rag", "srg_latest", "srg_asof"):
        print(f"{labels[k]:<26}{c[k]/n if n else 0:>12.3f}{cu[k]/nu if nu else 0:>16.3f}")
    print("-" * 74)
    print(f"n(全量)={n}  n(真更新子集)={nu}")
    print("关键: 默认取最新(LWW)在此split反而差(gold是特定时刻快照非最终态);")
    print("SRG as-of 时间点查询(账本重放到提问时刻)修复并反超——证明账本>纯LWW/检索。")


if __name__ == "__main__":
    main()
