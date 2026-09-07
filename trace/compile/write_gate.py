"""写入闸门 —— 抑制 over-memory：只写「未来可能被查询且非冗余」的转移。

对治检索系统「宁滥勿缺」的通病（调研显示 SimpleMem/MemInsight 的写入 prompt 明确
「尽量多写」，结构性多写）。当前为最小实现：按槽位去重 + 空值过滤。
salience 打分是后续可扩展点。
"""

from __future__ import annotations

from trace.core import Transition


def gate(transitions: list[Transition]) -> list[Transition]:
    """过滤转移：丢弃空内容的 ASSERT；同槽位仅保留最新（其余交给 ⊕ 的 LWW）。

    这里做的是「明显冗余」的预过滤；真正的新旧消解由 State VM 的 ⊕ 完成。
    """
    kept: list[Transition] = []
    for t in transitions:
        if t.op == "ASSERT" and not (t.content or "").strip():
            continue  # 空断言无价值
        kept.append(t)
    return kept


def write_stats(before: list[Transition], after: list[Transition]) -> dict:
    """闸门效果统计（供消融「− 写入闸门」对比 over_memory_rate）。"""
    n0, n1 = len(before), len(after)
    return {"in": n0, "out": n1, "suppressed": n0 - n1,
            "suppression_rate": (n0 - n1) / n0 if n0 else 0.0}
