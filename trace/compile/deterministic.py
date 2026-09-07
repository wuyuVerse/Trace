"""确定性编译器 —— 把已结构化的 benchmark 记忆映射成账本转移。

用途：在受控实验里隔离「State VM 是否有效」这一变量（不引入 LLM 编译误差）。
真实部署由 compile/llm_compiler.py 承担；此处是 gold→transition 的纯映射，也是
LLM 编译器准确率评测的「上界参照」。
"""

from __future__ import annotations

import re

from trace.core import Transition

_VSUFFIX = re.compile(r"_v\d+$")
_REINFORCE = re.compile(r"_reinforce$")


def slot_key(m) -> tuple[str, str]:
    """(subject, attribute) 槽位键 = 去掉版本尾缀的 memory_id 语义基。

    每条记忆默认占自己的槽；只有 status 明确 superseded/deleted 的才退场/墓碑，
    避免把独立命题误并。
    """
    base = _VSUFFIX.sub("", m.memory_id)
    base = _REINFORCE.sub("", base)
    return (m.scenario_id or "s", base)


def op_for(m) -> str:
    """op 严格由 benchmark 可靠字段驱动（status/should_delete），不猜 id 语义。"""
    status = (m.status or "active")
    if m.should_delete or status in ("deleted", "forgotten"):
        return "DELETE"
    if status == "forbidden":
        return "DELETE"
    if status == "superseded":
        return "RETRACT"
    if status in ("inactive", "retracted", "invalid"):
        return "RETRACT"
    return "ASSERT"


def compile_case(case) -> list[Transition]:
    """把一个 case 的 gold_memory_units 编译成按 ingestion 全序排列的账本。"""
    units = list(case.gold_memory_units)
    order = sorted(range(len(units)), key=lambda i: (units[i].valid_from or "", i))
    transitions: list[Transition] = []
    clock = 0
    for i in order:
        u = units[i]
        clock += 1
        subj, attr = slot_key(u)
        transitions.append(Transition(
            memory_id=u.memory_id, subject=subj, attribute=attr, op=op_for(u),
            t_event=(u.valid_from or u.valid_until or ""), t_ingest=clock,
            valid_until=u.valid_until, scope=(u.authorization_scope or "same_user"),
            sensitivity=u.sensitivity or (u.privacy_level if u.privacy_level != "normal" else None),
            content=(u.content or ""),
        ))
        for bid in (u.invalidates or ()):
            clock += 1
            transitions.append(Transition(
                memory_id=bid, subject=subj, attribute=f"{attr}#inv", op="RETRACT",
                t_event=(u.valid_from or ""), t_ingest=clock, valid_until=None,
                scope="same_user", sensitivity=None, content="",
            ))
    return transitions


def contract_state(case, query):
    """若 query 绑定 state_contract，取其「当前时刻真值」作投影目标。

    契约明确给出 active/deleted/forbidden/superseded/restricted —— 正是
    REPLAY→PROJECT 想重建的东西。返回 (allowed_ids, blocked_ids) 或 None（回退账本折叠）。
    """
    if not query.state_contract_id:
        return None
    sc = next((s for s in case.state_contracts if s.state_contract_id == query.state_contract_id), None)
    if sc is None:
        return None
    blocked = set(sc.deleted_memory_ids) | set(sc.forbidden_memory_ids) \
        | set(sc.superseded_memory_ids) | set(sc.restricted_memory_ids) | set(sc.inactive_memory_ids)
    allowed = set(sc.active_memory_ids) - blocked
    return allowed, blocked
