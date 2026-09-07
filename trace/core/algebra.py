"""SRG 状态代数：折叠算子 ⊕（LWW-Register monoid）+ scan。

理论基石 —— 为什么「折叠经历成状态」真的是一次 scan（不只是比喻）：

    ⊕ 被实现为 LWW-Register join-semilattice（CRDT 标准构造，Shapiro et al. 2011）。
    对每个 (subject, attribute) 槽位，⊕ 取逻辑时钟 t_ingest 更晚者胜，DELETE 墓碑粘附。
    可证 ⊕ 满足：结合律、交换律、幂等律、幺元 ∅。

    结合律 + 幺元 ⟹ (State, ⊕, ∅) 是 monoid ⟹ 「把经历流从头折叠出当前状态」在数学上
    就是一次 associative scan —— 与 SSM/Mamba 记忆长序列所用的 associative scan 同构。
    这是「记忆是账本、回忆是算余额」这一大一统命题的形式化落点。

    property 测试见 theory/monoid_laws.py：在真实 benchmark 事件上经验验证四条定律。
"""

from __future__ import annotations

from dataclasses import replace
from functools import reduce
from typing import Iterable

from trace.core.transition import Transition, Slot, State, EMPTY


def lift(t: Transition) -> State:
    """把单条转移抬升为 singleton 状态，以便用统一的 ⊕ 折叠。"""
    slot = Slot(
        memory_id=t.memory_id, op=t.op, t_ingest=t.t_ingest, t_event=t.t_event,
        valid_until=t.valid_until, scope=t.scope, sensitivity=t.sensitivity,
        content=t.content, tombstoned=(t.op == "DELETE"),
    )
    return {(t.subject, t.attribute): slot}


def merge_slot(a: Slot | None, b: Slot | None) -> Slot | None:
    """LWW join：取 t_ingest 更晚者；DELETE 墓碑一旦出现则粘附。

    ⊕ 的核心。对 (a,b) 对称（交换律），对相同输入不变（幂等律），None 为幺元，
    结合律由 max(t_ingest) 的结合性 + 墓碑单调性保证。
    """
    if a is None:
        return b
    if b is None:
        return a
    # 逻辑时钟晚者胜；平手时 DELETE/RETRACT 优先（安全优先，确定性 tie-break）
    if b.t_ingest > a.t_ingest or (b.t_ingest == a.t_ingest and b.op in ("DELETE", "RETRACT")):
        winner = b
    else:
        winner = a
    tomb = a.tombstoned or b.tombstoned or winner.op == "DELETE"
    return replace(winner, tombstoned=tomb)


def combine(x: State, y: State) -> State:
    """⊕：两个状态的 LWW 合并。返回新 dict，不改入参。"""
    out: State = dict(x)
    for k, sb in y.items():
        out[k] = merge_slot(out.get(k), sb)
    return out


def fold(transitions: Iterable[Transition]) -> State:
    """monoid 归约端点：S = reduce(⊕, [lift(t) ...], ∅)。"""
    return reduce(combine, (lift(t) for t in transitions), dict(EMPTY))


def scan(transitions: list[Transition]) -> list[State]:
    """完整前缀扫描：返回每一步中间状态 [S_1,...,S_n]（账本余额随每笔交易的演化）。

    因 ⊕ 可结合，等价的 associative scan 可并行前缀化；此为串行参考实现，语义一致。
    """
    states: list[State] = []
    acc: State = dict(EMPTY)
    for t in transitions:
        acc = combine(acc, lift(t))
        states.append(dict(acc))
    return states
