"""Memory Admissibility —— SRG 新范式的一等概念（记忆的第二半问题）。

Recall 问"哪条最相关"；Admissibility 问"哪条此刻还算数、能被采纳进答案/动作"。

可采纳集  Adm(t, p) = { m : 在 t 时刻成立 ∧ 未删除/撤销 ∧ 对 p 授权 ∧ 未被更新推翻 }
    = GOVERN( PROJECT( REPLAY(L, t), t ), p )   —— 恰好是 State VM 的投影结果。

本模块把这个"恰好等于"显式化，并给出**带理由的审计**（每条被排除的记忆属于哪类不可采纳），
让"该忘的一定不被用"成为可插拔、可证明、可解释的核心能力，而非埋在 state_vm 里的副作用。

四类不可采纳（现有 benchmark 全不测的负半边）：
    stale         已过期（valid_until < t）
    superseded    被更新推翻（RETRACT / 同槽新值取代）
    deleted       治理删除 / 遗忘（DELETE 墓碑）
    unauthorized  越权（scope/sensitivity 对 p 不可见）
"""

from __future__ import annotations

from dataclasses import dataclass

from trace.core.transition import Transition
from trace.core.governance import Principal
from trace.core.machine import Proposition, execute, excluded_report

# 不可采纳类型
STALE, SUPERSEDED, DELETED, UNAUTHORIZED = "stale", "superseded", "deleted", "unauthorized"


@dataclass(frozen=True)
class AdmissibilityResult:
    """一次可采纳性裁决：可采纳状态 + 被排除项的理由（可审计）。

    admissible 直接是执行引擎的输出 S(t)=list[Proposition]（该忘的从没执行出来），
    excluded 是 excluded_report 的审计。一个引擎派生，无第二实现。
    """
    admissible: list[Proposition]           # Adm(t,p)=S(t)：可被采纳的成立命题
    excluded: list[dict]                    # [{subject, attribute, memory_id, reason}]

    @property
    def admissible_ids(self) -> set[str]:
        return {pp.memory_id for pp in self.admissible}

    @property
    def forbidden_ids(self) -> set[str]:
        """不可采纳的记忆 id（供 benchmark 判"是否误用了该忘的记忆"）。"""
        return {e["memory_id"] for e in self.excluded if e.get("memory_id")}


def admissible(transitions: list[Transition], t_query: str | None, p: Principal) -> AdmissibilityResult:
    """计算 Adm(t,p) 并给出每条被排除记忆的理由（四类不可采纳）。

    唯一引擎派生：Adm(t,p) = execute(L,t,p)（S(t) 恰好是可采纳集，该忘的从没执行出来）；
    理由审计由 excluded_report 从指令流重建。表达器/动作层只应绑定 result.admissible。
    """
    return AdmissibilityResult(
        admissible=execute(transitions, t_query, principal=p),
        excluded=excluded_report(transitions, t_query, p),
    )


def is_admissible(transitions: list[Transition], memory_id: str,
                  t_query: str | None, p: Principal) -> bool:
    """单条查询：memory_id 在 (t,p) 下是否可采纳。供动作层作准入判定。"""
    return memory_id in admissible(transitions, t_query, p).admissible_ids


def audit(transitions: list[Transition], memory_id: str) -> list[dict]:
    """审计某条记忆的全部转移历史（为何在/不在当前状态的因果链）。"""
    return [{"op": t.op, "subject": t.subject, "attribute": t.attribute,
             "value": t.content, "t_event": t.t_event, "t_ingest": t.t_ingest,
             "scope": t.scope, "sensitivity": t.sensitivity}
            for t in transitions if t.memory_id == memory_id]
