"""SRG 核心层：唯一引擎（回忆即执行）+ 账本代数（理论基石）+ 可采纳性 + 凭证。

唯一引擎：machine.execute —— 一切能力（当前值/计数/时序/可采纳性/反事实/凭证）都从它派生。
分层：
  transition.py     —— 双时态转移与槽位数据类型
  algebra.py        —— 折叠算子 ⊕（LWW monoid）+ scan（理论基石：证明 execute ≡ fold）
  governance.py     —— Principal + 可见性（GOVERN 单元，两处共用）
  machine.py        —— ★ 唯一引擎：execute→S(t)；execute_counterfactual/excluded_report/trace/read_*
  admissibility.py  —— ★ 记忆第二半：Adm(t,p)=execute 输出 + excluded_report 审计（由引擎派生）
  projections.py    —— ★ 可扩展投影算子谱系（current_value/temporal_order/aggregate…）
  certificate.py    —— ★ 回忆即可自证的证明：certify/verify（sound/minimal/complete）
"""

from trace.core.transition import (
    Transition, Slot, State, EMPTY, OPS,
)
from trace.core.algebra import lift, merge_slot, combine, fold, scan
from trace.core.governance import Principal, visible
from trace.core.machine import (
    Proposition, execute, execute_counterfactual, trace,
    read_current, read_count, read_temporal, read_intervals, read_slot_history, read_amounts, excluded_report,
    active_memory_ids, admissible_state,
)
from trace.core.admissibility import (
    admissible, is_admissible, audit, AdmissibilityResult,
    STALE, SUPERSEDED, DELETED, UNAUTHORIZED,
)
from trace.core.projections import (
    project_for_query, register as register_projection, get as get_projection, available as available_projections,
)
from trace.core.certificate import Certificate, certify, verify
from trace.core.machine import NARRATIVE_ATTR
from trace.core.relevance import tokenize, rank_relevant, register_scorer, lexical_score

__all__ = [
    "Transition", "Slot", "State", "EMPTY", "OPS",
    "lift", "merge_slot", "combine", "fold", "scan",
    "Principal", "visible",
    "Proposition", "execute", "execute_counterfactual", "trace",
    "read_current", "read_count", "read_temporal", "read_intervals", "read_slot_history", "read_amounts", "excluded_report",
    "active_memory_ids", "admissible_state",
    "admissible", "is_admissible", "audit", "AdmissibilityResult",
    "STALE", "SUPERSEDED", "DELETED", "UNAUTHORIZED",
    "project_for_query", "register_projection", "get_projection", "available_projections",
    "Certificate", "certify", "verify",
    "NARRATIVE_ATTR", "tokenize", "rank_relevant", "register_scorer", "lexical_score",
]
