"""TraceMemory —— SRG 的顶层门面（协议无关）。

这是所有集成（MCP server / Codex hook / OpenCode plugin / benchmark 适配）共用的入口。
围绕【记忆可采纳性】组织动作：

    observe(episode, who, t)         把一段经历编译成转移、写入账本
    admissible(t, who)               ★ 记忆第二半：返回 Adm(t,p) + 四类不可采纳审计
    recall(query, t, who, view)      按投影视角(current_value/temporal_order/…)取可采纳证据
    fork(delta, ..., t, who)         反事实：分叉账本重放
    check(memory_id, t, who)         ★ 准入判定：该记忆此刻能否被采纳（动作层守门）
    why(memory_id)                   审计：某记忆的转移历史（在/不在的因果）

设计原则（可扩展 + 易接入）：纯 Python、无外部依赖、不进关键路径、失败可降级；
投影算子/治理规则均可注册扩展；任何 agent/bench 通过这一个门面接入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trace.core import (
    Transition, Principal, execute_counterfactual as _cf,
    admissible as _admissible, is_admissible as _is_adm, project_for_query, audit as _audit,
)
from trace.compile import compile_episode, gate
from trace.runtime.llm_client import DEFAULT_MODEL


@dataclass
class TraceMemory:
    """一个 principal 命名空间下的账本记忆。"""
    model: str = DEFAULT_MODEL
    use_llm_compiler: bool = True
    ledger: list[Transition] = field(default_factory=list)
    _clock: int = 0

    # ---- 写入 ----
    def observe(self, episode: str, *, who: str = "same_user", t: str = "") -> list[Transition]:
        """把一段自由文本经历编译成转移并写入账本。返回本次新增的转移。"""
        self._clock += 1
        if self.use_llm_compiler:
            new = compile_episode(episode, t_event=t, t_ingest=self._clock, model=self.model)
        else:
            new = []
        new = gate(new)
        # 重写 scope（principal 命名空间）
        stamped = [Transition(**{**tr.__dict__, "scope": who}) if who != "same_user" else tr for tr in new]
        self.ledger.extend(stamped)
        return stamped

    def observe_transitions(self, transitions: list[Transition]) -> None:
        """直接写入已构造好的转移（供确定性/离线场景）。"""
        self.ledger.extend(gate(transitions))

    # ---- 可采纳性（记忆第二半，一等能力）----
    def admissible(self, *, t: str | None = None, who: str = "same_user",
                   allow_sensitive: bool = True) -> dict:
        """★ 返回 Adm(t,p)：此刻可被采纳的记忆 + 每条被排除记忆的理由（stale/superseded/deleted/unauthorized）。"""
        p = Principal(scope=who, allow_sensitive=allow_sensitive)
        r = _admissible(self.ledger, t, p)
        return {
            "admissible": [{"subject": pp.subject, "attribute": pp.attribute, "value": pp.value,
                            "valid_until": pp.valid_until, "source": pp.memory_id}
                           for pp in r.admissible],
            "excluded": r.excluded,
            "admissible_ids": sorted(r.admissible_ids),
            "forbidden_ids": sorted(r.forbidden_ids),
        }

    def check(self, memory_id: str, *, t: str | None = None, who: str = "same_user",
              allow_sensitive: bool = True) -> bool:
        """★ 准入守门：memory_id 在 (t,who) 下能否被采纳。动作层调它防止用了该忘的记忆。"""
        return _is_adm(self.ledger, memory_id, t, Principal(scope=who, allow_sensitive=allow_sensitive))

    # ---- 读出（投影视角）----
    def recall(self, query: str = "", *, t: str | None = None, who: str = "same_user",
               allow_sensitive: bool = True, view: str = "current_value") -> dict:
        """按投影视角取可采纳证据。view ∈ current_value/temporal_order/aggregate（可注册扩展）。

        证据只含可采纳记忆；excluded 是被治理排除的（可审计、供判"是否误用该忘的记忆"）。
        """
        p = Principal(scope=who, allow_sensitive=allow_sensitive)
        return project_for_query(self.ledger, t, p, view=view, query=query)

    def fork(self, delta: list[Transition] | None = None, *, remove_memory_ids: set[str] | None = None,
             query: str = "", t: str | None = None,
             who: str = "same_user", allow_sensitive: bool = True) -> dict:
        """反事实回忆：注入假设转移 Δ（delta）和/或移除某些原转移（remove_memory_ids）后重放。"""
        p = Principal(scope=who, allow_sensitive=allow_sensitive)
        state = _cf(self.ledger, remove_memory_ids=remove_memory_ids, inject=delta or [],
                    t_query=t, principal=p)
        return {"state": [{"subject": pp.subject, "attribute": pp.attribute, "value": pp.value,
                           "source": pp.memory_id} for pp in state]}

    # ---- 审计 ----
    def why(self, memory_id: str) -> list[dict]:
        """返回涉及某 memory_id 的转移历史（因果链，为何在/不在当前可采纳集）。"""
        return _audit(self.ledger, memory_id)
