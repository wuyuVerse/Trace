"""回忆凭证（Recall Certificate）—— 核心创新的更高一层：回忆即【可自证的证明】。

一句话：别的记忆只能给答案，SRG 的记忆能给出答案的【证明】，并同时证明它没用任何不该用的记忆。

为什么这是核心创新的纯粹推论（不是新子系统）：
    回忆 = 执行程序到 t 得 S(t)（唯一核心）。因为回忆是【程序执行】，每个答案天然带一个
    可独立验证的凭证 C(t,q)，由已有原语组合而成：
      · 最小指令切片 slice   —— 只有真正推出答案命题的那几条指令（program slicing）。重放它得同答案。
      · 排除报告 excluded    —— 哪些记忆被【故意没用】（deleted/retracted/expired/unauthorized）+理由。
      · 答案命题 answer_pids —— 答案落在 S(t) 的哪些命题上。
    三者皆从 execute/trace/excluded_report 导出，同一引擎，零缝合。

凭证的三个别人给不了的性质（皆可确定性核验，见 verify()）：
    可核验 soundness    —— 重放 slice → 得到与完整程序【相同】的目标命题（不是"相信 LLM"，是数学重放）。
    最小   minimality   —— slice 里每条指令都必要：去掉任一条，目标命题就变（无冗余）。
    完备   completeness —— excluded 覆盖所有被排除的禁忌命题；答案里不含任何禁忌命题（无泄漏）。

现有记忆系统（mem0/Zep/RAG）返回相似 chunk，结构性无法出示推导、无法证明没用禁忌记忆 —— 此维度 0 分。
SRG 每次回忆 = 一个可审计的证明对象。这是"可问责记忆（accountable memory）"新范畴。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trace.core.transition import Transition
from trace.core.governance import Principal
from trace.core.machine import Proposition, execute, excluded_report


@dataclass(frozen=True)
class Certificate:
    """一次回忆的证明对象 C(t,q)。可序列化、可独立重放核验。"""
    t_query: str | None
    answer_pids: tuple[tuple[str, str], ...]          # 答案命题 (subject, attribute)
    slice: tuple[Transition, ...]                      # 最小指令切片：推出答案的指令
    excluded: tuple[dict, ...]                         # 被执行语义排除的禁忌记忆 + 理由
    principal: Principal | None = None
    meta: dict = field(default_factory=dict)

    # ---- 可读呈现（供审计/表达器/日志）----
    def explain(self) -> str:
        lines = ["答案依据以下指令推出（重放可核验）："]
        for ins in sorted(self.slice, key=lambda i: i.t_ingest):
            lines.append(f"  [{ins.t_event}] {ins.op} {ins.subject}.{ins.attribute} = {ins.content}")
        if self.excluded:
            lines.append("以下记忆被【故意排除】，未参与作答：")
            for e in self.excluded:
                lines.append(f"  ✗ {e['subject']}.{e['attribute']} —— {e['reason']}")
        return "\n".join(lines)


def _target_props(program: list[Transition], t_query, principal, answer_pids):
    """执行完整程序，取答案命题在 S(t) 上的取值（作为核验基准）。"""
    S = {p.pid: p for p in execute(program, t_query, principal=principal)}
    return {pid: S[pid].value for pid in answer_pids if pid in S}


def certify(program: list[Transition], answer_pids, *, t_query=None,
            principal: Principal | None = None) -> Certificate:
    """为一次回忆生成凭证：对答案命题做 program slicing + 附排除报告。

    最小切片算法（确定性）：一条指令进入切片，当且仅当它作用于某个答案命题 (s,a)，
    且是【执行到 t_query 时真正影响该命题最终成立性】的指令——即从最后一次 DELETE/该命题
    存活起点到最终值的指令链（更早被覆盖掉的历史值不进切片，因为它们不 entail 最终答案）。
    """
    answer_pids = [tuple(p) for p in answer_pids]
    insns = sorted(program, key=lambda i: i.t_ingest)
    want = set(answer_pids)

    # 对每个答案命题，找"决定其最终成立值"的指令链：最后一条使其成立的 ASSERT/SUPERSEDE，
    # 以及在它之前、该命题被 DELETE/RETRACT 后重新起算的边界（保证重放 slice 得同值）。
    keep_ids: set[str] = set()
    for pid in want:
        chain = [i for i in insns
                 if (i.subject, i.attribute) == pid
                 and not (t_query is not None and i.t_event and i.t_event > t_query)]
        if not chain:
            continue
        # 从后往前：最后一条 ASSERT/SUPERSEDE 决定最终值；若其后有 DELETE/RETRACT 则该命题不成立（无切片）。
        last = chain[-1]
        if last.op in ("DELETE", "RETRACT"):
            continue  # 该命题最终不成立，不该出现在答案里（completeness 会抓）
        keep_ids.add(last.memory_id)

    slice_insns = tuple(i for i in insns if i.memory_id in keep_ids)
    excluded = tuple(excluded_report(program, t_query, principal))
    return Certificate(
        t_query=t_query,
        answer_pids=tuple(answer_pids),
        slice=slice_insns,
        excluded=excluded,
        principal=principal,
    )


def verify(cert: Certificate, program: list[Transition]) -> dict:
    """独立核验凭证的三性质（确定性，不依赖 LLM）。返回 {sound, minimal, complete, ok}。

    soundness   : 重放 slice → 目标命题取值 == 重放完整 program → 同命题取值。
    minimality  : slice 中每条指令都必要（去掉任一条，至少一个目标命题的值/成立性改变）。
    complete    : (a) excluded 覆盖所有被完整执行排除的禁忌命题；
                  (b) 答案命题里【没有】任何禁忌命题（无泄漏）。
    """
    p = cert.principal
    full_vals = _target_props(program, cert.t_query, p, cert.answer_pids)

    # soundness：只用 slice 重放
    slice_vals = _target_props(list(cert.slice), cert.t_query, p, cert.answer_pids)
    sound = slice_vals == full_vals

    # minimality：逐条移除，看目标命题是否改变
    minimal = True
    for ins in cert.slice:
        reduced = [i for i in cert.slice if i.memory_id != ins.memory_id]
        if _target_props(reduced, cert.t_query, p, cert.answer_pids) == full_vals:
            minimal = False  # 去掉它结果不变 → 冗余
            break

    # completeness：excluded 覆盖 + 答案无禁忌泄漏
    excluded_pids = {(e["subject"], e["attribute"]) for e in cert.excluded}
    # 完整执行下被排除的禁忌命题（无授权门的全体 vs 有门/删除后的）
    report_now = {(e["subject"], e["attribute"]) for e in excluded_report(program, cert.t_query, p)}
    covers = report_now.issubset(excluded_pids)
    no_leak = not (set(cert.answer_pids) & report_now)  # 答案不含任何被排除的禁忌命题
    complete = covers and no_leak

    return {"sound": sound, "minimal": minimal, "complete": complete,
            "ok": sound and minimal and complete}
