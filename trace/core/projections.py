"""投影算子谱系（Projection Algebra）—— 可扩展、可注册。

统一范式：一切记忆查询 = 对经历流的一次投影。查询视角决定用哪个投影算子。
所有算子都在**可采纳集 Adm(t,p) 之上**运行——即先保证"此刻可采纳"，再按视角组织答案。
这样时序/聚合等能力不牺牲可采纳性保证（该忘的仍不被用）。

之前 LongMemEval 端到端崩，根因是只实现了 last_write 一个算子（问"哪个先"却取最新值）。
本模块把算子做成可注册谱系，覆盖 current-value / temporal-order / aggregate，可继续扩展。

算子签名: op(admissible_state, transitions_in_scope, t_query, principal) -> 结构化证据(list[dict])
表达器拿这个结构化证据作答，仍然只看可采纳记忆。
"""

from __future__ import annotations

from trace.core.transition import Transition
from trace.core.admissibility import admissible, AdmissibilityResult
from trace.core.relevance import rank_relevant

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name):
    return _REGISTRY.get(name, _REGISTRY["current_value"])


def available() -> list[str]:
    return sorted(_REGISTRY)


@register("current_value")
def _current_value(adm: AdmissibilityResult, transitions, t_query, p) -> list[dict]:
    """当前值投影：问"现在的 X"。取可采纳状态里每命题的当前值。
    结构读跳过 narrative 语言体命题（那是 read_relevant 的召回素材，非结构状态）。"""
    return [{"subject": pp.subject, "attribute": pp.attribute, "value": pp.value, "t_event": pp.t_event}
            for pp in adm.admissible if pp.value and not pp.is_narrative]


@register("temporal_order")
def _temporal_order(adm: AdmissibilityResult, transitions, t_query, p) -> list[dict]:
    """时序投影：问"X 和 Y 哪个先 / 事件顺序"。

    不折叠成当前值——保留【可采纳记忆对应的事件】并按 t_event 排序。
    只纳入其 memory_id 在可采纳集里的事件（该忘的事件不参与时序推理）。
    """
    adm_ids = adm.admissible_ids
    events = [{"t_event": t.t_event, "subject": t.subject, "attribute": t.attribute,
               "value": t.content, "op": t.op}
              for t in transitions
              if t.memory_id in adm_ids and t.op in ("ASSERT", "SUPERSEDE") and t.content]
    events.sort(key=lambda e: str(e["t_event"]))
    return events


@register("aggregate")
def _aggregate(adm: AdmissibilityResult, transitions, t_query, p) -> list[dict]:
    """聚合投影：问"多少个 / 综合多次"。按 subject 分组列出可采纳事件，供计数/综合。"""
    adm_ids = adm.admissible_ids
    groups: dict[str, list] = {}
    for t in transitions:
        if t.memory_id in adm_ids and t.content:
            groups.setdefault(t.subject, []).append(
                {"attribute": t.attribute, "value": t.content, "t_event": t.t_event})
    return [{"subject": s, "items": items, "count": len(items)} for s, items in groups.items()]


@register("relevant")
def _relevant(adm: AdmissibilityResult, transitions, t_query, p, query: str = "", k: int = 8) -> list[dict]:
    """★相关性读算子 read_relevant（0722 §3）：一切读=对 S(t) 的读算子，这是第 4 个。

    读富 S(t) 每命题的【语言体】(body=value 原文 + evidence_keys)，按与 query 的词法相关性
    (BM25 式 token 重叠 + key 命中) 排序取 top-k。确定性、无 LLM、无外部 embedding 依赖。
    关键：输入是 adm.admissible（已 execute+GOVERN 的可采纳集）——被删/过期/越权命题连 body 从不在此，
    故召回结构上不可能捞到禁忌/旧值（定理2级"召回也合规"，非工程约定）。dense 可作可选增强(见 retrieve/)。
    """
    hits = rank_relevant(adm.admissible, query, k=k)   # 单一权威评分（core/relevance.py）
    return [{"subject": pp.subject, "attribute": pp.attribute, "value": pp.value,
             "t_event": pp.t_event, "source": pp.memory_id}
            for pp in hits]


def project_for_query(transitions: list[Transition], t_query, p, view: str = "current_value",
                      query: str = "") -> dict:
    """统一入口：算可采纳集 + 按视角投影出结构化证据 + 附审计。

    返回 {view, evidence, excluded, admissible_ids}——evidence 只含可采纳记忆，
    excluded 是被治理排除的（供审计/判"是否误用该忘的记忆"）。
    """
    adm = admissible(transitions, t_query, p)
    fn = get(view)
    # relevant 算子需要 query 文本；其余算子签名兼容（**忽略多余 kw）。
    try:
        evidence = fn(adm, transitions, t_query, p, query=query) if view == "relevant" else fn(adm, transitions, t_query, p)
    except TypeError:
        evidence = fn(adm, transitions, t_query, p)
    return {"view": view, "evidence": evidence, "excluded": adm.excluded,
            "admissible_ids": sorted(adm.admissible_ids)}
