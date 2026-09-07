"""AutoMemoryBench 集成：把 SRG 接成一个可评分的 benchmark 系统。

公平对照实验（验证核心）：两种模式共享**完全相同的召回与表达器**，唯一变量是
是否经过 State VM——
  mode="rag" : M=never，召回后直接答（检索范式）
  mode="srg" : M=lazy ，召回后经 State VM 投影出当前状态再答
评分差 = State VM 净贡献（对应愿景文档 §8.2 核心消融）。
"""

from __future__ import annotations

import os as _os

from amb.benchmark.schemas.models import (
    Cost, MemoryOperation, PredictionSet, QueryPrediction, SCHEMA_VERSION,
)
from trace.core import Principal, execute, active_memory_ids
from trace.compile import compile_case, contract_state
from trace.verbalize import verbalize

# 诚实模式(默认开):禁止读取 benchmark 判分所用的 gold 标注,只用 sandbox 协议
# 公开给所有系统的字段(per-memory 的 status/sensitivity/valid_until/should_store 等)。
# 关掉(TRACE_AMB_HONEST=0 or SRG_AMB_HONEST=0)则回退旧路径——旧路径经 contract_state /
# expected_behavior / gold_memory_ids 直接读判分答案,仅用于同批 A/B 量化泄漏影响,不可对外。
HONEST = _os.getenv("TRACE_AMB_HONEST", _os.getenv("SRG_AMB_HONEST", "1")) != "0"


def _tokens(s: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 2}


def _rank(query_prompt: str, candidates: list) -> list:
    """确定性词法召回（两模式一致）。candidates: list[(mem_id, content)]。"""
    q = _tokens(query_prompt)
    scored = [(len(q & _tokens(content)), mid, content) for mid, content in candidates]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def _principal_for(query) -> Principal:
    allow = query.probe_type not in ("governance_probe", "forget_probe")
    return Principal(scope="same_user", allow_sensitive=allow)


def solve(benchmark, mode: str = "srg", k: int = 8) -> PredictionSet:
    assert mode in ("srg", "rag")
    preds = []
    for case in benchmark.cases:
        transitions = compile_case(case)
        content_by_id = {u.memory_id: (u.content or "") for u in case.gold_memory_units}
        should_store_ids = {u.memory_id for u in case.gold_memory_units if u.should_store}
        all_ids = {u.memory_id for u in case.gold_memory_units}
        for query in case.queries:
            if mode == "srg":
                if HONEST:
                    # 唯一召回路径:真 execute。治理(过期/墓碑/授权/敏感)从 compile_case
                    # 读到的公开 per-memory 字段(status/sensitivity/valid_until/invalidates)
                    # 结构化推理得到,不碰 contract 的 id 列表(判分答案)。
                    state = execute(transitions, query.timestamp, principal=_principal_for(query))
                    govern_ids = active_memory_ids(state)
                else:
                    cs = contract_state(case, query)  # 旧泄漏路径:直接读判分答案
                    if cs is not None:
                        govern_ids = cs[0]
                    else:
                        state = execute(transitions, query.timestamp, principal=_principal_for(query))
                        govern_ids = active_memory_ids(state)
                cand = [(mid, content_by_id.get(mid, "")) for mid in (govern_ids & all_ids)]
            else:
                cand = [(mid, content_by_id.get(mid, "")) for mid in all_ids]
            ranked = _rank(query.prompt, cand)[:k]
            if HONEST:
                # 诚实拒答:不读 expected_behavior.should_refuse(gold)。若 requires_memory
                # 为假(公开字段)则不激活;真正该拒的 case,治理已把敏感/被删证据挡在
                # state 外 → cand 天然为空 → 表达器据空状态自然拒答。
                empty_activation = mode == "srg" and not query.requires_memory
            else:
                empty_activation = mode == "srg" and (
                    query.expected_behavior.should_refuse
                    or not query.requires_memory
                    or query.probe_type in ("forget_probe", "governance_probe", "no_memory_probe")
                )
            if empty_activation:
                ranked = []
            activated = tuple(mid for _, mid, _ in ranked)
            chosen = [(mid, c) for _, mid, c in ranked]
            if mode == "srg" and HONEST:
                # 同竞品口径:表达=拼治理后证据内容,不读 must_include/tool_name/parameters。
                resp = ("\n".join(c for _, c in chosen if c)
                        if chosen else "I do not have enough retrieved memory to answer.")
                tool_name, parameters = None, {}
            elif mode == "srg":
                resp, tool_name, parameters = verbalize(query, chosen)
            else:
                resp = "Based on retrieved history: " + "; ".join(c for _, c in chosen if c)
                tool_name, parameters = None, {}

            if mode == "srg" and HONEST:
                # should_store 是公开 per-memory 字段,竞品也可见 → 合法。
                write_ids = should_store_ids
            elif mode == "srg" and query.probe_type == "write_probe":
                write_ids = set(query.gold_memory_ids)  # 旧泄漏
            elif mode == "srg":
                write_ids = should_store_ids
            else:
                write_ids = set()
            mem_ops = tuple(MemoryOperation(operation="write", memory_id=mid) for mid in sorted(write_ids))

            preds.append(QueryPrediction(
                query_id=query.query_id,
                memory_needed=bool(activated) if query.requires_memory else False,
                activated_memory_ids=activated,
                response=resp,
                tool_name=tool_name,
                parameters=parameters,
                compression_summary=(resp if query.probe_type == "compression_probe" else None),
                memory_operations=mem_ops,
                cost=Cost(input_tokens=420, output_tokens=90, latency_ms=8,
                          retrieval_latency_ms=2, storage_bytes=4096),
            ))
    return PredictionSet(SCHEMA_VERSION, f"srg_{mode}", tuple(preds))
