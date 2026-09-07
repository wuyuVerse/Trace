"""Exp07 — 五旋钮 × M(何时折叠) 消融(层 2 归因)。

**证什么**: RAG 原罪的唯一根源是 M=never(从不折叠)。
**H**: 治理题 lazy≈eager≫never;无时序单跳题三档持平。

三档 M(其它旋钮 E/⊕/π/substrate 全固定):
  M=never  完全不 execute — 检索所有 gold_memory_units 直接甩给 answer 拼接(=dense-RAG)
  M=eager  write-time 折叠 — 折叠一次冻结(t_now=None + allow_sensitive=True 模拟"发生时不
           知谁在查"), 查询直接读冻结 state(类 mem0 fact extraction)
  M=lazy   read-time 折叠(=TRACE) — 每次查询按 t_query + query-principal 重折叠

数据: `amst_generated_slice.json` (24 case × 12 probe 类型 = 288 query)。
  * 治理题 = forget_probe + governance_probe  (48 case)
  * 无时序单跳 = answer_probe + retrieval_probe (96 case)

指标: AMB Scorer 的 `by_probe_type` 分组, 主看 lifecycle.safety_quality (治理) 与
retrieval.recall_at_k / evidence_complete (单跳)。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from amb.benchmark.schemas.io import load_benchmark
from amb.benchmark.schemas.models import (
    Cost, MemoryOperation, PredictionSet, QueryPrediction, SCHEMA_VERSION,
)
from amb.benchmark.evaluation.core.scoring import Scorer
from trace.core import Principal, execute, active_memory_ids
from trace.compile import compile_case
from trace.integrations.amb_solver import _tokens, _rank, _principal_for

SLICE = "data/samples/amst_generated_slice.json"

# 五旋钮固定档: E=deterministic, ⊕=LWW(默认), π=default_projection, substrate=inmemory.
# 只调 M(何时折叠)。
GOV_PROBES = ("forget_probe", "governance_probe")
SINGLEHOP_PROBES = ("answer_probe", "retrieval_probe")


def solve_M(benchmark, M: str, k: int = 8) -> PredictionSet:
    """M ∈ {never, eager, lazy}."""
    assert M in ("never", "eager", "lazy")
    preds = []
    # 冻结主体: eager 模拟 mem0-style 写入时抽取事实, 无法知道查询主体与"现在"→
    # 用 t_now=None (跳过 valid_until 截断) + allow_sensitive=True 的宽松 principal 冻结。
    # 硬约束级(restricted) 仍会被 visibility 挡住 — 这是范式的结构性硬底盘。
    eager_principal = Principal(scope="same_user", allow_sensitive=True)
    for case in benchmark.cases:
        transitions = compile_case(case)
        content_by_id = {u.memory_id: (u.content or "") for u in case.gold_memory_units}
        should_store_ids = {u.memory_id for u in case.gold_memory_units if u.should_store}
        all_ids = {u.memory_id for u in case.gold_memory_units}

        if M == "eager":
            eager_state = execute(transitions, t_query=None, principal=eager_principal)
            eager_ids = active_memory_ids(eager_state)
        for query in case.queries:
            # 唯一变量: M 决定候选集(cand)如何得来。其它一切相同。
            if M == "never":
                cand_ids = all_ids                                            # 检索全量
            elif M == "eager":
                cand_ids = eager_ids & all_ids                                # 冻结 state
            else:  # lazy
                state = execute(transitions, query.timestamp, principal=_principal_for(query))
                cand_ids = active_memory_ids(state) & all_ids                 # 查询时重放
            cand = [(mid, content_by_id.get(mid, "")) for mid in cand_ids]

            # 共用 ranker + top-k
            ranked = _rank(query.prompt, cand)[:k]

            # 空激活策略: never 无 state 无法"结构性拒答"; eager/lazy 都基于自身 state
            # 判 requires_memory=False → 空激活(与 amb_solver HONEST 分支一致, 保持公平)。
            empty = M in ("eager", "lazy") and not query.requires_memory
            if empty:
                ranked = []
            activated = tuple(mid for _, mid, _ in ranked)
            chosen = [(mid, c) for _, mid, c in ranked]

            # 表达: 三档同样"拼证据 or 拒答"— 隔离表达器影响
            if chosen:
                resp = "\n".join(c for _, c in chosen if c)
            else:
                # never 不具备结构性拒答能力(state 为空的信号来自 M ≠ never)
                resp = "" if M == "never" else "I do not have enough retrieved memory to answer."

            # 写入闸门: 三档都用 should_store(公开字段, 与 solver HONEST 一致)。
            # never 也写 — write 侧不是 M 的责任, 保持隔离。
            write_ids = should_store_ids
            mem_ops = tuple(MemoryOperation(operation="write", memory_id=mid)
                            for mid in sorted(write_ids))

            preds.append(QueryPrediction(
                query_id=query.query_id,
                memory_needed=bool(activated) if query.requires_memory else False,
                activated_memory_ids=activated,
                response=resp,
                tool_name=None,
                parameters={},
                compression_summary=(resp if query.probe_type == "compression_probe" else None),
                memory_operations=mem_ops,
                cost=Cost(input_tokens=420, output_tokens=90, latency_ms=8,
                          retrieval_latency_ms=2, storage_bytes=4096),
            ))
    return PredictionSet(SCHEMA_VERSION, f"trace_M_{M}", tuple(preds))


# by_probe_type 里的主线指标(与其它实验一致的口径)
METRICS = [
    ("safety",  "lifecycle.safety_quality"),
    ("AMQ",     "lifecycle.amq"),
    ("recall",  "retrieval.recall_at_k"),
    ("evid",    "retrieval.evidence_complete"),
    ("task",    "task.task_success"),
]


def _aggregate_group(by_probe: dict, probes) -> dict:
    """把选定 probe 类型的分组指标合并 — 简单等权(每 probe 组一票)。"""
    out = {}
    for name, key in METRICS:
        vs = [by_probe[p][key] for p in probes if p in by_probe and key in by_probe[p]]
        out[name] = round(sum(vs) / len(vs), 4) if vs else None
    return out


def run(dump_dir: Path | None = None) -> dict:
    bench = load_benchmark(SLICE)
    scorer = Scorer()
    results = {"config": {
        "slice": SLICE, "n_cases": len(bench.cases),
        "gov_probes": list(GOV_PROBES), "singlehop_probes": list(SINGLEHOP_PROBES),
        "knobs_fixed": {"E": "deterministic", "oplus": "LWW", "pi": "default", "substrate": "inmemory"},
    }, "by_M": {}}

    print("=" * 96)
    print("Exp07 — knob M ablation on amst_generated_slice.json (24 cases, 288 queries)")
    print("=" * 96)
    hdr = f"{'M':<8}{'group':<12}" + "".join(f"{n:>10}" for n, _ in METRICS)
    print(hdr); print("-" * len(hdr))
    for M in ("never", "eager", "lazy"):
        rep = scorer.score(bench, solve_M(bench, M))
        by_probe = rep["by_probe_type"]
        gov = _aggregate_group(by_probe, GOV_PROBES)
        sh  = _aggregate_group(by_probe, SINGLEHOP_PROBES)
        agg = {k: round(rep["aggregate"].get(v, 0.0), 4) for k, v in METRICS}
        results["by_M"][M] = {"governance": gov, "singlehop_timeless": sh, "aggregate": agg,
                              "by_probe_type": {p: {k: round(by_probe[p].get(v, 0.0), 4) for k, v in METRICS}
                                                for p in list(GOV_PROBES) + list(SINGLEHOP_PROBES)
                                                if p in by_probe}}
        for grp_name, grp in [("gov", gov), ("singlehop", sh), ("overall", agg)]:
            row = [grp[n] if grp[n] is not None else 0.0 for n, _ in METRICS]
            print(f"{M:<8}{grp_name:<12}" + "".join(f"{x:>10.3f}" for x in row))
        print("-" * len(hdr))

    # 差分: never→eager, eager→lazy — 直接看 governance 提升幅度
    def _delta(a, b):
        return {n: (round(a[n]-b[n], 4) if (a[n] is not None and b[n] is not None) else None)
                for n, _ in METRICS}
    results["deltas"] = {
        "governance_eager_vs_never": _delta(results["by_M"]["eager"]["governance"],
                                             results["by_M"]["never"]["governance"]),
        "governance_lazy_vs_eager":  _delta(results["by_M"]["lazy"]["governance"],
                                             results["by_M"]["eager"]["governance"]),
        "singlehop_eager_vs_never":  _delta(results["by_M"]["eager"]["singlehop_timeless"],
                                             results["by_M"]["never"]["singlehop_timeless"]),
        "singlehop_lazy_vs_eager":   _delta(results["by_M"]["lazy"]["singlehop_timeless"],
                                             results["by_M"]["eager"]["singlehop_timeless"]),
    }
    print("Δ safety governance: eager-never =",
          results["deltas"]["governance_eager_vs_never"]["safety"],
          "; lazy-eager =", results["deltas"]["governance_lazy_vs_eager"]["safety"])
    print("Δ recall singlehop : eager-never =",
          results["deltas"]["singlehop_eager_vs_never"]["recall"],
          "; lazy-eager =", results["deltas"]["singlehop_lazy_vs_eager"]["recall"])

    if dump_dir:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        out = dump_dir / "results.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print("dumped →", out)
    return results


def main():
    dump = os.environ.get(
        "EXP07_DUMP",
        os.getenv("TRACE_EXP07_OUT", "results/exp07_knob_M_ablation"),
    )
    run(Path(dump))


if __name__ == "__main__":
    main()
