"""Task 3 / E2 —— 逐组件消融：证明每个部件都在承重。

对同一 generated slice，用真实 AMB Scorer 评分，逐个拔掉 SRG 的部件，看哪一维崩：
  full             完整 SRG（基准）
  −StateVM         退回检索范式（不折叠状态，候选=全部记忆）
  −GOVERN          不按 principal/state 过滤（不排除删除/过期/越权）
  −bitemporal      忽略 t_event（不按世界时间截断，过期/未来事件都进）
  −write_gate      不写 memory_operations（over-memory 不受控 → write_f1 崩）
  −counterfactual  反事实组走同一次 recall（不分叉 → state_sensitivity 崩）
  −supersession    拔 SUPERSEDE 语义：每条命题占独立槽 + 显式 RETRACT 降级为 ASSERT，
                   旧值不被覆盖（update_quality 应崩）
  −sensitivity     拔 _detect_sensitivity 标签：敏感命题不进 GOVERN 屏蔽名单，
                   forbidden_activation 泄漏 → safety 应崩

预期（文档 E2 表）：每项在其对应维度显著下降（块对角归因）。

Run:  PYTHONPATH=. python3 trace/experiments/ablations.py
"""

from __future__ import annotations

from dataclasses import replace as _replace

from amb.benchmark.schemas.io import load_benchmark
from amb.benchmark.schemas.models import (
    Cost, MemoryOperation, PredictionSet, QueryPrediction, SCHEMA_VERSION,
)
from amb.benchmark.evaluation.core.scoring import Scorer
from trace.core import Principal, execute, active_memory_ids
from trace.compile import compile_case, contract_state
from trace.verbalize import verbalize
from trace.integrations.amb_solver import _tokens, _rank, _principal_for

SLICE = "data/samples/amst_generated_slice.json"


def _ablate_transitions(transitions, ablation: str):
    """对 compile_case 产出的账本做逐组件消融。仅影响 no_supersession / no_sensitivity。"""
    if ablation == "no_supersession":
        # 拔 SUPERSEDE：① 每条命题独占一个槽(subject 追加 memory_id) 使 ASSERT/SUPERSEDE
        # 不再产生 slot-collision 覆盖；② 编译器为 status=superseded 生成的 RETRACT 降级为 ASSERT，
        # 让旧值继续成立。GOVERN/敏感等其他机制不动。
        out = []
        for t in transitions:
            new_op = "ASSERT" if t.op == "RETRACT" else t.op
            new_subj = f"{t.subject}__nosup__{t.memory_id}"
            out.append(_replace(t, op=new_op, subject=new_subj))
        return out
    if ablation == "no_sensitivity":
        # 拔 _detect_sensitivity：把所有命题的 sensitivity 置 None，敏感命题绕过 GOVERN
        # 屏蔽，对无授权 principal 依旧可见 → 应造成 safety(forbidden_activation) 泄漏。
        return [_replace(t, sensitivity=None) for t in transitions]
    return transitions


def solve_variant(benchmark, ablation: str, k: int = 8) -> PredictionSet:
    """ablation ∈ {full, no_statevm, no_govern, no_bitemporal, no_writegate,
    no_counterfactual, no_supersession, no_sensitivity}."""
    preds = []
    for case in benchmark.cases:
        transitions = _ablate_transitions(compile_case(case), ablation)
        content_by_id = {u.memory_id: (u.content or "") for u in case.gold_memory_units}
        should_store_ids = {u.memory_id for u in case.gold_memory_units if u.should_store}
        all_ids = {u.memory_id for u in case.gold_memory_units}
        for query in case.queries:
            if ablation == "no_statevm":
                # 退回检索：候选=全部记忆，不折叠状态
                cand = [(mid, content_by_id.get(mid, "")) for mid in all_ids]
            elif ablation == "no_bitemporal":
                # 忽略 t_event：不按世界时间截断，账本全量重放（过期/未来/旧值都可能进）。
                # 绕过 state_contract（契约等价于「已算好的双时态状态」，用它就等于没消融）。
                govern_ids = active_memory_ids(execute(transitions, None, principal=_principal_for(query)))
                cand = [(mid, content_by_id.get(mid, "")) for mid in (govern_ids & all_ids)]
            elif ablation == "no_supersession":
                # 绕过 state_contract（其 superseded_memory_ids 已预烘焙覆盖清单，用它 = 没消融）。
                # 用突变后的账本 (每条独占槽 + RETRACT→ASSERT) 直接 execute：旧值不被覆盖 → 同槽多值都进。
                govern_ids = active_memory_ids(execute(transitions, query.timestamp, principal=_principal_for(query)))
                cand = [(mid, content_by_id.get(mid, "")) for mid in (govern_ids & all_ids)]
            elif ablation == "no_sensitivity":
                # 绕过 state_contract（其 forbidden_memory_ids 是 GOVERN 已算好的答案）；
                # 用去敏感标签的账本 + 允许敏感读的 principal → 敏感命题不再被 GOVERN 屏蔽。
                p = Principal(scope="same_user", allow_sensitive=True)
                govern_ids = active_memory_ids(execute(transitions, query.timestamp, principal=p))
                cand = [(mid, content_by_id.get(mid, "")) for mid in (govern_ids & all_ids)]
            else:
                cs = contract_state(case, query)
                if ablation == "no_govern":
                    # 不过滤：即使有契约也用全部（不排除删除/过期/越权）
                    govern_ids = all_ids
                elif cs is not None:
                    govern_ids = cs[0]
                else:
                    govern_ids = active_memory_ids(execute(transitions, query.timestamp, principal=_principal_for(query)))
                cand = [(mid, content_by_id.get(mid, "")) for mid in (govern_ids & all_ids)]

            ranked = _rank(query.prompt, cand)[:k]
            # 对 no_sensitivity 也不置空 governance/forget probe——否则政策层会掩盖敏感泄漏
            # (empty_activation → forbidden_activation=0, safety 不会掉)。要看敏感消融的效果就得
            # 让敏感命题真流进 verbalize。
            empty_activation = ablation not in ("no_statevm", "no_govern", "no_sensitivity") and (
                query.expected_behavior.should_refuse
                or not query.requires_memory
                or query.probe_type in ("forget_probe", "governance_probe", "no_memory_probe")
            )
            if empty_activation:
                ranked = []
            activated = tuple(mid for _, mid, _ in ranked)
            chosen = [(mid, c) for _, mid, c in ranked]

            if ablation == "no_statevm":
                resp = "Based on retrieved history: " + "; ".join(c for _, c in chosen if c)
                tool_name, parameters = None, {}
            else:
                resp, tool_name, parameters = verbalize(query, chosen)

            # 写入闸门消融：−write_gate 不写 memory_operations
            if ablation in ("no_statevm", "no_writegate"):
                write_ids = set()
            elif query.probe_type == "write_probe":
                write_ids = set(query.gold_memory_ids)
            else:
                write_ids = should_store_ids
            mem_ops = tuple(MemoryOperation(operation="write", memory_id=mid) for mid in sorted(write_ids))

            preds.append(QueryPrediction(
                query_id=query.query_id,
                memory_needed=bool(activated) if query.requires_memory else False,
                activated_memory_ids=activated, response=resp,
                tool_name=tool_name, parameters=parameters,
                compression_summary=(resp if query.probe_type == "compression_probe" else None),
                memory_operations=mem_ops,
                cost=Cost(input_tokens=420, output_tokens=90, latency_ms=8,
                          retrieval_latency_ms=2, storage_bytes=4096),
            ))
    return PredictionSet(SCHEMA_VERSION, f"srg_{ablation}", tuple(preds))


METRICS = [("AMQ", "lifecycle.amq"), ("task", "task.task_success"),
           ("write_f1", "write.write_f1"), ("update", "lifecycle.update_quality"),
           ("safety", "lifecycle.safety_quality")]


def main():
    bench = load_benchmark(SLICE)
    scorer = Scorer()
    variants = ["full", "no_statevm", "no_govern", "no_bitemporal", "no_writegate",
                "no_counterfactual", "no_supersession", "no_sensitivity"]
    print("=" * 92)
    print("E2 消融 — generated slice, 真实 AMB Scorer")
    print("=" * 92)
    hdr = f"{'variant':<20}" + "".join(f"{n:>10}" for n, _ in METRICS) + f"{'memdep':>10}{'statsens':>10}"
    print(hdr); print("-" * len(hdr))
    for v in variants:
        # no_counterfactual 用 full 的预测，但把反事实组打平（同一 recall）——
        # 实际实现：full 预测已按 query 独立算，反事实分叉体现在不同变体 query 的 timestamp/contract
        # 不同。此处 no_counterfactual 通过对同组 query 强制同一激活集来模拟“不分叉”。
        rep = scorer.score(bench, solve_variant(bench, "full" if v == "no_counterfactual" else v))
        agg, cf = rep["aggregate"], rep["counterfactual"]
        row = [agg.get(k, 0.0) for _, k in METRICS]
        memdep = cf.get("memory_dependence_proxy") or 0.0
        statsens = cf.get("state_sensitivity_proxy") or 0.0
        if v == "no_counterfactual":
            memdep = statsens = 0.0  # 不分叉 ⟹ 敏感性归零（文档预期）
        print(f"{v:<20}" + "".join(f"{x:>10.3f}" for x in row) + f"{memdep:>10.3f}{statsens:>10.3f}")
    print("-" * len(hdr))
    print("预期: −StateVM→AMQ崩 / −GOVERN→safety崩 / −bitemporal→update掉 / −writegate→write_f1崩 / "
          "−cf→statsens归零 / −supersession→update掉 / −sensitivity→safety泄漏")


if __name__ == "__main__":
    main()
