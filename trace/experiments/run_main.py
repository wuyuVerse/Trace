"""Task 7 / E1 —— 主实验 + 统计严谨性（bootstrap CI + 配对显著性检验 + 分层）。

对 SRG vs 基线在 AMB 上出带**置信区间**和**配对检验 p 值**的主表（reviewer 要）。
不用 numpy/scipy（pod 未必有），纯 stdlib 实现 bootstrap 与置换检验，固定种子可复现。

指标：per-query AMQ（lifecycle.amq）。
  - bootstrap CI：对 per-query AMQ 重采样 1000 次求 2.5%/97.5% 分位。
  - 配对置换检验：SRG vs 最强基线，每 query 配对，10000 次符号翻转求 p。
分层：by_probe_type。

Run:  PYTHONPATH=. python3 trace/experiments/run_main.py
（全量 main-v1-strict：换 --benchmark 指向全量 manifest，经集群提交）
"""

from __future__ import annotations

import argparse
import random

from amb.benchmark.schemas.io import load_benchmark
from amb.benchmark.evaluation.baselines_pkg.baselines import make_baseline
from amb.benchmark.evaluation.core.scoring import Scorer
from trace.integrations.amb_solver import solve

RNG = random.Random(20260705)


def per_query_amq(report) -> dict:
    """query_id -> lifecycle.amq。"""
    out = {}
    for q in report["queries"]:
        out[q["query_id"]] = q["scores"]["lifecycle"]["amq"]
    return out


def bootstrap_ci(values, n=1000, alpha=0.05):
    if not values:
        return (0.0, 0.0, 0.0)
    m = sum(values) / len(values)
    means = []
    k = len(values)
    for _ in range(n):
        s = sum(values[RNG.randrange(k)] for _ in range(k)) / k
        means.append(s)
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return (m, lo, hi)


def paired_permutation_p(a: dict, b: dict, n=10000) -> float:
    """配对置换检验：H0 = SRG 与基线每 query 无系统差异。返回双尾 p。"""
    keys = [k for k in a if k in b]
    diffs = [a[k] - b[k] for k in keys]
    if not diffs:
        return 1.0
    obs = abs(sum(diffs) / len(diffs))
    cnt = 0
    for _ in range(n):
        s = sum(d if RNG.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(s) >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/samples/amst_generated_slice.json")
    ap.add_argument("--baselines", default="a_mem_agentic_memory_proxy,state_guard_memory,hybrid_memory,graph_memory,oracle_memory")
    args = ap.parse_args()

    bench = load_benchmark(args.benchmark)
    scorer = Scorer()

    systems = {}
    for kind in args.baselines.split(","):
        systems[kind] = per_query_amq(scorer.score(bench, make_baseline(bench, kind)))
    systems["SRG(full)"] = per_query_amq(scorer.score(bench, solve(bench, "srg")))

    print("=" * 78)
    print(f"E1 主实验 + 统计 — {args.benchmark}")
    print("=" * 78)
    print(f"{'system':<32}{'AMQ':>8}{'95% CI':>20}{'vs SRG p':>12}")
    print("-" * 78)
    srg = systems["SRG(full)"]
    # 最强非 oracle 基线（按均值）
    non_oracle = {k: v for k, v in systems.items() if k != "SRG(full)" and "oracle" not in k}
    best_base = max(non_oracle, key=lambda k: sum(non_oracle[k].values()) / len(non_oracle[k]))
    for name, pq in systems.items():
        vals = list(pq.values())
        m, lo, hi = bootstrap_ci(vals)
        if name == "SRG(full)":
            p_str = "—"
        else:
            p = paired_permutation_p(srg, pq)
            p_str = f"{p:.4f}"
        star = "  ← 最强基线" if name == best_base else ("  ← 本文" if name == "SRG(full)" else "")
        print(f"{name:<32}{m:>8.3f}   [{lo:.3f}, {hi:.3f}]{p_str:>12}{star}")
    print("-" * 78)
    print(f"配对置换检验：SRG(full) vs {best_base}，1000×bootstrap CI + 10000×permutation，种子固定。")


if __name__ == "__main__":
    main()
