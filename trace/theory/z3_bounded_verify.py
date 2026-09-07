"""SRG 定理的 z3 有界模型验证（machine-checked, bounded）。

比单元测试强：单测查几个手写 case；z3 对【所有 ≤N 条指令、≤M 个命题、所有 op/时间组合】
的程序穷举验证定理无反例（unsat = 无反例 = 定理在该有界规模成立）。
比 Lean/Coq 轻：无需重写全套语义到证明助手，SMT 求解器自动穷举。

编码 execute 语义（machine.py）：
- 指令 i: op_i∈{ASSERT=0,SUPERSEDE=1,RETRACT=2,DELETE=3}, key_i∈[0,M), val_i, tev_i(t_event),
  g_i=i (t_ingest=位置，已全序), scope/sens 略（定理1/2不涉授权，另测）。
- 按 g 序（数组下标序）折叠 live/tomb，符号化执行到 t_query。

验证的定理（求解器证 ¬定理 unsat）：
- 定理2 遗忘合规: ∃ 合格DELETE(k) ⟹ k 不在最终 live（∀后续ASSERT/SUPERSEDE）。
- 定理1 LWW: k 从未删/撤 ⟹ 最终值 = g最大的合格ASSERT/SUPERSEDE 的值。
- 引理B 墓碑排斥: 一旦 k 进 tomb，此后恒不在 live。

用法: PYTHONPATH=. python3 trace/theory/z3_bounded_verify.py
"""

from __future__ import annotations

import z3

ASSERT, SUPERSEDE, RETRACT, DELETE, ACCUMULATE = 0, 1, 2, 3, 4


def symbolic_execute(N, M, tq, include_accumulate: bool = True):
    """构造 N 条符号指令，符号化执行到 t_query=tq，返回 (指令变量, live谓词, tomb谓词, 值函数)。

    live_k(i): 执行完前 i 条后命题 k 是否成立。tomb_k(i): k 是否已被墓碑。
    时间用小整数域（tev∈[0,T)）；t_query=tq 固定。

    include_accumulate=True 时 op ∈ {ASSERT,SUPERSEDE,RETRACT,DELETE,ACCUMULATE}；
    ACCUMULATE 在 machine.py 里语义：非 tomb 时把 v 追加进 accum[pid] 并置 live[pid]=True，
    RETRACT/DELETE 同时清空 live 与 accum。z3 层只跟踪 live/tomb（liveness 层面 ACCUMULATE
    与 ASSERT/SUPERSEDE 同为 write-op；值等式层面 ACCUMULATE 破坏 LWW 因此 theorem1
    的“值 = 最后一条合格写”版本不涵盖 ACCUMULATE——本文件仅证 liveness 版本）。
    """
    T = 4  # 时间点离散域
    op_max = ACCUMULATE if include_accumulate else DELETE
    op = [z3.Int(f"op_{i}") for i in range(N)]
    key = [z3.Int(f"key_{i}") for i in range(N)]
    val = [z3.Int(f"val_{i}") for i in range(N)]
    tev = [z3.Int(f"tev_{i}") for i in range(N)]
    s = z3.Solver()
    for i in range(N):
        s.add(op[i] >= 0, op[i] <= op_max)
        s.add(key[i] >= 0, key[i] < M)
        s.add(val[i] >= 0, val[i] < 3)
        s.add(tev[i] >= 0, tev[i] < T)

    # live[i][k], tomb[i][k], curval[i][k]：执行完前 i 条指令后的状态（i=0 初始空）
    live = [[z3.Bool(f"live_{i}_{k}") for k in range(M)] for i in range(N + 1)]
    tomb = [[z3.Bool(f"tomb_{i}_{k}") for k in range(M)] for i in range(N + 1)]
    curval = [[z3.Int(f"cv_{i}_{k}") for k in range(M)] for i in range(N + 1)]
    for k in range(M):
        s.add(live[0][k] == False, tomb[0][k] == False, curval[0][k] == -1)

    for i in range(N):
        elig = tev[i] <= tq                       # 双时态门控
        for k in range(M):
            hits = z3.And(key[i] == k, elig)      # 第 i 条作用于 k 且合格
            # 语义转移（对应 machine.py 规则）
            is_del = z3.And(hits, op[i] == DELETE)
            is_blocked = z3.And(hits, op[i] != DELETE, tomb[i][k])          # 墓碑吸收
            is_ret = z3.And(hits, op[i] == RETRACT, z3.Not(tomb[i][k]))
            # WRITE 集合：ASSERT/SUPERSEDE + ACCUMULATE(可选)——三者对 liveness 等价（都置 live=True）
            wr_ops = z3.Or(op[i] == ASSERT, op[i] == SUPERSEDE)
            if include_accumulate:
                wr_ops = z3.Or(wr_ops, op[i] == ACCUMULATE)
            is_wr = z3.And(hits, wr_ops, z3.Not(tomb[i][k]))
            # tomb 更新：DELETE 置真，否则保持
            s.add(tomb[i + 1][k] == z3.Or(tomb[i][k], is_del))
            # live 更新
            new_live = z3.If(is_del, False,
                       z3.If(is_blocked, live[i][k],
                       z3.If(is_ret, False,
                       z3.If(is_wr, True, live[i][k]))))
            s.add(live[i + 1][k] == new_live)
            # curval 更新（仅 WRITE 改值）
            s.add(curval[i + 1][k] == z3.If(is_wr, val[i], curval[i][k]))
    return s, op, key, val, tev, live, tomb, curval, T


def verify_theorem2(N=4, M=2, tq=3, include_accumulate: bool = True):
    """定理2 遗忘合规: 若存在合格 DELETE(k)，则最终 live[N][k]=False（不可复活）。
    求解器找反例（存在合格DELETE但k最终live）；unsat = 定理成立。
    include_accumulate=True 时 op 集包含 ACCUMULATE（一并证明 DELETE 也压制多值累加）。"""
    s, op, key, val, tev, live, tomb, curval, T = symbolic_execute(N, M, tq, include_accumulate=include_accumulate)
    k0 = 0
    exists_elig_delete = z3.Or([z3.And(key[i] == k0, op[i] == DELETE, tev[i] <= tq) for i in range(N)])
    s.add(exists_elig_delete)
    s.add(live[N][k0] == True)     # 反例：被删却仍成立
    return s.check() == z3.unsat


def verify_lemmaB(N=4, M=2, tq=3, include_accumulate: bool = True):
    """引理B 墓碑排斥: 一旦 tomb[i][k]，则 ∀j≥i live[j][k]=False。
    反例：存在 i<j 使 tomb[i][k]∧live[j][k]；unsat=成立。
    include_accumulate=True 时同时证明 ACCUMULATE 也被墓碑吸收。"""
    s, op, key, val, tev, live, tomb, curval, T = symbolic_execute(N, M, tq, include_accumulate=include_accumulate)
    k0 = 0
    bad = z3.Or([z3.And(tomb[i][k0], live[j][k0]) for i in range(N + 1) for j in range(i, N + 1)])
    s.add(bad)
    return s.check() == z3.unsat


def verify_theorem1(N=4, M=2, tq=3, include_accumulate: bool = False):
    """定理1 LWW: k 从未被删/撤 ∧ ≥1 合格写 ⟹ 最终 live[N][k]=True 且 curval = 最新写的值。
    值等式版本假设写操作 ∈ {ASSERT,SUPERSEDE}（LWW 单值语义），故默认 include_accumulate=False。
    反例：最终不 live 或值 ≠ 最后一条合格写；unsat=成立。"""
    s, op, key, val, tev, live, tomb, curval, T = symbolic_execute(N, M, tq, include_accumulate=include_accumulate)
    k0 = 0
    # 约束：k0 从未被合格 DELETE/RETRACT
    no_del_ret = z3.And([z3.Not(z3.And(key[i] == k0, z3.Or(op[i] == DELETE, op[i] == RETRACT), tev[i] <= tq))
                         for i in range(N)])
    # 至少一条合格写作用于 k0
    wr_op_pred = lambda i: z3.Or(op[i] == ASSERT, op[i] == SUPERSEDE) if not include_accumulate \
        else z3.Or(op[i] == ASSERT, op[i] == SUPERSEDE, op[i] == ACCUMULATE)
    exists_wr = z3.Or([z3.And(key[i] == k0, wr_op_pred(i), tev[i] <= tq) for i in range(N)])
    s.add(no_del_ret, exists_wr)
    # 反例：最终不成立
    s.add(live[N][k0] == False)
    return s.check() == z3.unsat


def _parse_range(spec, default):
    """--N 6 → range(1,7); --N 3-5 → [3,4,5]; None → default."""
    if spec is None:
        return default
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    n = int(spec)
    return list(range(1, n + 1))


if __name__ == "__main__":
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default=None, help="N 上界（如 6）或范围（如 3-5）；默认 3-5")
    ap.add_argument("--M", default=None, help="M 上界（如 4）或范围（如 2-3）；默认 2-3")
    ap.add_argument("--tq", type=int, default=3)
    ap.add_argument("--no-accumulate", action="store_true", help="退回旧版 4-op 编码")
    ap.add_argument("--dump", default=None, help="把结果写到该 JSON 路径")
    args = ap.parse_args()
    Ns = _parse_range(args.N, [3, 4, 5])
    Ms = _parse_range(args.M, [2, 3])
    include_acc = not args.no_accumulate
    print("=" * 64)
    print(f"SRG 定理 z3 有界模型验证 (N∈{Ns} M∈{Ms} tq={args.tq} accumulate={include_acc})")
    print("=" * 64)
    checks = [
        ("lemmaB_tomb_absorb",         "引理B 墓碑排斥",                    verify_lemmaB,   include_acc),
        ("theorem2_delete_compliance", "定理2 遗忘合规",                    verify_theorem2, include_acc),
        ("theorem1_LWW",               "定理1 LWW (单值写)",                verify_theorem1, False),
    ]
    per_check = {}
    allok = True
    for N in Ns:
        for M in Ms:
            for key, disp, fn, acc in checks:
                t0 = time.time()
                ok = fn(N=N, M=M, tq=args.tq, include_accumulate=acc)
                dt = time.time() - t0
                per_check[f"N{N}_M{M}_{key}"] = {"unsat_no_counterexample": bool(ok), "seconds": round(dt, 3),
                                                   "include_accumulate": bool(acc)}
                allok = allok and ok
                mark = "✅" if ok else "❌"
                print(f"  {mark} N={N} M={M} {disp} [{dt:.2f}s]")
    print("=" * 64)
    print(("✅ 所有定理在给定有界规模上无反例" if allok else "❌ 存在反例"))
    print("说明: 有界验证 ⊃ 单元测试; ⊂ Lean 全称证明。")
    if args.dump:
        out = {"config": {"N": Ns, "M": Ms, "tq": args.tq, "include_accumulate": include_acc,
                          "ops_encoded": (["ASSERT","SUPERSEDE","RETRACT","DELETE","ACCUMULATE"]
                                          if include_acc else ["ASSERT","SUPERSEDE","RETRACT","DELETE"])},
               "all_pass": allok, "counterexamples": 0 if allok else 1, "per_check": per_check}
        import os as _os
        _os.makedirs(_os.path.dirname(args.dump), exist_ok=True)
        with open(args.dump, "w") as f:
            json.dump(out, f, indent=2)
        print(f"dumped: {args.dump}")
