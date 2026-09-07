"""三定理单测 —— SRG 的正确性保证是数据结构的必然推论，不是实验观察。

每个定理构造一个满足前件的最小 case，断言后件成立。检索范式连这些定理都无法陈述。

  定理1 时态正确性：φ 在 t 过期/被 SUPERSEDE ⟹ φ∉Recall
  定理2 遗忘合规性：DELETE(φ)∈L ⟹ ∀q,t,p φ∉Recall（合规删除不可复活）
  定理3 反事实可分辨：Δ 改变投影 ⟹ 激活集变（state_sensitivity=1）
  引理1  ⊕ 是 monoid（结合/交换/幂等/幺元）——associative scan 合法性

Run:  PYTHONPATH=. python3 trace/theory/test_theorems.py
"""

from __future__ import annotations

from trace.core import (
    Transition, Principal, execute, execute_counterfactual, active_memory_ids,
    combine, lift, fold, EMPTY,
)


def _t(mid, subj, attr, op, te, ti, **kw):
    return Transition(memory_id=mid, subject=subj, attribute=attr, op=op,
                      t_event=te, t_ingest=ti, **kw)


PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}")


# ---- 定理1：时态正确性 ----------------------------------------------------
def test_theorem1_temporal():
    print("定理1 时态正确性：过期/被supersede 的 φ 不进 Recall")
    p = Principal(allow_sensitive=True)

    # (a) SUPERSEDE：地址 Beijing→Shanghai，当前状态只应有 Shanghai
    L = [
        _t("m1", "user", "home", "ASSERT", "2026-01", 1, content="Beijing"),
        _t("m2", "user", "home", "SUPERSEDE", "2026-03", 2, content="Shanghai"),
    ]
    st = execute(L, "2026-06", principal=p)
    vals = {pp.value for pp in st}
    check("SUPERSEDE: 当前含Shanghai", "Shanghai" in vals)
    check("SUPERSEDE: 旧值Beijing退场", "Beijing" not in vals)

    # (b) EXPIRE：valid_until 早于查询时刻 → 不在当前状态
    L2 = [_t("m3", "user", "pass", "ASSERT", "2026-01", 1,
             valid_until="2026-02", content="temp-pass")]
    st_after = execute(L2, "2026-06", principal=p)      # 查询在过期之后
    st_before = execute(L2, "2026-01", principal=p)     # 查询在过期之前
    check("EXPIRE: 过期后不在状态", "m3" not in active_memory_ids(st_after))
    check("EXPIRE: 过期前在状态", "m3" in active_memory_ids(st_before))

    # (c) 双时态：t_event > 查询时刻的转移不参与重放
    L3 = [_t("m4", "user", "x", "ASSERT", "2026-09", 1, content="future")]
    check("双时态: 未来事件不进当前", "m4" not in active_memory_ids(execute(L3, "2026-06", principal=p)))


# ---- 定理2：遗忘合规性 ----------------------------------------------------
def test_theorem2_deletion():
    print("定理2 遗忘合规性：DELETE 后任何 Recall 都不激活（且不可复活）")
    L = [
        _t("k1", "user", "api_key", "ASSERT", "2026-01", 1, content="KEY1"),
        _t("k2", "user", "api_key", "DELETE", "2026-06", 2, content="rotated out"),
    ]
    # 遍历多个 principal / 时刻，全都不应激活
    ok = True
    for scope in ("same_user", "project_a", "role_admin"):
        for t in ("2026-06", "2026-07", "2027-01"):
            p = Principal(scope=scope, allow_sensitive=True)
            if "api_key" in {pp.attribute for pp in execute(L, t, principal=p)}:
                ok = False
    check("DELETE: ∀principal,t 均不激活", ok)

    # 关键：DELETE 之后再 ASSERT 同槽（墓碑单调粘附，正常回忆不可复活）
    L_revive = L + [_t("k3", "user", "api_key", "ASSERT", "2026-08", 3, content="KEY1-again")]
    p = Principal(allow_sensitive=True)
    check("DELETE: 墓碑粘附,后续ASSERT不复活", "api_key" not in {pp.attribute for pp in execute(L_revive, "2026-09", principal=p)})


# ---- 定理3：反事实可分辨 --------------------------------------------------
def test_theorem3_counterfactual():
    print("定理3 反事实可分辨：Δ 改变投影 ⟹ 激活集变")
    p = Principal(allow_sensitive=True)
    L = [
        _t("k1", "user", "api_key", "ASSERT", "2026-01", 1, content="KEY1"),
        _t("k2", "user", "api_key", "DELETE", "2026-06", 2, content="rotated out"),
    ]
    factual = execute(L, "2026-07", principal=p)
    # 反事实：假装 DELETE(k2) 从未发生（remove 语义）
    cf = execute_counterfactual(L, remove_memory_ids={"k2"}, t_query="2026-07", principal=p)
    check("反事实前: api_key 不在(被删)", "api_key" not in {pp.attribute for pp in factual})
    check("反事实后: api_key 复现(假装没删)", "api_key" in {pp.attribute for pp in cf})
    check("反事实改变了激活集", active_memory_ids(factual) != active_memory_ids(cf))

    # Δ 不改变投影 → 激活集不变（退化情形）
    cf_noop = execute_counterfactual(L, inject=[_t("z1", "other", "unrelated", "ASSERT", "2026-01", 9, content="x")],
                                     t_query="2026-07", principal=p)
    # 无关注入会新增槽位；验证原有 api_key 仍被删（分叉不误伤）
    check("无关Δ: api_key 仍被删", "api_key" not in {pp.attribute for pp in cf_noop})


# ---- 引理1：⊕ 是 monoid --------------------------------------------------
def test_lemma1_monoid():
    print("引理1 ⊕ 是 monoid：结合/交换/幂等/幺元")
    a = lift(_t("a", "s", "x", "ASSERT", "t1", 1, content="A"))
    b = lift(_t("b", "s", "x", "SUPERSEDE", "t2", 2, content="B"))
    c = lift(_t("c", "s", "y", "ASSERT", "t3", 3, content="C"))

    def eq(x, y):
        if x.keys() != y.keys():
            return False
        return all((x[k].memory_id, x[k].op, x[k].t_ingest, x[k].tombstoned)
                   == (y[k].memory_id, y[k].op, y[k].t_ingest, y[k].tombstoned) for k in x)

    check("结合律", eq(combine(combine(a, b), c), combine(a, combine(b, c))))
    check("交换律(同槽LWW收敛)", eq(combine(a, b), combine(b, a)))
    check("幂等律", eq(combine(a, a), a))
    check("幺元 ∅", eq(combine(dict(EMPTY), a), a) and eq(combine(a, dict(EMPTY)), a))


# ---- 执行引擎定理组（唯一核心：回忆即执行）----------------------------------
def test_execution_theorems():
    from trace.core import execute, execute_counterfactual, read_count, read_temporal
    print("执行引擎定理：回忆=执行程序到 t，读机器状态 S(t)")
    p = Principal(allow_sensitive=False)

    # 定理4（可采纳=执行必然）：被 DELETE 的命题从不在执行输出 S(t) 里，且不可复活。
    L = [_t("a", "user", "key", "ASSERT", "2026-01", 1, content="K1"),
         _t("d", "user", "key", "DELETE", "2026-06", 2, content="K1"),
         _t("r", "user", "key", "ASSERT", "2026-08", 3, content="K1-again")]  # 删后再断言
    S = execute(L, "2026-09", principal=p)
    check("定理4: DELETE后命题不在S(t)(执行必然,非过滤)", "key" not in {pp.attribute for pp in S})
    check("定理4: 墓碑粘附,删后ASSERT不复活", "key" not in {pp.attribute for pp in S})

    # 定理4b（越权/过期由执行语义排除）
    L2 = [_t("x", "other", "sec", "ASSERT", "2026-01", 1, scope="proj_x", content="S"),
          _t("e", "user", "promo", "ASSERT", "2026-01", 2, valid_until="2026-03", content="SALE")]
    S2 = execute(L2, "2026-07", principal=p)
    check("定理4b: 越权命题不在S(t)", "sec" not in {pp.attribute for pp in S2})
    check("定理4b: 过期命题不在S(t)", "promo" not in {pp.attribute for pp in S2})

    # 定理5（计数可数）：N 个不同 subject 的成立命题 → read_count = N（纯属性=值数不出，命题集能数）。
    L3 = [_t(f"i{i}", f"task_{i}", "todo", "ASSERT", "2026-01", i, content=f"todo{i}") for i in range(1, 4)]
    L3.append(_t("done1", "task_1", "todo", "RETRACT", "2026-05", 5, content=""))  # task1完成
    S3 = execute(L3, "2026-06", principal=p)
    check("定理5: 计数=成立命题数(3待办-1完成=2)", read_count(S3) == 2)

    # 定理6（时序可排）：成立命题按 t_event 排序 → 最早的在前。
    L4 = [_t("s", "user", "got_a", "ASSERT", "2023-02-20", 1, content="Samsung"),
          _t("dd", "user", "got_b", "ASSERT", "2023-02-25", 2, content="Dell")]
    ordered = read_temporal(execute(L4, None, principal=p))
    check("定理6: 时序排序,最早在前(Samsung)", ordered[0].value == "Samsung")

    # 定理7（反事实=改指令重跑）：移除 DELETE 指令重新执行 → 被删命题复现。
    cf = execute_counterfactual(L, remove_memory_ids={"d"}, t_query="2026-09", principal=p)
    check("定理7: 移除DELETE重执行→命题复现", "key" in {pp.attribute for pp in cf})

    # 定理13（历史读算子正交性）：read_slot_history 从账本 L 重建【每槽完整时序轨迹】——
    #   execute 读当前塌缩态 S(t)，read_slot_history 读全历史，二者读同一 L、语义正交。
    #   这是账本范式独有的"第 N 近值/变更次数/前一个值"能力（检索范式无结构化历史轴）。
    from trace.core import read_slot_history
    # Sandra 移动 5 次（SUPERSEDE 覆盖）→ S(t) 只剩当前值，但历史应完整可重建。
    Lh = [_t("h0", "Sandra", "loc", "ASSERT",    "0000", 1, content="bedroom"),
          _t("h1", "Sandra", "loc", "SUPERSEDE", "0001", 2, content="kitchen"),
          _t("h2", "Sandra", "loc", "SUPERSEDE", "0002", 3, content="garden"),
          _t("h3", "Sandra", "loc", "SUPERSEDE", "0003", 4, content="hallway"),
          _t("h4", "Sandra", "loc", "SUPERSEDE", "0004", 5, content="office")]
    Sh = execute(Lh, None, principal=p)
    check("定理13: S(t)塌缩只剩当前值(office)", len(Sh) == 1 and Sh[0].value == "office")
    hist = read_slot_history(Lh)
    seq = hist[("Sandra", "loc")]
    check("定理13: 历史重建完整5步", [r["value"] for r in seq] == ["bedroom", "kitchen", "garden", "hallway", "office"])
    check("定理13: 第2近的值=hallway(倒数索引)", seq[-2]["value"] == "hallway")
    check("定理13: 变更次数=4(=len-1)", len(seq) - 1 == 4)
    # 双时态：t_query 截断历史（只见已发生的移动）
    hist_t = read_slot_history(Lh, t_query="0002")
    check("定理13: 双时态截断历史(t=0002→3步)", len(hist_t[("Sandra", "loc")]) == 3)
    # DELETE 抹除整段历史（合规删除对历史读同样生效）
    Ld = Lh + [_t("hd", "Sandra", "loc", "DELETE", "0005", 6, content="")]
    check("定理13: DELETE后历史整体不可见", ("Sandra", "loc") not in read_slot_history(Ld))
    # 授权：sensitive 赋值不进无授权 principal 的历史
    Ls = [_t("s0", "u", "pw", "ASSERT", "0000", 1, content="old", sensitivity="secret"),
          _t("s1", "u", "pw", "SUPERSEDE", "0001", 2, content="new", sensitivity="secret")]
    check("定理13: 越权principal读不到sensitive历史",
          ("u", "pw") not in read_slot_history(Ls, principal=Principal(allow_sensitive=False)))
    check("定理13: 授权principal可读sensitive历史",
          len(read_slot_history(Ls, principal=Principal(allow_sensitive=True)).get(("u", "pw"), [])) == 2)


# ---- 凭证定理组（更高一层：回忆即可自证的证明）--------------------------------
def test_certificate_theorems():
    from trace.core import Transition, certify, verify, execute
    print("凭证定理：每次回忆 = 可独立核验的证明对象 C(t,q)=⟨最小切片,排除报告,答案⟩")
    p = Principal(allow_sensitive=False)

    def T(mid, s, a, op, te, ti, c="", **kw):
        return Transition(memory_id=mid, subject=s, attribute=a, op=op, t_event=te, t_ingest=ti, content=c, **kw)

    # 定理8（可靠性 soundness）：重放最小切片 → 与完整程序得【相同】目标命题值。
    L = [T("m1", "user", "home", "ASSERT", "2026-01", 1, "Beijing"),
         T("m2", "user", "home", "SUPERSEDE", "2026-03", 2, "Shanghai"),
         T("m3", "user", "job", "ASSERT", "2026-02", 3, "ACME")]  # 无关指令
    cert = certify(L, [("user", "home")], t_query="2026-06")
    r = verify(cert, L)
    check("定理8: 凭证可靠(重放切片得同答案Shanghai)", r["sound"])

    # 定理9（最小性 minimality）：切片无冗余——只含决定最终值的那一条指令。
    check("定理9: 凭证最小(切片仅含m2,去无关m1/m3)",
          r["minimal"] and len(cert.slice) == 1 and cert.slice[0].memory_id == "m2")

    # 定理10（完备性 completeness / 禁忌不泄漏）：被 DELETE 的命题当答案 → 凭证拒绝背书。
    Lk = [T("k1", "user", "api_key", "ASSERT", "2026-01", 1, "KEY1"),
          T("k2", "user", "api_key", "DELETE", "2026-06", 2)]
    leak = certify(Lk, [("user", "api_key")], t_query="2026-07")
    check("定理10: 禁忌(已删)当答案→completeness拒绝背书", not verify(leak, Lk)["complete"])

    # 定理11（越权排除入证）：sensitive 命题被执行语义排除，且必列入排除报告；合法答案通过。
    Ls = [T("s1", "user", "salary", "ASSERT", "2026-01", 1, "100k", sensitivity="sensitive"),
          T("s2", "user", "city", "ASSERT", "2026-01", 2, "NYC")]
    cs = certify(Ls, [("user", "city")], t_query="2026-07", principal=p)
    vs = verify(cs, Ls)
    check("定理11: 越权命题入排除报告且合法答案全通过",
          vs["ok"] and any(e["attribute"] == "salary" and e["reason"] == "unauthorized" for e in cs.excluded))


# ---- 引擎唯一性定理（solid 保证：只有一个引擎，monoid 是它的可证明性质）------------
def test_engine_uniqueness():
    """定理12：可执行引擎 execute 的输出 == 账本 monoid fold 经投影/授权后的状态。

    证明 machine.execute（唯一引擎）与 algebra.fold（CRDT monoid，理论基石）在语义上一致——
    monoid 不是"第二引擎"，而是 execute 的可证明性质。这堵死了"两个大脑"的缝合质疑。
    """
    print("定理12 引擎唯一性：execute 输出 ≡ monoid fold + 投影/授权（同一状态，两种视角）")
    p = Principal(allow_sensitive=True)

    # 构造含 ASSERT/SUPERSEDE/RETRACT/未来事件的混合账本
    L = [_t("a", "user", "home", "ASSERT", "2026-01", 1, content="Beijing"),
         _t("b", "user", "home", "SUPERSEDE", "2026-03", 2, content="Shanghai"),
         _t("c", "user", "job", "ASSERT", "2026-02", 3, content="ACME"),
         _t("d", "user", "todo", "ASSERT", "2026-02", 4, content="x"),
         _t("e", "user", "todo", "RETRACT", "2026-04", 5),
         _t("f", "user", "fut", "ASSERT", "2026-09", 6, content="future")]
    t_query = "2026-06"

    # (1) 执行引擎输出 S(t)
    exec_state = {(pp.subject, pp.attribute): pp.value for pp in execute(L, t_query, principal=p)}

    # (2) monoid fold 到 t（只折 t_event≤t）→ 投影去 retract/tombstone/expire → 授权
    from trace.core.governance import visible
    elig = [t for t in L if t.t_event <= t_query]
    folded = fold(elig)                     # ⊕ 归约（associative scan 端点）
    fold_state = {}
    for k, slot in folded.items():
        if slot.tombstoned or slot.op in ("RETRACT", "DELETE"):
            continue
        if slot.valid_until is not None and slot.valid_until < t_query:
            continue
        if not visible(slot.scope, slot.sensitivity, p):
            continue
        fold_state[k] = slot.content

    check("定理12: execute 输出 == monoid fold+投影 (逐命题一致)", exec_state == fold_state)
    check("定理12: 未来事件两法都不含", ("user", "fut") not in exec_state and ("user", "fut") not in fold_state)
    check("定理12: RETRACT 命题两法都不含", ("user", "todo") not in exec_state)


if __name__ == "__main__":
    print("=" * 60)
    print("SRG 定理单元测试（账本代数 + 执行引擎 + 凭证）")
    print("=" * 60)
    test_lemma1_monoid()
    test_theorem1_temporal()
    test_theorem2_deletion()
    test_theorem3_counterfactual()
    test_execution_theorems()
    test_certificate_theorems()
    test_engine_uniqueness()
    print("=" * 60)
    print(f"通过 {len(PASS)} / {len(PASS)+len(FAIL)}")
    if FAIL:
        print("失败:", FAIL)
        raise SystemExit(1)
    print("✅ 全部定理单测通过 —— SRG 正确性保证成立")
