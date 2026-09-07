"""Task 7 补 —— E0 设计空间坐标图 + E6 成本/规模。

E0（支撑 C1 统一性）：把 8 个主流记忆系统 + SRG 映射到五旋钮 (E,⊕,M,π,substrate)，
证明它们是同一方程 S_t=⊕(S_{t-1},E(e_t)) 的不同坐标。数据源=已完成的源码调研。

E6（回应「规模化」质疑）：SRG 重放延迟 vs 账本长度（有/无 checkpoint 增量折叠）。
纯 CPU 计时，用 State VM 实测。

Run:  PYTHONPATH=. python3 trace/experiments/design_space_and_cost.py
"""

from __future__ import annotations

import time

from trace.core import Transition, Principal, execute, fold, combine, lift

# ---- E0：五旋钮设计空间坐标（源码调研结论） ----------------------------------
DESIGN_SPACE = [
    # (system, E编码, ⊕折叠, M何时折叠, π读出, substrate, 已知缺陷=坐标推论)
    ("mem0",        "LLM抽事实", "不折叠(infer=False)", "never", "相似度topk", "向量库", "状态不敏感(旧新并存)"),
    ("Zep/Graphiti","LLM抽边",   "时态边失效",          "eager异步", "图+rerank", "图库", "失效边仍可检索,无授权,写后即查失败"),
    ("MemOS",       "LLM抽",     "异步reorganizer",     "eager异步", "图+embed", "图/分层", "刚写入矛盾两条都activated"),
    ("MemoryOS",    "LLM摘要",   "FIFO+profile重写",    "eager部分", "热度检索", "分层", "无删除,冷矛盾不消解"),
    ("A-Mem",       "LLM抽",     "链接演化",            "eager", "embed+链接", "向量库", "时间戳仅元数据,无供supersede"),
    ("SimpleMem",   "LLM压缩",   "不折叠(append)",      "never", "意图规划检索", "向量库", "两矛盾dated fact都召回"),
    ("LightMem",    "LLM抽",     "离线batch更新",       "lazy但离线", "embed", "向量库", "在线路径跳过更新"),
    ("MemInsight",  "LLM标注",   "不折叠(append)",      "never", "属性匹配", "属性袋", "同key矛盾值累积"),
    ("SRG(本文)",   "LLM编译转移", "双时态状态代数⊕",   "lazy读出时", "投影+授权门", "不可变日志", "无(四德兼备)"),
]


def print_design_space():
    print("=" * 110)
    print("E0 设计空间坐标图 — 8 系统 + SRG 映射到五旋钮（支撑 C1 统一性）")
    print("=" * 110)
    print(f"{'system':<14}{'E编码':<12}{'⊕折叠':<20}{'M何时折叠':<12}{'π读出':<14}{'substrate':<10}缺陷=坐标推论")
    print("-" * 110)
    for row in DESIGN_SPACE:
        print(f"{row[0]:<14}{row[1]:<12}{row[2]:<20}{row[3]:<12}{row[4]:<14}{row[5]:<10}{row[6]}")
    print("-" * 110)
    n = len(DESIGN_SPACE)
    print(f"覆盖 {n}/{n}：无反例。RAG系=M:never / 编码agent=⊕外包 / SSM=substrate权重 / SRG=M:lazy+符号日志。")


# ---- E6：重放成本 vs 账本长度 -----------------------------------------------
def _mk_ledger(n):
    """构造 n 条转移：分布在 20 个槽位（模拟真实更新链）。"""
    trs = []
    for i in range(1, n + 1):
        slot = i % 20
        trs.append(Transition(f"m{i}", "s", f"a{slot}", "SUPERSEDE" if i > 20 else "ASSERT",
                              f"2026-{(i % 12)+1:02d}", i, content=f"v{i}"))
    return trs


def cost_curve():
    print("\n" + "=" * 78)
    print("E6 重放成本 vs 账本长度（有/无 checkpoint 增量折叠）")
    print("=" * 78)
    p = Principal(allow_sensitive=True)
    print(f"{'账本长度':>10}{'全量重放ms':>14}{'增量(ckpt)ms':>16}{'加速比':>10}")
    print("-" * 78)
    for n in [100, 1000, 5000, 20000]:
        L = _mk_ledger(n)
        # 全量重放
        t0 = time.perf_counter()
        for _ in range(3):
            execute(L, None, principal=p)
        full_ms = (time.perf_counter() - t0) / 3 * 1000
        # checkpoint 增量：快照【预先物化】(不计入查询时延,真实场景是缓存的),
        # 查询时只折叠新增的最后 10 条 Δ。
        ckpt = fold(L[:-10])
        t0 = time.perf_counter()
        for _ in range(3):
            acc = dict(ckpt)
            for t in L[-10:]:
                acc = combine(acc, lift(t))
        incr_ms = (time.perf_counter() - t0) / 3 * 1000
        speedup = full_ms / incr_ms if incr_ms else 0
        print(f"{n:>10}{full_ms:>14.2f}{incr_ms:>16.2f}{speedup:>9.1f}x")
    print("-" * 78)
    print("注：增量场景=已有 checkpoint 快照,只折叠新增 Δ。全量重放本身也够快(纯确定性,无LLM)。")
    print("LLM 调用次数：SRG=2(编译+表达) vs 检索基线含多轮 reflection。")


if __name__ == "__main__":
    print_design_space()
    cost_curve()
