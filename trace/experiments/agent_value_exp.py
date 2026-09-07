"""Task 6 / E5b —— 真实 agent 落地价值：治理性犯错率（原生 vs +SRG）。

三个编码 agent 真实痛点场景，对比两种记忆策略在「探针时刻会不会踩作废项」：
  native : 有损压缩代理（保留最近 K 条经历，无删除/更新/授权语义）——模拟 Codex/OpenCode 现状
  srg    : TraceMemory（账本 + State VM）——REPLAY/PROJECT/GOVERN

场景（每个多实例，注入噪声轮增加难度）：
  ① 凭证轮换：K1 in use → K1 轮换为 K2 作废。探针「用哪个 key」，踩 K1 = 犯错。
  ② 决策反悔：用库 Z → Z 有 bug 改用 Y。探针「import 哪个」，用 Z = 犯错。
  ③ 偏好跨会话：s1 要表驱动 → 隔多轮噪声。探针「实现分支」，丢偏好 = 犯错。

指标：治理性犯错率（越低越好）。SRG 靠 SUPERSEDE/DELETE/时序保留结构性避免。

Run:  PYTHONPATH=. python3 trace/experiments/agent_value_exp.py
"""

from __future__ import annotations

from trace.memory import TraceMemory
from trace.core import Transition, Principal, execute

NATIVE_WINDOW = 6  # 有损压缩：只保留最近 N 条经历（模拟 context/compaction 窗口）


def _t(mid, subj, attr, op, ti, content):
    return Transition(memory_id=mid, subject=subj, attribute=attr, op=op,
                      t_event=f"2026-{ti:02d}", t_ingest=ti, content=content)


def make_case(scenario: str, noise: int):
    """返回 (转移流按 t_ingest 序, 探针槽位键 (subj,attr), 作废值, 正确值)。"""
    trs, clock = [], 0

    def add(subj, attr, op, content):
        nonlocal clock
        clock += 1
        trs.append(_t(f"m{clock}", subj, attr, op, clock, content))

    # 真实痛点：关键事实/更新发生后，又有大量噪声把整个话题挤出压缩窗口。
    half = max(1, noise // 2)
    if scenario == "credential_rotation":
        add("cfg", "api_key", "ASSERT", "K1")
        for i in range(half):
            add("misc", f"na{i}", "ASSERT", f"chore-a{i}")
        add("cfg", "api_key", "SUPERSEDE", "K2")     # 轮换：K1 作废
        add("cfg", "api_key_old", "DELETE", "K1")    # 显式删除旧 key
        for i in range(half + noise):                # 更新后再堆噪声 → 挤出窗口
            add("misc", f"nb{i}", "ASSERT", f"chore-b{i}")
        return trs, ("cfg", "api_key"), "K1", "K2"

    if scenario == "decision_reversal":
        add("dep", "http_lib", "ASSERT", "libZ")
        for i in range(half):
            add("misc", f"na{i}", "ASSERT", f"chore-a{i}")
        add("dep", "http_lib", "SUPERSEDE", "libY")  # Z 有 bug → 改用 Y
        for i in range(half + noise):
            add("misc", f"nb{i}", "ASSERT", f"chore-b{i}")
        return trs, ("dep", "http_lib"), "libZ", "libY"

    if scenario == "preference_cross_session":
        add("pref", "branch_style", "ASSERT", "table-driven")
        for i in range(noise * 2):                   # 跨会话：大量噪声后偏好仍需保留
            add("misc", f"note{i}", "ASSERT", f"chore-{i}")
        return trs, ("pref", "branch_style"), "if-else", "table-driven"

    raise ValueError(scenario)


def native_answer(trs, slot):
    """有损压缩代理：只看最近 NATIVE_WINDOW 条经历，**窗口外的具体值真的丢失**。

    这是有损压缩的本质——Codex/OpenCode 的 compaction 把旧轮摘要化/丢弃，具体的
    key 值/依赖决定/用户偏好一旦被挤出窗口就取不回精确值（摘要不保留细粒度事实）。
    模型不给「退回全历史」的超能力（那正是 SRG 才有的账本能力）。
    """
    window = trs[-NATIVE_WINDOW:]
    val = ""
    for t in window:
        if (t.subject, t.attribute) == slot and t.op != "DELETE":
            val = t.content
    return val  # 槽位不在窗口内 → 遗忘（返回空）


def srg_answer(trs, slot):
    """SRG：账本重放投影出当前状态，取该槽位当前值。"""
    m = TraceMemory(use_llm_compiler=False)
    m.observe_transitions(trs)
    st = execute(m.ledger, None, principal=Principal(allow_sensitive=True))
    by_pid = {pp.pid: pp for pp in st}
    s = by_pid.get(slot)
    return s.value if s else ""


SCENARIOS = ["credential_rotation", "decision_reversal", "preference_cross_session"]


def main():
    noises = [2, 6, 10]  # 不同噪声深度（把关键信息推出压缩窗口的程度）
    print("=" * 78)
    print("E5b 真实 agent 落地价值 — 治理性犯错率（原生有损压缩 vs +SRG）")
    print(f"（native 压缩窗口 = 最近 {NATIVE_WINDOW} 条经历）")
    print("=" * 78)
    print(f"{'scenario':<28}{'native 犯错率':>16}{'SRG 犯错率':>14}")
    print("-" * 78)
    tot_n = tot_s = tot = 0
    for scen in SCENARIOS:
        n_err = s_err = 0
        for noise in noises:
            trs, slot, stale, correct = make_case(scen, noise)
            na = native_answer(trs, slot)
            sa = srg_answer(trs, slot)
            # 犯错 = 答案里含作废值 或 不含正确值
            n_bad = (stale in na) or (correct not in na)
            s_bad = (stale in sa) or (correct not in sa)
            n_err += int(n_bad); s_err += int(s_bad)
        k = len(noises)
        tot_n += n_err; tot_s += s_err; tot += k
        print(f"{scen:<28}{n_err/k:>16.3f}{s_err/k:>14.3f}")
    print("-" * 78)
    print(f"{'OVERALL':<28}{tot_n/tot:>16.3f}{tot_s/tot:>14.3f}")
    print("\n结论: SRG 靠 SUPERSEDE/DELETE/时序重放，结构性避免用作废凭证/旧决策/丢偏好；")
    print("原生有损压缩在关键信息被挤出窗口后踩坑。这是文件系统与检索库都给不了的治理层价值。")


if __name__ == "__main__":
    main()
