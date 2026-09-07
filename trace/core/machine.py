"""记忆虚拟机（Memory Machine）—— 唯一核心：回忆即执行，不是检索。

核心创新点（一个 idea 解释全部）：
    记忆不是"你去读的东西"，是"你去运行的程序"。
    - 经历 → 编译成【指令】(instruction)：断言/收回/取代/删除一个命题的成立性。
    - 账本 = 程序（不可变指令序列，字节码）。
    - 回忆 = 把程序【执行】到查询点 t，读出那一刻的机器状态。
    - 机器状态 S(t) = 此刻【成立的命题集合】(propositions-in-force)。
    - 一切 query = 对 S(t) 的读取（读当前值/计数/时序/可采纳性/反事实），同一个执行引擎。

为什么各维度 SOTA 是自然结果（零缝合）：
    能力 = 指令集表达力。指令集可扩展（不限"属性=值"，可含事件/关系/计数命题），
    但执行引擎唯一。补全指令集 = 补全能力，同一引擎执行。

为什么可采纳性天生正确：
    被 DELETE 的命题从不被执行进状态、被 RETRACT 的执行后收回——不是"过滤掉"，
    是"从没执行出来"。合规是执行语义的必然，不是附加过滤器。

指令 = 复用 Transition（op/subject/attribute/content/t_event/t_ingest/scope/sensitivity/valid_until），
但语义是"改变命题成立性"而非"给属性赋值"。命题 id = (subject, attribute)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from trace.core.transition import Transition
from trace.core.governance import Principal, visible

# 语言体命题的 attribute 标记（原文安全网）。单一权威常量，杜绝散落的魔法字符串。
NARRATIVE_ATTR = "narrative"


@dataclass(frozen=True)
class Proposition:
    """执行后仍成立的一个命题（富机器状态 S(t) 的一个元素）。

    两个面（0722 统一本体）：结构面 (subject,attribute,value) 供治理/计数/时序读；
    语言体 body=(value 即原文时 content, evidence_keys) 供 read_relevant 语义召回。
    body 只是透传，不参与成员资格判定——被删/过期/越权命题连同 body 从不进 S(t)（定理2）。"""
    subject: str
    attribute: str
    value: str
    t_event: str
    memory_id: str
    scope: str = "same_user"
    sensitivity: str | None = None
    valid_until: str | None = None
    evidence_keys: tuple = ()   # 语言体：多表述检索 key（透传自指令，不进成员资格判定）

    @property
    def pid(self):
        return (self.subject, self.attribute)

    @property
    def body(self) -> str:
        """语言体：用于 read_relevant 语义召回的原文文本（value 即编译出的原文/证据句）。"""
        return self.value

    @property
    def is_narrative(self) -> bool:
        """是否语言体命题（原文安全网，非结构状态）。结构读跳过它，read_relevant 纳入它。"""
        return self.attribute == NARRATIVE_ATTR


def execute(program: list[Transition], t_query: str | None = None,
            principal: Principal | None = None) -> list[Proposition]:
    """执行程序到 t_query，返回此刻【成立的命题集合】S(t)。

    这是唯一的核心操作。所有 query 都读它的输出。
    执行语义（按 t_ingest 顺序逐条执行指令，改变命题成立性）：
      ASSERT (s,a)=v     → 命题 (s,a) 成立，值 v
      SUPERSEDE (s,a)=v  → 取代 (s,a) 的旧值为 v（旧命题不再成立）——单值状态收敛（当前值）
      ACCUMULATE (s,a)+=v→ 把 v 累加进 (s,a) 的值集合（多值状态收敛，如爱好/清单）——检索范式无此语义
      RETRACT (s,a)      → (s,a) 收回，不再成立
      DELETE (s,a)       → (s,a) 抹除且墓碑（此后任何执行不得使其成立）
      EXPIRE/valid_until → 执行到 t_query 时若已过期则不成立
    双时态：只执行 t_event ≤ t_query 的指令（未来事件尚未发生）。
    """
    # 按 ingestion 顺序执行（LWW：晚执行的覆盖早的）
    insns = sorted(program, key=lambda i: i.t_ingest)
    live: dict[tuple, Proposition] = {}   # 当前成立的命题 (s,a) -> Proposition
    tomb: set[tuple] = set()              # 墓碑：被 DELETE 的命题永不复活
    accum: dict[tuple, list] = {}         # ACCUMULATE 槽的有序去重值集合

    for ins in insns:
        if t_query is not None and ins.t_event and ins.t_event > t_query:
            continue  # 未来事件，尚未执行到
        pid = (ins.subject, ins.attribute)
        op = ins.op
        if op == "DELETE":
            live.pop(pid, None); accum.pop(pid, None)
            tomb.add(pid)
            continue
        if pid in tomb:
            continue  # 合规删除后不可再成立（执行语义强制）
        if op in ("RETRACT",):
            live.pop(pid, None); accum.pop(pid, None)
            continue
        if op == "ACCUMULATE":
            # 多值累加：同槽 v 进有序去重集合，值 = 集合的逗号拼接（read 直接得全部值）。
            vals = accum.setdefault(pid, [])
            v = ins.content.strip()
            if v and v not in vals:
                vals.append(v)
            live[pid] = Proposition(
                subject=ins.subject, attribute=ins.attribute, value=", ".join(vals),
                t_event=ins.t_event, memory_id=ins.memory_id,
                scope=ins.scope, sensitivity=ins.sensitivity, valid_until=ins.valid_until,
                evidence_keys=getattr(ins, "evidence_keys", ()))
            continue
        if op in ("ASSERT", "SUPERSEDE"):
            live[pid] = Proposition(
                subject=ins.subject, attribute=ins.attribute, value=ins.content,
                t_event=ins.t_event, memory_id=ins.memory_id,
                scope=ins.scope, sensitivity=ins.sensitivity, valid_until=ins.valid_until,
                evidence_keys=getattr(ins, "evidence_keys", ()))   # 语言体透传

    # 投影：过期命题不成立
    out = []
    for p in live.values():
        if t_query is not None and p.valid_until is not None and p.valid_until < t_query:
            continue
        # 授权：越权/敏感对该 principal 不成立（可见性也是执行语义的一部分）
        if principal is not None and not visible(p.scope, p.sensitivity, principal):
            continue
        out.append(p)
    return out


def execute_counterfactual(program: list[Transition], *, remove_memory_ids=None,
                           inject: list[Transition] | None = None,
                           t_query: str | None = None, principal: Principal | None = None):
    """反事实执行：改指令后重跑（像改代码重跑）。移除某些指令 / 注入假设指令。"""
    remove = remove_memory_ids or set()
    forked = [i for i in program if i.memory_id not in remove] + list(inject or [])
    return execute(forked, t_query, principal)


def trace(program: list[Transition], memory_id: str) -> list[dict]:
    """执行轨迹（audit）：某命题被哪些指令改变过成立性——可复现、可审计。"""
    return [{"op": i.op, "pid": (i.subject, i.attribute), "value": i.content,
             "t_event": i.t_event, "t_ingest": i.t_ingest}
            for i in sorted(program, key=lambda i: i.t_ingest) if i.memory_id == memory_id]


# ---- 读出：一切 query 都是对机器状态 S(t) 的纯读取（同一执行输出，不同读法）----

def read_current(state: list[Proposition], subject_hint: str = "") -> list[Proposition]:
    """读当前值：问"现在的 X"。可按 subject/attribute 词过滤。"""
    if not subject_hint:
        return state
    h = subject_hint.lower()
    return [p for p in state if h in f"{p.subject} {p.attribute}".lower()] or state


def read_count(state: list[Proposition], predicate_hint: str = "") -> int:
    """计数：问"多少个 X"。数成立命题的个数（之前 (s,a)=v 三元组数不出，命题集能数）。"""
    if not predicate_hint:
        return len(state)
    h = predicate_hint.lower()
    return sum(1 for p in state if h in f"{p.subject} {p.attribute} {p.value}".lower())


# 数值/货币抽取：从命题值里拉出可加总的量（$1,500 / 3 hours / 8 plants）。
_AMOUNT_RE = re.compile(
    r"(?P<cur>[$£€¥])?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|km|kg|miles?|"
    r"dollars?|usd|pounds?|euros?|plants?|books?|times?|people|guests?|items?)?",
    re.I)


def _extract_amount(text: str):
    """从一段文本抽第一个数值 → (float, 归一化单位标签)。抽不到返回 None。
    货币符号/货币词归一为 'currency'；时长词归一为 'duration'；其余用原词或 'count'。"""
    m = _AMOUNT_RE.search(str(text))
    if not m:
        return None
    try:
        val = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    cur, unit = m.group("cur"), (m.group("unit") or "").lower()
    if cur or unit in ("dollars", "dollar", "usd", "pounds", "pound", "euros", "euro"):
        label = "currency"
    elif unit in ("hours", "hour", "hrs", "hr", "minutes", "minute", "mins", "min",
                  "days", "day", "weeks", "week", "months", "month", "years", "year"):
        label = "duration"
    else:
        label = unit or "count"
    return (val, label)


def read_amounts(state: list[Proposition], subject_hint: str = "") -> dict:
    """★确定性数值聚合读出（执行范式独有，与 read_intervals 同一设计哲学）：从成立命题集合
    S(t) 里抽出所有可加总的数值（金额/时长/数量），按归一化单位分组给出【和、明细、条数】——
    直接为"how much did I raise in total / how many hours combined"这类【跨会话求和】题服务。

    为什么这是范式内生能力：execute(L,t) 拿到的是【完整】状态（所有捐款命题都在 S(t)，不像
    检索式召回只捞回一部分），故求和是确定性的、不漏项。检索范式召回不全 → 天然求不准跨会话总额。
    这正是 multi-session 聚合题（longmem 里占 84%）该由账本范式碾压的场景。

    只聚合【结构面命题】（跳过 narrative 语言体，避免同一笔金额在原文句里重复计一次）。
    subject_hint 非空时按 subject/attribute 词过滤（只加相关的那类量，如只加 charity 相关金额）。
    返回 {label: {"sum": float, "items": [(value_str, amount)], "n": int}}。
    """
    h = subject_hint.lower()
    groups: dict[str, dict] = {}
    seen = set()
    for p in state:
        if p.is_narrative:
            continue  # 语言体是原文片段，与结构命题重复计数，跳过
        if h and h not in f"{p.subject} {p.attribute} {p.value}".lower():
            continue
        got = _extract_amount(p.value)
        if got is None:
            continue
        amount, label = got
        # 去重：同一 (subject,attribute,amount) 只计一次（防同笔金额被 ASSERT 多次）
        key = (p.subject, p.attribute, amount)
        if key in seen:
            continue
        seen.add(key)
        g = groups.setdefault(label, {"sum": 0.0, "items": [], "n": 0})
        g["sum"] += amount
        # items 带 (subject, attribute) 供表达器按话题归类（求和题几乎全是话题限定，
        # 如 bike-related：chain_replacement|cost / bike_lights_install|cost 属之，
        # japan_travel|cost 不属）。structured 命题的 subject/attribute 是确定性话题锚。
        g["items"].append((str(p.value)[:60], amount, str(p.subject), str(p.attribute)))
        g["n"] += 1
    return groups


def _parse_date(t_event: str):
    """从 t_event 抽日期为 (year,month,day) 序数，供确定性时间差。解析失败返回 None。
    支持 '2023/01/13 (Fri) 18:07' / '2023-01-13' / '2023/1/13' 等常见格式。"""
    import re
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", str(t_event))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # 转为绝对天序数（近似：年*365+月*30+日，够算区间天数差；跨年也单调）。
    import datetime
    try:
        return datetime.date(y, mo, d).toordinal()
    except ValueError:
        return None


def humanize_days(days: int) -> str:
    """把天数转人类可读时长（"4 years 9 months" / "3 months" / "Two weeks"），供"持续多久"题直接读出。
    执行范式独有：状态命题带 t_event(状态起点)，t_query 已知 → 时长确定性可算，无需 LLM 猜。"""
    if days is None:
        return ""
    d = abs(int(days))
    if d < 14:
        return f"{d} days"
    if d < 60:
        return f"{d // 7} weeks"
    y, rem = divmod(d, 365)
    mo = rem // 30
    if y and mo:
        return f"{y} years {mo} months"
    if y:
        return f"{y} years"
    return f"{max(1, mo)} months"


def read_intervals(state: list[Proposition], t_query: str | None = None) -> list[dict]:
    """★确定性时间差读出（执行范式独有）：命题带 t_event 世界时间 → 直接算事件间隔天数，
    不让 LLM 从文本猜日期。检索范式无结构化 t_event，做不到——这是 SRG 解 temporal 算术的范式内生能力。

    返回按时间排序的事件 + 相邻/相对 t_query 的天数差 + 可读时长（duration_since，供"持续多久"题）。
    """
    dated = [(p, _parse_date(p.t_event)) for p in state]
    dated = [(p, d) for p, d in dated if d is not None]
    dated.sort(key=lambda x: x[1])
    q_ord = _parse_date(t_query) if t_query else None
    out = []
    for i, (p, d) in enumerate(dated):
        row = {"subject": p.subject, "attribute": p.attribute, "value": p.value,
               "t_event": p.t_event, "days_from_prev": (d - dated[i - 1][1]) if i > 0 else 0}
        if q_ord is not None:
            row["days_before_query"] = q_ord - d
            row["duration_since"] = humanize_days(q_ord - d)   # 可读时长："持续多久"题直读
        out.append(row)
    return out


def read_temporal(state: list[Proposition]) -> list[Proposition]:
    """时序：问"哪个先/顺序"。成立命题按 t_event 排序。"""
    return sorted(state, key=lambda p: str(p.t_event))


def read_slot_history(program: list[Transition], t_query: str | None = None,
                      principal: Principal | None = None) -> dict[tuple, list[dict]]:
    """★历史序列读出（执行范式独有的主场能力）：从账本 L 重建每个 (subject,attribute) 槽的
    【完整时序值列表】——不做 SUPERSEDE 塌缩，保留每一次赋值。检索范式没有结构化时间轴、
    做不到"第 N 近的值/变更了几次/前一个值"，这类历史追踪题正是账本范式该碾压的场景。

    与 execute 的关系（语义正交，不破坏任何定理）：execute 读【当前成立集 S(t)】（LWW 塌缩，
    每槽一条），本算子读【同一账本的历史轨迹】（每槽多条按时序）。二者读同一个 L、同一执行顺序，
    只是一个投影到"当前"、一个投影到"全历史"。

    尊重的执行语义：① 双时态——只纳入 t_event ≤ t_query 的赋值；② 墓碑——被 DELETE 的槽其历史
    整体不可见（合规删除对历史读同样生效）；③ 授权——越权/敏感命题对该 principal 不进历史。
    RETRACT 只结束"当前成立"，不抹历史（该值确实曾成立过），故保留在历史中。

    返回：{(subject,attribute): [{value,t_event,t_ingest,op,memory_id}, ...]}，每槽按 t_ingest 升序。
    """
    insns = sorted(program, key=lambda i: i.t_ingest)
    tomb: set[tuple] = set()
    hist: dict[tuple, list[dict]] = {}
    for ins in insns:
        if t_query is not None and ins.t_event and ins.t_event > t_query:
            continue  # 未来事件尚未发生（双时态一致）
        pid = (ins.subject, ins.attribute)
        if ins.op == "DELETE":
            tomb.add(pid)
            hist.pop(pid, None)   # 合规删除：历史整体抹除
            continue
        if pid in tomb:
            continue
        if ins.op in ("ASSERT", "SUPERSEDE", "ACCUMULATE"):
            # 授权/敏感：不可见的赋值不进历史（可见性也是执行语义的一部分）
            if principal is not None and not visible(ins.scope, ins.sensitivity, principal):
                continue
            hist.setdefault(pid, []).append({
                "value": ins.content, "t_event": ins.t_event, "t_ingest": ins.t_ingest,
                "op": ins.op, "memory_id": ins.memory_id})
    return hist


def excluded_report(program: list[Transition], t_query: str | None, principal: Principal | None):
    """可采纳性审计：哪些命题被执行语义排除了(deleted/retracted/expired/unauthorized)+理由+memory_id。

    memory_id 供 benchmark 判"是否误用了该忘的记忆"（forbidden_ids）。
    """
    adm = {p.pid for p in execute(program, t_query, principal)}
    # 重放到 live 槽（未投影前），以便检出"曾成立但被过期/越权排除"的命题——
    # 它们已被 execute 投影删掉、不在输出里，需从 live 单独检出。
    insns = sorted(program, key=lambda i: i.t_ingest)
    live: dict = {}
    tomb: set = set()
    killed = {}
    for ins in insns:
        pid = (ins.subject, ins.attribute)
        if t_query is not None and ins.t_event and ins.t_event > t_query:
            continue
        if ins.op == "DELETE":
            live.pop(pid, None); tomb.add(pid)
            killed[pid] = ("deleted", ins.memory_id)
            continue
        if pid in tomb:
            continue
        if ins.op == "RETRACT":
            live.pop(pid, None)
            killed[pid] = ("superseded", ins.memory_id)
            continue
        if ins.op in ("ASSERT", "SUPERSEDE"):
            live[pid] = ins
            killed.pop(pid, None)  # 重新成立，清掉之前的 retract 记录
    out = []
    for pid, (reason, mid) in killed.items():
        out.append({"subject": pid[0], "attribute": pid[1], "reason": reason, "memory_id": mid})
    # 过期(stale)/越权(unauthorized)：仍在 live 但被投影/授权门排除
    for pid, ins in live.items():
        if pid in adm:
            continue
        if t_query is not None and ins.valid_until is not None and ins.valid_until < t_query:
            reason = "stale"
        else:
            reason = "unauthorized"
        out.append({"subject": pid[0], "attribute": pid[1], "reason": reason, "memory_id": ins.memory_id})
    return out


def active_memory_ids(state: list[Proposition]) -> set[str]:
    """当前成立命题集合对应的 memory_id 集合（替代旧 state_vm 同名函数）。"""
    return {p.memory_id for p in state}


def admissible_state(program: list[Transition], t_query: str | None = None,
                     principal: Principal | None = None):
    """可采纳性单一事实源：一次返回 (成立命题集合 S(t), 被排除项+理由)。

    S(t)=execute 输出恰好是可采纳集 Adm(t,p)（该忘的从没执行出来）；excluded 是审计。
    admissibility/门面/集成都从这一个函数派生，杜绝第二引擎。
    """
    return execute(program, t_query, principal=principal), excluded_report(program, t_query, principal)
