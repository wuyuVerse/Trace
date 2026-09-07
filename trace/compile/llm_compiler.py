"""LLM 编译器 —— SRG 的 E 部件：把自由文本经历编译成结构化状态转移。

SRG 唯一需要 LLM 的写入侧部件（读出侧 State VM 纯确定性）。设计：
- 结构化 JSON 输出（op 枚举 + subject/attribute/value + 时间），杜绝自由文本漂移。
- 逐条经历（带时间戳）编译；同一 (subject, attribute) 的新值自动构成 SUPERSEDE。
- 输出可直接喂进 State VM。
"""

from __future__ import annotations

import re as _re

from trace.runtime.llm_client import chat, extract_json, DEFAULT_MODEL
from trace.core import Transition
from trace.core.machine import NARRATIVE_ATTR
from trace.core.relevance import tokenize

_SYS = """You are a memory compiler. You convert a user's message into structured STATE TRANSITIONS for a memory ledger.

A transition records how the world STATE changes, not just what was said. Output ONLY a JSON array. Each element:
{"op": "ASSERT|SUPERSEDE|RETRACT|DELETE", "subject": "<entity>", "attribute": "<property>", "value": "<current value>"}

Rules:
- subject = the entity the fact is about (e.g. "user", "user's 5K time", "user's address").
- attribute = the specific property (e.g. "personal_best_5k", "home_city"). Use a STABLE snake_case key so that a later update to the SAME property reuses the SAME (subject, attribute).
- value = the concrete current value (keep numbers/times/places verbatim, e.g. "25:50", "Shanghai").
- op = ASSERT for a new fact; SUPERSEDE if it updates/replaces a previously stated value of the same property; RETRACT if the user says something is no longer true; DELETE if the user asks to forget it. (ACCUMULATE op also exists for multi-valued lists but is applied by deterministic post-processing, not by you — just emit one ASSERT per value with the SAME attribute for list properties like hobbies/interests.)
- Emit a transition for EVERY durable fact: not only "property = value" facts, but also EVENTS, ACTIONS, TASKS, and RELATIONS the user mentions. Examples that MUST be captured:
    * task/todo: "I need to return the boots to Zara" → {"op":"ASSERT","subject":"return_boots_zara","attribute":"todo","value":"return boots to Zara"}
    * action/event: "I picked up my navy blazer from dry cleaning" → {"op":"ASSERT","subject":"pickup_navy_blazer","attribute":"event","value":"picked up navy blazer"}
    * one distinct proposition PER item — do NOT merge multiple items into one; if the user mentions 3 clothing tasks, emit 3 transitions with DIFFERENT subjects (so they can be counted).
- Use a UNIQUE subject per distinct thing/event (e.g. return_boots_zara, pickup_navy_blazer). Only reuse the SAME (subject,attribute) when a LATER message updates the SAME thing (then SUPERSEDE/RETRACT).
- STATE STATEMENTS (residence / identity / job / preference / relationship) MUST be captured with a STABLE attribute so a later value OVERWRITES (SUPERSEDE) the old one. **Prefer over-capturing: whenever the user mentions WHERE THEY LIVE / A CITY OR NEIGHBORHOOD they are in, moving to, relocating to, or "in X now" — in ANY tense or phrasing (moved to / moving to / relocating to / live in / I'm in / based in / X for a job/project) — ALWAYS emit `user.current_residence = <place>`.** Ignore the reason clause ("for a creative project"); the PLACE is the fact.
    * "I'm moving to Roma Norte, Mexico City for a project" → user.current_residence = Roma Norte, Mexico City
    * "I'm relocating to Gangnam, Seoul" → user.current_residence = Gangnam, Seoul
    * "my neighbor is Soo-jin" → user.neighbor = Soo-jin ; "I work at X" → user.job = X.
  Every new residence/job/neighbor value for the SAME attribute is a SUPERSEDE (old one no longer holds). NEVER skip a place-mention as chit-chat.
- Also capture RECURRING ROUTINES tied to the user with STABLE attributes: "On Saturdays I visit the flea market" → user.saturday_activity ; "my morning coffee is at X" → user.morning_routine ; "my breakfast is Y" → user.breakfast_routine. A message like "I live in Alfama. On Saturdays I visit the flea market" yields TWO transitions (current_residence AND saturday_activity). A later different routine SUPERSEDEs the old.
- UPDATE SIGNALS: "now/currently/updated to/changed to/new X is/beat my previous/my latest" = CURRENT value → emit (SUPERSEDE if property seen before, else ASSERT). Capture newest concrete value even if phrased as a goal that was achieved.
- **EVENT TIME (critical for "which came first / ordering / how long ago" questions): if the fact mentions WHEN the event actually happened relative to now — "last month", "two weeks ago", "since February 20th", "yesterday", "recently", "pre-ordered", "just got" — add an "event_time" field with that phrase.** The event's real time is often DIFFERENT from when it was mentioned. Example: "I pre-ordered the Dell last month, and just got the Samsung today" → two transitions, Dell with {"event_time":"last month"}, Samsung with {"event_time":"today"} — so ordering reflects the true acquisition order, not the mention order.
- COMPLETION/CANCEL SIGNALS: "done/finished/completed/already did/no longer need/cancelled" → RETRACT the corresponding todo/task proposition.
- Emit [] only for pure chit-chat/questions with no durable fact. Otherwise capture ALL facts/events/tasks (can be 3-6 per message).

Output ONLY the JSON array, nothing else."""


def compile_episode(text: str, *, t_event: str, t_ingest: int, model: str = DEFAULT_MODEL,
                    prior: str = "") -> list[Transition]:
    """把一条经历（一句话/一轮）编译成 0..N 条 Transition。"""
    user = text if not prior else f"Known so far: {prior}\n\nNew message: {text}"
    # ★max_tokens 必须给足：deepseek-v4-flash 是 reasoning 模型，先输出 reasoning_content(思考)再输出
    #   JSON。512/2048 会被思考吃光 → JSON 未生成就截断 → extract_json 返回 None → 结构命题全空、
    #   只剩 narrative 兜底(实证:整批崩到 0.017)。给足 token + 抽不到 JSON 时加倍重试，确保结构化编译不空。
    import os as _os
    _mt = int(_os.getenv("TRACE_COMPILE_MAX_TOKENS",
                          _os.getenv("SRG_COMPILE_MAX_TOKENS", "4096")))
    arr = None
    for _mt_try in (_mt, _mt * 2):   # 首轮足量；仍抽不到(reasoning 特别长)则加倍再试一次
        out = chat(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
            model=model, temperature=0.0, max_tokens=_mt_try,
        )
        arr = extract_json(out)
        if isinstance(arr, list):
            break
    if not isinstance(arr, list):
        arr = []   # 两轮仍失败 → 结构面空,narrative 安全网兜底(仅极端情况,不应常发生)
    # ★强 forget/expire 指令检测（确定性，修 AMB safety：弱编译器常把"has expired"/"forget and delete"
    #   编成 ASSERT 而非 DELETE→被禁记忆存活）。命中则该轮结构命题的 ASSERT/SUPERSEDE 强制降为 DELETE
    #   （该轮是在【宣告某事物失效/删除】，不是在断言新事实）。高精度短语，避免误删普通更新。
    forget_directive = _is_forget_directive(text)
    transitions = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "ASSERT")).upper()
        if op not in ("ASSERT", "SUPERSEDE", "ACCUMULATE", "RETRACT", "DELETE"):
            op = "ASSERT"
        if forget_directive and op in ("ASSERT", "SUPERSEDE", "ACCUMULATE"):
            op = "DELETE"   # 该轮宣告失效/删除 → 结构命题墓碑（execute 排除，不入 S(t)）
        subj = str(item.get("subject", "")).strip() or "user"
        attr = str(item.get("attribute", "")).strip() or "fact"
        val = str(item.get("value", "")).strip()
        subj, attr = _canonicalize(subj, attr)   # 确定性归一：单值状态类归到固定槽位
        # 事件时间：LLM 抽的 event_time 相对短语(last month/today/…)解析成绝对 t_event(相对 session 日期),
        # 让"哪个先"按【事件真实发生时间】排序,而非对话提及时间。解析失败回退 session 日期。
        te = _resolve_event_time(str(item.get("event_time", "")).strip(), t_event)
        # ✗回退(2026-08-07)：曾加确定性兜底 `if te==t_event: te=_resolve_narrative_time(text,t_event)`
        #   想修 locomo WHEN 题日期漂移。同批 A/B 证伪并【净负】：locomo temporal-reasoning 0.893→0.429。
        #   根因是我方向理解反了——locomo gold 本身是【相对措辞】("the week before 6 July 2023"),baseline
        #   靠 verbalize 复述相对短语("The week prior to July 6, 2023")命中 judge;把结构命题 t_event 确定化
        #   成【绝对日期】反而诱导 verbalize 输出具体日期(2023-06-08)→偏离 gold 措辞→judge 判错。故回退。
        #   (longmem 侧 0.733 零回退,因其 temporal 是"X天前"数值差,与措辞无关;但 locomo 明确净负,遵防自欺纪律回退。)
        # ★敏感标签抽取（让 GOVERN 治理生效——之前恒 None，敏感命题绕过治理导致 AMB safety 泄漏）：
        #   ① LLM 抽的 sensitivity 字段；② 确定性兜底：值/属性含密钥/密码/PII 模式则标 sensitive。
        #   引擎的 governance.visible 早已支持按 sensitivity 投影，只是编译从不喂标签——这是编译层缺口。
        sens = _detect_sensitivity(item.get("sensitivity"), attr, val, subj)
        transitions.append(Transition(
            memory_id=f"c{t_ingest}_{i}", subject=subj, attribute=attr, op=op,
            t_event=te, t_ingest=t_ingest * 100 + i,
            valid_until=None, scope="same_user", sensitivity=sens, content=val,
            evidence_keys=_keys(subj, attr, val),   # 语言体：多表述检索 key
        ))
    # 注：ACCUMULATE 执行 op 保留在引擎（语义正确、单测通过、是指令集扩展资产），但【编译侧不自动触发】——
    # 实测在 locomo 上把多值 ASSERT 合并成一条集合命题反而降召回(0.317→0.2)：合并后单值查询召回变差,
    # 多值题救的<单值题丢的。故默认不后处理;需要多值聚合的场景由调用方显式用 ACCUMULATE 指令。
    # transitions = _accumulate_multivalue(transitions)  # 默认关闭(净负)
    # 语言体安全网：把整条经历原文作为一条 narrative 命题挂进 S(t)（结构面空槽,不参与治理收敛,
    # 仅供 read_relevant 召回）。这样即便结构化抽取漏了/压没了信息,原文仍在 S(t) 可被语义召回。
    # 编译从"单点故障"降级为"锦上添花"（0722 §5 不足②的消解）。
    raw = text.strip()
    if raw:
        # narrative 命题的 t_event：默认 session 日期，但若原文含明确相对时间短语(yesterday/last week/
        # the Friday before/two weeks ago)，解析成事件真实日期——WHEN 题命中的多是 narrative，用 session
        # 日期会差 1 天/一周（元凶）。解析失败/无短语则保留 session 日期（安全回退，不臆测）。
        narr_te = _resolve_narrative_time(raw, t_event)
        transitions.append(Transition(
            memory_id=f"c{t_ingest}_narr", subject=f"episode_{t_ingest}", attribute=NARRATIVE_ATTR, op="ASSERT",
            t_event=narr_te, t_ingest=t_ingest * 100 + 99,
            valid_until=None, scope="same_user", sensitivity=None, content=raw,
            evidence_keys=_keys("", "", raw),
        ))
    return transitions


def _resolve_narrative_time(text: str, session_t_event: str) -> str:
    """从 narrative 原文里找【首个明确相对时间短语】解析成事件真实日期。无明确短语则保留 session 日期。
    保守：只认高置信短语，避免把含多事件的长叙事错误偏移（A/B 纪律：不臆测）。"""
    import re
    p = text.lower()
    # 高置信相对短语（按优先级），命中首个即解析
    for cue in ("day before yesterday", "yesterday", "the day before",
                "last monday", "last tuesday", "last wednesday", "last thursday",
                "last friday", "last saturday", "last sunday",
                "monday before", "tuesday before", "wednesday before", "thursday before",
                "friday before", "saturday before", "sunday before",
                "the week before", "last week", "the weekend before",
                "two weeks ago", "three weeks ago", "a week ago", "couple of weeks ago",
                "last month", "last year"):
        if cue in p:
            return _resolve_event_time(cue, session_t_event)
    return session_t_event


# ★高精度 forget/expire 指令正则（2026-08-06 收窄防召回误伤）：
#   曾用宽正则(forget the/this、remove/discard/scrub/purge、do-not-use)在 longmem/locomo 上大量误命中
#   assistant 建议散文("remove the stains"/"discard the water"/"purge")与否定"don't forget the"(=记住!)
#   → 把合法 ASSERT 误翻 DELETE 墓碑 → 召回真回归(longmem 0.833→0.733/locomo 0.633→0.433)。
#   收窄为【只认强销毁/失效指令】：destroy 短语必须带明确宾语(temporary/hotel/preference/context/…)，
#   "forget about" 加否定前瞻挡 "don't forget about"，删掉 use/keep/purge/discard/scrub 等模糊词。
_FORGET_CUES = _re.compile(
    r"\b(forget and delete|delete and forget|please (delete|forget)|"
    r"(?<!don't )(?<!do not )forget about (that|the|this|my)|"
    r"has expired|have expired|this .{0,30}expired|expired.{0,10}context|"
    r"no longer (valid|store|keep|remember)|"
    r"do not (store|retain)|don't (store|retain)|"
    r"delete (that|the|this) (temporary|hotel|preference|context|note|value|key|password|card))\b",
    _re.IGNORECASE)

# 编译单元文本常带 "[user] "/"[assistant] " 角色前缀（srg_e2e/srg_adapter 都加）。遗忘/删除指令
# 只可能来自 **user**（"Forget and delete my…"/"This context has expired"）——assistant 的建议散文
# ("don't store water in…")与确认("I will delete that")绝不能触发墓碑。故 forget-flip 只在 user 轮生效。
_ROLE_PREFIX = _re.compile(r"^\s*\[(user|assistant|system|tool)\]\s*", _re.IGNORECASE)


def _is_forget_directive(text: str) -> bool:
    """该轮是否在【宣告某事物失效/要求删除】（高精度确定性检测）。命中则结构命题降为 DELETE。
    双层防误伤：① 只认强销毁/失效短语（收窄正则）；② 只在 **user** 轮生效（跳过 assistant/system/tool
    的建议散文与确认语，它们绝非删除指令）。无角色前缀（如 locomo 单 blob）时按内容判——但收窄正则
    在无角色场景已实测 0 误命中。"""
    t = text or ""
    m = _ROLE_PREFIX.match(t)
    if m:
        role = m.group(1).lower()
        if role != "user":          # assistant/system/tool 轮：绝不触发遗忘墓碑
            return False
        t = t[m.end():]             # 去掉角色前缀后再匹配指令
    return bool(_FORGET_CUES.search(t))


_FORGET_STOP = {"the", "a", "an", "for", "of", "from", "to", "and", "or", "is", "are", "this",
                "that", "temporary", "context", "preference", "older", "current", "project",
                "workspace", "course", "note", "value", "user", "my", "please", "delete",
                "forget", "remove", "expired", "expire", "old", "no", "longer"}


def _forget_tokens(*parts) -> set:
    """孤儿 DELETE/RETRACT 的判别性 token（去停用词）——只留能指认目标槽的实义词。"""
    toks = set()
    for s in parts:
        toks |= tokenize(s)
    return {t for t in toks if t not in _FORGET_STOP and len(t) > 2}


def reconcile_forget_ops(new_transitions: list, existing_ledger: list) -> list:
    """★编译层确定性指代消解（修 AMB safety forbidden_activation 真根因，不动 execute/定理）：

    弱编译器跨轮给同一现实事物命名漂移——t4 assert `user.hotel_room_preference=near elevator`，
    t5 "forget and delete..." 却出 `hotel_preference_ember_course.room_location` DELETE（不同槽 key）→
    DELETE 墓碑打空槽，原 ASSERT（带 forbidden memory_id）存活→被召回→forbidden_activation。

    本后处理：对每条【孤儿 DELETE/RETRACT】（其 (subject,attribute) 在既有在册槽里找不到），按【值/属性
    判别性 token 重叠】找它真正指代的在册 ASSERT 槽，补发一条指向该槽的 DELETE/RETRACT（execute 即可正确
    墓碑/收回）。高精度阈值（要求实义 token 强重叠）避免误删合法记忆伤召回。零 LLM、确定性、幂等。

    只补发、不改原指令（原孤儿指令墓碑空槽无害）。返回 new_transitions + 补发的重定向指令。
    """
    from dataclasses import replace as _replace
    # 重放既有账本得【在册槽】(subject,attribute)→代表命题（供 token 匹配）
    live: dict = {}
    tomb: set = set()
    for t in sorted(existing_ledger, key=lambda x: x.t_ingest):
        if t.attribute == NARRATIVE_ATTR:
            continue
        pid = (t.subject, t.attribute)
        if t.op == "DELETE":
            live.pop(pid, None); tomb.add(pid); continue
        if pid in tomb:
            continue
        if t.op == "RETRACT":
            live.pop(pid, None); continue
        if t.op in ("ASSERT", "SUPERSEDE", "ACCUMULATE"):
            live[pid] = t
    live_slots = set(live)
    extra = []
    for d in new_transitions:
        if d.op not in ("DELETE", "RETRACT") or d.attribute == NARRATIVE_ATTR:
            continue
        if (d.subject, d.attribute) in live_slots:
            continue  # 非孤儿：已精确命中在册槽，execute 直接处理
        otoks = _forget_tokens(d.subject, d.attribute, d.content)
        if not otoks:
            continue
        # 找判别性 token 强重叠的在册槽：overlap / |otoks| ≥ 0.5 且绝对重叠 ≥ 2
        for pid, t in live.items():
            stoks = _forget_tokens(t.subject, t.attribute, t.content)
            if not stoks:
                continue
            inter = otoks & stoks
            if len(inter) >= 2 and len(inter) / len(otoks) >= 0.5:
                extra.append(_replace(d, subject=pid[0], attribute=pid[1],
                                      memory_id=f"{d.memory_id}_reconcile_{pid[1]}"))
    return list(new_transitions) + extra


def _accumulate_multivalue(transitions: list) -> list:
    """确定性后处理：同一 (subject,attribute) 有 ≥2 条不同值的 ASSERT → 判定为多值列表，改 ACCUMULATE。
    单值状态（一条）保持 ASSERT/SUPERSEDE 不变。narrative 语言体不参与。零 LLM、确定性。"""
    from collections import defaultdict
    from dataclasses import replace
    groups = defaultdict(list)
    for t in transitions:
        if t.op == "ASSERT" and t.attribute != NARRATIVE_ATTR:
            groups[(t.subject, t.attribute)].append(t)
    multi = {k for k, v in groups.items() if len({t.content for t in v if t.content}) >= 2}
    return [replace(t, op="ACCUMULATE") if (t.subject, t.attribute) in multi and t.op == "ASSERT" else t
            for t in transitions]


def _resolve_event_time(phrase: str, session_t_event: str) -> str:
    """把相对时间短语解析成绝对 t_event(相对 session 日期偏移)。让"哪个先"按事件真实时间排序。
    解析不出就回退 session 日期。确定性、无 LLM。"""
    if not phrase:
        return session_t_event
    import re, datetime
    p = phrase.lower()
    # 提取 session 基准日期
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", str(session_t_event))
    if not m:
        return session_t_event
    try:
        base = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return session_t_event
    # 相对偏移（天）
    days = 0
    # 英文数字词（"two weeks ago" / "a week ago" / "couple of days"）——正则只认阿拉伯数字会漏，
    # 实测 "two weeks ago" 匹配失败 → 事件停在 session 日期 → duration 题算出 0。补英文数词表。
    _numw = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "couple": 2, "few": 3, "several": 3}
    nm = re.search(r"(\d+)\s*(day|week|month|year)", p)
    nmw = re.search(r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|several)\b(?:\s+of)?\s*(day|week|month|year)", p)
    _weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                 "friday": 4, "saturday": 5, "sunday": 6}
    # 相对锚点："the Friday/Monday before" / "last Tuesday" / "the week before" —— locomo temporal gold
    #   大量是这种（gold "The Tuesday before 20 July"）。按 base 的星期回退到最近的目标星期。
    anchor = re.search(r"(?:the\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before", p)
    last_wd = re.search(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", p)
    if "yesterday" in p:
        days = -1
    elif "day before" in p:
        days = -1 if "the day before" in p else -2  # "day before yesterday"
    elif anchor or last_wd:
        wd = (anchor or last_wd).group(1)
        target = _weekdays[wd]
        back = (base.weekday() - target) % 7
        back = back or 7   # 同一天则回退整周（"before" 语义必在过去）
        days = -back
    elif "the week before" in p or "week before" in p or "last week" in p:
        days = -7
    elif "the weekend before" in p or "over the weekend" in p:
        days = -((base.weekday() - 5) % 7 or 7)  # 回退到最近的周六
    elif nm:
        n = int(nm.group(1)); unit = nm.group(2)
        days = -n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    elif nmw:
        n = _numw[nmw.group(1)]; unit = nmw.group(2)
        days = -n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    elif "last month" in p:
        days = -30
    elif "last week" in p:
        days = -7
    elif "last year" in p:
        days = -365
    elif any(w in p for w in ("today", "just", "recently", "now")):
        days = 0
    else:
        # 绝对日期短语(如 "February 20th")→尝试当年
        mo = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
              "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        am = re.search(r"([a-z]{3})[a-z]*\s+(\d{1,2})", p)
        if am and am.group(1)[:3] in mo:
            try:
                return datetime.date(base.year, mo[am.group(1)[:3]], int(am.group(2))).isoformat()
            except ValueError:
                return session_t_event
        return session_t_event
    return (base + datetime.timedelta(days=days)).isoformat()


def _keys(subject: str, attribute: str, value: str) -> tuple:
    """从结构面+值生成多表述检索 key（桥接自然语言 query）。确定性、无 LLM。
    复用 core 统一分词 tokenize（单一权威），不自造分词。"""
    toks = set()
    for s in (subject, attribute, value):
        toks |= tokenize(s)
    return tuple(sorted(toks))


# 敏感模式（确定性兜底，无 LLM）：密钥/token/密码/信用卡/SSN 等。命中即标 sensitive，让 GOVERN 治理。
_SENSITIVE_ATTR = ("password", "passwd", "api_key", "apikey", "secret", "token", "credential",
                   "ssn", "credit_card", "card_number", "pin", "private_key", "access_key")
# 主题匹配用【去分隔符后的强复合 token】——只认不会做无辜子串的长/复合词，避免 pin⊂shopping、token 等短词
# 在自由文本 subject 上误标 sensitive（那会把合法记忆对无授权 principal 排除→伤召回）。
_SENSITIVE_SUBJ_NORM = ("apikey", "password", "passwd", "credential", "privatekey", "accesskey",
                        "creditcard", "cardnumber", "ssn", "secretkey", "authtoken", "accesstoken")
_SENSITIVE_VAL = _re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|api[_\- ]?key|password|secret|\bssn\b|\d{3}-\d{2}-\d{4}|"
    r"\b\d{13,16}\b|redacted_api_key|access[_\- ]?key)", _re.IGNORECASE)


def _detect_sensitivity(llm_field, attr: str, val: str, subj: str = "") -> str | None:
    """决定命题的 sensitivity（喂给 execute 的 GOVERN 投影）。
    ① LLM 若显式标了 sensitive/private/secret → sensitive；② 确定性兜底：主题/属性名或值命中密钥/密码/PII 模式。
    无信号返回 None（普通命题）。引擎 governance.visible 按此投影：sensitive 命题对无授权 principal 不可见。

    ★纳入 subject（修 AMB forbidden 缺口）：弱编译常把密钥拆成 `ember_course_api_key.source_workspace/
      .temporary/.sensitive=true` —— 密钥语义落在【主题】(ember_course_api_key 含 api_key)，而 value/attr 是
      纯属性值('older workspace'/'true')不含密钥模式 → 旧逻辑只看 attr/val 恒漏标。纳入 subject 后整簇标 sensitive,
      GOVERN 对无授权 principal 结构排除,不进 S(t)、不被召回(forbidden_activation↓)。另:属性名/值为 sensitive/
      confidential 的显式自标记(.sensitive=true)也直接判 sensitive。"""
    if isinstance(llm_field, str) and llm_field.strip().lower() in ("sensitive", "private", "secret", "confidential"):
        return "sensitive"
    a = (attr or "").lower()
    if any(k in a for k in _SENSITIVE_ATTR):
        return "sensitive"
    # 主题侧：去掉 _/-/空格后匹配强复合词（apikey/password/…）——弱编译常把密钥语义塞进 subject
    #   (ember_course_api_key.*)，而这些复合词无辜子串风险低，安全放行治理。
    s_norm = _re.sub(r"[\s_\-]+", "", (subj or "").lower())
    if any(k in s_norm for k in _SENSITIVE_SUBJ_NORM):
        return "sensitive"
    # 显式自标记：属性名是 sensitive/confidential 且值为真 → 该现实事物被声明敏感（整簇同 subject 已由上面兜住）。
    if a in ("sensitive", "confidential", "private", "secret") and (val or "").strip().lower() in ("true", "yes", "1"):
        return "sensitive"
    if val and _SENSITIVE_VAL.search(val):
        return "sensitive"
    return None


# 单值状态槽位归一：LLM 可能把"住哪"命名成 current_residence/residence/home/current_city/moved_to…，
# 确定性地把它们归到同一 (user, 固定attr)，这样多次搬家落同槽 → SUPERSEDE 收敛到最新（不靠 LLM 每次命名一致）。
_CANON = [
    ("current_residence", ("residen", "home_city", "current_city", "home_location", "moved_to",
                           "moving_to", "relocat", "based_in", "current_residence", "current_location")),
    ("job", ("job", "employ", "occupation", "current_company", "workplace")),
    ("neighbor", ("neighbor", "neighbour")),
    ("saturday_activity", ("saturday_activ", "saturday_rout", "weekend_activ")),
    ("morning_routine", ("morning_rout", "morning_coffee", "breakfast_rout")),
]
# 只有 subject 明确指"用户本人"时才归一——避免把活动/他人实体的 location/time 等误并到 user 槽位
# (如 membench 的 ClimbFest.location 不是用户居住地)。
_USER_SUBJ = ("user", "i", "me", "my", "我", "self")


# 强搬家信号：LLM 常把"relocating to Kyoto"编成 gion.moved_to/kyoto.relocation 等【非 user subject】，
# 窄归一不处理→最新居住地丢失(实证:16个系统性失败全是residence停在旧城)。故强搬家信号无视 subject 归一。
# 注意 attr 需含明确"搬家"动作词(moved/relocat)，不含裸 location/city(避免误并活动地点)。
_STRONG_MOVE = ("moved_to", "moving_to", "relocat", "moved to", "moving to", "relocation")


def _canonicalize(subject: str, attr: str) -> tuple[str, str]:
    key = f"{subject} {attr}".lower()
    if any(c in key for c in _STRONG_MOVE):
        return "user", "current_residence"   # 强搬家信号(任意subject)→用户当前居住地
    s = subject.lower().strip()
    if s not in _USER_SUBJ and not s.startswith("user"):
        return subject, attr   # 非用户实体(活动/他人)的弱信号(location/city)不归一，避免误并
    for canon, cues in _CANON:
        if any(c in key for c in cues):
            return "user", canon
    return subject, attr


def compile_stream(episodes: list[tuple[str, str]], *, model: str = DEFAULT_MODEL) -> list[Transition]:
    """把一串带时间戳的经历 [(t_event, text)] 顺序编译成完整账本。"""
    ledger: list[Transition] = []
    for clock, (t_event, text) in enumerate(episodes, start=1):
        ledger.extend(compile_episode(text, t_event=t_event, t_ingest=clock, model=model))
    return ledger
