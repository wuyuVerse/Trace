"""SRG 核心数据类型：双时态状态转移与折叠后的槽位。

一条 Transition 记录「世界状态如何改变」，不是「说了什么」。这是 SRG 与所有
检索式记忆最本质的分野——存的是 change，不是 content。
"""

from __future__ import annotations

from dataclasses import dataclass

# 六类状态转移（对应愿景文档 §4.1）。
#   ASSERT    新增一个成立的命题
#   SUPERSEDE 新值取代同 (subject,attribute) 的旧值（旧值退场但不删）
#   RETRACT   收回该 (subject,attribute)（不再成立）
#   DELETE    治理性删除（此后任何回忆不得激活；留墓碑，可审计）
#   EXPIRE    赋予有效期上界
#   AUTHORIZE 赋予授权域 / 敏感级（读出时按 principal 投影）
OPS = ("ASSERT", "SUPERSEDE", "ACCUMULATE", "RETRACT", "DELETE", "EXPIRE", "AUTHORIZE")



@dataclass(frozen=True)
class Transition:
    """双时态状态转移。一条指令有两个面（0722 统一本体）：
      · 结构面：(subject, attribute) 决定它作用的状态槽位，value/op 是可治理断言。
      · 语言体：content(原文片段) + evidence_keys(多表述) 是该断言的自然语言出处，供 read_relevant 召回。
    二者是同一条指令的投影与出处，非两条数据。"""
    memory_id: str
    subject: str
    attribute: str
    op: str
    t_event: str          # 世界里生效时间（event time）
    t_ingest: int         # agent 知晓次序（ingestion clock，全序）
    valid_until: str | None = None
    scope: str = "same_user"
    sensitivity: str | None = None
    content: str = ""     # 语言体：原文片段（证据句）
    evidence_keys: tuple = ()  # 语言体：fact/keyphrase/summary 多表述，桥接自然语言 query（不进 core 成员资格判定）


@dataclass(frozen=True)
class Slot:
    """一个 (subject,attribute) 槽位折叠后的最新裁决（LWW 语义）。"""
    memory_id: str
    op: str
    t_ingest: int
    t_event: str
    valid_until: str | None
    scope: str
    sensitivity: str | None
    content: str
    tombstoned: bool = False


# State = 从槽位 key (subject,attribute) 到 Slot 的映射。不可变语义，⊕ 返回新 dict。
State = dict
EMPTY: State = {}  # 幺元 ∅
