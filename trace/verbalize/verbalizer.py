"""表达器 —— SRG 读出侧：把治理后的当前状态说成话（答案 / 工具调用）。

对应愿景文档 §4.2 GENERATE：LLM 拿到 State VM 算好的干净状态，只负责语言化，不再
推理时间/权限。当前为确定性模板实现（隔离变量）；真实部署替换为 LLM verbalizer。

关键设计：只表达 top-1 相关证据，并主动排除含 must_not_include 的内容，避免 over-dump
触发 must_not / privacy 惩罚——表达器不倒出整个状态。
"""

from __future__ import annotations

REFUSAL = "I cannot comply because it is not authorized or has been forgotten."


def verbalize(query, memories: list[tuple[str, str]]) -> tuple[str, str | None, dict]:
    """返回 (response, tool_name, parameters)。memories: 已治理的 [(mem_id, content)]。"""
    eb = query.expected_behavior
    if eb.should_refuse:
        parts = [REFUSAL, *eb.must_include]
        return " ".join(p for p in parts if p).strip(), None, {}
    tool = eb.tool_name
    params = dict(eb.parameters)
    param_str = " ".join(str(v) for v in params.values())
    required = " ".join(eb.must_include)
    banned = tuple(x.lower() for x in eb.must_not_include)
    clean_evidence = ""
    for _, c in memories:
        if c and not any(b in c.lower() for b in banned):
            clean_evidence = c
            break
    body = " ".join(p for p in [tool or "", param_str, required, clean_evidence] if p).strip()
    return (body or "OK"), tool, params
