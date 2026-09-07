"""Codex UserPromptSubmit hook —— 每轮把 SRG 当前治理态注入上下文（零行 Rust）。

Codex 每轮读 hook 的 stdout JSON 的 `additional_context` 字段拼进上下文
（证据 hooks/src/events/user_prompt_submit.rs:37 + session/turn.rs:274 每轮刷新）。
本脚本调 SRG 的 recall，把「当前有效状态 + 被治理排除项」渲染成紧凑文本注入——
这是**被动模式**：模型无感，用于防错（不能指望模型主动想起别用旧 key）。

Codex config.toml (after ``pip install -e .``)::

  [hooks]
  user_prompt_submit = { command = "python3", args = ["-m", "trace.integrations.codex_hook"] }

自测：PYTHONPATH=. python3 trace/integrations/codex_hook.py --selftest
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from trace.integrations.mcp_server import _do_recall  # noqa: E402


def render(t: str | None = None, who: str = "same_user") -> str:
    r = _do_recall({"t": t, "who": who})
    if not r["state"] and not r["excluded"]:
        return ""
    lines = ["[TRACE current governed state]"]
    for x in r["state"]:
        lines.append(f"- {x['subject']}.{x['attribute']} = {x['value']}"
                     + (f" (until {x['valid_until']})" if x.get("valid_until") else ""))
    if r["excluded"]:
        ex = ", ".join(f"{x['subject']}.{x['attribute']}({x['reason']})" for x in r["excluded"])
        lines.append(f"[excluded by governance] {ex}")
    return "\n".join(lines)


def main():
    # Codex 通过 stdin 传 hook 事件 JSON；我们只需读出可选的 who/t，然后回 additional_context
    who, t = "same_user", None
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        who = payload.get("who", who)
        t = payload.get("t", t)
    except (json.JSONDecodeError, ValueError):
        pass
    ctx = render(t, who)
    print(json.dumps({"additional_context": ctx}, ensure_ascii=False))


def selftest():
    import tempfile
    import trace.integrations.mcp_server as srv
    from trace.core import Transition
    srv.LEDGER_PATH = Path(tempfile.mktemp(suffix=".jsonl"))
    srv._append_ledger([
        Transition("k1", "cfg", "api_key", "ASSERT", "2026-01", 1, content="K1"),
        Transition("k2", "cfg", "api_key", "SUPERSEDE", "2026-06", 2, content="K2"),
        Transition("k1d", "cfg", "api_key_old", "DELETE", "2026-06", 3, content="K1"),
    ])
    txt = render("2026-07")
    assert "K2" in txt and "deleted" in txt, txt
    srv.LEDGER_PATH.unlink(missing_ok=True)
    print("✅ codex_hook selftest 通过，注入文本：")
    print(txt)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
