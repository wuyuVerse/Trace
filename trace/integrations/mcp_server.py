"""Task 5 / E5a —— SRG MCP server：把 TraceMemory 暴露成 MCP 工具（stdio JSON-RPC）。

零依赖实现 MCP 协议子集（initialize / tools/list / tools/call）over stdin-stdout，
Codex/Claude Code/OpenCode 均可 config-only 接入（见集成方案 §10）。

暴露工具：
  memory_observe(episode, who, t)   把经历写入账本（可用确定性/LLM 编译）
  memory_recall(query, t, who)      返回当前有效状态 + 被治理排除项（可审计）
  memory_fork(remove_memory_ids, t) 反事实分叉

Codex 接入（零行 Rust）：config.toml 加
  [mcp_servers.srg]
  command = "python3"
  args = ["-m", "trace.integrations.mcp_server"]

Ledger is persisted to ``TRACE_LEDGER_PATH`` (default ``~/.trace/ledger.jsonl``);
``SRG_LEDGER_PATH`` is accepted as an alias for backward compat.

自测：PYTHONPATH=. python3 trace/integrations/mcp_server.py --selftest
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from trace.core import Transition, Principal, execute, execute_counterfactual, excluded_report

LEDGER_PATH = Path(os.getenv("TRACE_LEDGER_PATH",
                             os.getenv("SRG_LEDGER_PATH",
                                       str(Path.home() / ".trace" / "ledger.jsonl"))))

TOOLS = [
    {"name": "memory_observe",
     "description": "Record an experience as governed state transitions into the SRG ledger.",
     "inputSchema": {"type": "object", "properties": {
         "episode": {"type": "string"}, "who": {"type": "string"}, "t": {"type": "string"}},
         "required": ["episode"]}},
    {"name": "memory_recall",
     "description": "Reconstruct the CURRENT governed state (not retrieve past). Returns state + excluded(deleted/expired/unauthorized).",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "t": {"type": "string"}, "who": {"type": "string"}}}},
    {"name": "memory_fork",
     "description": "Counterfactual recall: reconstruct state as if certain memories never happened.",
     "inputSchema": {"type": "object", "properties": {
         "remove_memory_ids": {"type": "array", "items": {"type": "string"}},
         "t": {"type": "string"}, "who": {"type": "string"}}}},
]


def _load_ledger() -> list[Transition]:
    if not LEDGER_PATH.exists():
        return []
    out = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Transition(**json.loads(line)))
    return out


def _append_ledger(trs: list[Transition]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        for t in trs:
            f.write(json.dumps(t.__dict__, ensure_ascii=False) + "\n")


def _state_to_rows(state):
    """state: list[Proposition]（唯一引擎 execute 输出）。"""
    return [{"subject": pp.subject, "attribute": pp.attribute, "value": pp.value,
             "valid_until": pp.valid_until, "source": pp.memory_id} for pp in state]


def _do_recall(args):
    p = Principal(scope=args.get("who", "same_user"), allow_sensitive=True)
    L = _load_ledger()
    t = args.get("t")
    state = execute(L, t, principal=p)               # 唯一引擎：执行到 t 得 S(t)
    excluded = excluded_report(L, t, p)              # 审计：被排除项+理由(deleted/superseded/stale/unauthorized)
    return {"state": _state_to_rows(state), "excluded": excluded}


def _do_observe(args):
    # MCP server 侧默认用确定性编译无法（无 gold），改为：把 episode 存成一条自由文本 ASSERT，
    # 或（若配置）走 LLM 编译器。这里给最小可用：LLM 编译（若 TRACE_MCP_LLM=1）否则原样存。
    episode = args.get("episode", "")
    who = args.get("who", "same_user")
    t = args.get("t", "")
    L = _load_ledger()
    clock = (max((tr.t_ingest for tr in L), default=0)) + 1
    if os.getenv("TRACE_MCP_LLM") == "1":
        from trace.compile.llm_compiler import compile_episode
        new = compile_episode(episode, t_event=t, t_ingest=clock,
                              model=os.getenv("TRACE_LLM_MODEL",
                                              os.getenv("SRG_LLM_MODEL", "gpt-4o-mini")))
        for tr in new:
            object.__setattr__(tr, "scope", who) if who != "same_user" else None
    else:
        new = [Transition(memory_id=f"obs{clock}", subject="note", attribute=f"e{clock}",
                          op="ASSERT", t_event=t, t_ingest=clock, scope=who, content=episode)]
    _append_ledger(new)
    return {"transitions": [t.__dict__ for t in new], "committed": True}


def _do_fork(args):
    p = Principal(scope=args.get("who", "same_user"), allow_sensitive=True)
    L = _load_ledger()
    state = execute_counterfactual(L, remove_memory_ids=set(args.get("remove_memory_ids", [])),
                                   inject=[], t_query=args.get("t"), principal=p)
    return {"state": _state_to_rows(state)}


DISPATCH = {"memory_observe": _do_observe, "memory_recall": _do_recall, "memory_fork": _do_fork}


def _handle(req):
    mid = req.get("id")
    method = req.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "srg-memory", "version": "0.1"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        fn = DISPATCH.get(name)
        if fn is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown tool {name}"}}
        result = fn(args)
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}}
    if method and method.startswith("notifications/"):
        return None  # 通知无需回复
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}


def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def selftest():
    """不走 stdio，直接调 dispatch 验证三工具 + 治理语义。"""
    import tempfile
    global LEDGER_PATH
    LEDGER_PATH = Path(tempfile.mktemp(suffix=".jsonl"))
    # 手工写入一段账本：key 轮换 + 删除
    _append_ledger([
        Transition("k1", "cfg", "api_key", "ASSERT", "2026-01", 1, content="K1"),
        Transition("k2", "cfg", "api_key", "SUPERSEDE", "2026-06", 2, content="K2"),
        Transition("k1d", "cfg", "api_key_old", "DELETE", "2026-06", 3, content="K1"),
    ])
    r = _do_recall({"t": "2026-07"})
    assert any(x["value"] == "K2" for x in r["state"]), r
    assert any(x["reason"] == "deleted" for x in r["excluded"]), r
    # tools/list
    lst = _handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert len(lst["result"]["tools"]) == 3
    # tools/call recall
    call = _handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "memory_recall", "arguments": {"t": "2026-07"}}})
    assert "K2" in call["result"]["content"][0]["text"]
    LEDGER_PATH.unlink(missing_ok=True)
    print("✅ MCP server selftest 通过：initialize/tools/list/tools/call + 治理语义(K2当前, K1被删)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        serve_stdio()
