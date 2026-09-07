# TRACE Integrations (E5a: zero-source-change agent hookup)

Attach TRACE memory as a **side-car service** to coding agents (Codex / OpenCode /
Claude Code). Three contracts: no agent-source edits; not on the critical path
(failures degrade silently); no agent-specific assumptions. Two complementary
modes: **active** (MCP tools, the model calls them) and **passive** (inject the
governed state each turn as guardrail).

## Components

| File | Purpose | Verification |
|---|---|---|
| `mcp_server.py` | MCP stdio server exposing `memory_observe` / `memory_recall` / `memory_fork` | `--selftest` + end-to-end stdio |
| `codex_hook.py` | Codex `UserPromptSubmit` hook injecting the governed state | `--selftest` |
| `opencode_plugin.ts` | OpenCode plugin (custom tools + `chat.system.transform`) | `python -m` bridge tested |
| `amb_solver.py` | AutoMemoryBench evaluation adapter (TRACE vs. RAG A/B) | integration test |

## Codex (config-only)

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.trace]                     # active: model can call memory_recall/observe/fork
command = "python3"
args = ["-m", "trace.integrations.mcp_server"]

[hooks]                                 # passive: inject governed state every turn
user_prompt_submit = { command = "python3", args = ["-m", "trace.integrations.codex_hook"] }
```

Environment (all optional): `TRACE_LEDGER_PATH` for a persistent ledger location,
`TRACE_LLM_BASE_URL` / `TRACE_LLM_API_KEY` / `TRACE_LLM_MODEL` for the LLM
compiler back-end.

## OpenCode (no core fork)

In `opencode.json`:

```json
{ "plugin": ["node_modules/trace/integrations/opencode_plugin.ts"] }
```

The plugin registers active tools and hooks `chat.system.transform` to inject
the current governed state each turn.

## Claude Code

Same `mcp_server.py` — Claude Code speaks MCP natively. Add a `trace` server to
`.mcp.json` in the same shape as Codex.

## Local smoke tests

```bash
pip install -e .
python3 -m trace.integrations.mcp_server --selftest
python3 -m trace.integrations.codex_hook --selftest
# stdio end-to-end:
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n' | python3 -m trace.integrations.mcp_server
```
