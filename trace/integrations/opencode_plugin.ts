// OpenCode plugin — expose TRACE memory to an OpenCode-based agent (no core fork).
//
// Hooks (see packages/plugin/src/index.ts:225/290):
//   1) tool: memory_recall / memory_observe — custom tools (active mode, model decides)
//   2) experimental.chat.system.transform  — inject governed state each turn (passive mode)
//
// TRACE side reuses the Python MCP server (subprocess call, zero core fork).
//
// Install: after `pip install -e .`, add to opencode.json:
//   "plugin": ["node_modules/trace/integrations/opencode_plugin.ts"]
// or point to the file directly. Uses `python3 -m trace.integrations.mcp_server`.

import { execFileSync } from "node:child_process";

const REPO = process.env.TRACE_ROOT || process.cwd();

// 调 TRACE MCP server 的单个工具（一次性 stdio 请求）。失败静默降级（不进关键路径）。
function traceCall(tool: string, args: Record<string, unknown>): any {
  try {
    const req =
      JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call",
        params: { name: tool, arguments: args } }) + "\n";
    const out = execFileSync("python3",
      ["-m", "trace.integrations.mcp_server"],
      { cwd: REPO, input: req, env: { ...process.env }, timeout: 15000 });
    const line = out.toString().trim().split("\n").pop() || "{}";
    const resp = JSON.parse(line);
    return JSON.parse(resp?.result?.content?.[0]?.text || "{}");
  } catch {
    return null; // 降级：TRACE 不可用时 agent 照常运行
  }
}

function renderState(t?: string): string {
  const r = traceCall("memory_recall", { t });
  if (!r || (!r.state?.length && !r.excluded?.length)) return "";
  const lines = ["[TRACE current governed state]"];
  for (const x of r.state || [])
    lines.push(`- ${x.subject}.${x.attribute} = ${x.value}`
      + (x.valid_until ? ` (until ${x.valid_until})` : ""));
  if (r.excluded?.length)
    lines.push("[excluded by governance] "
      + r.excluded.map((x: any) => `${x.subject}.${x.attribute}(${x.reason})`).join(", "));
  return lines.join("\n");
}

export default async function TraceMemoryPlugin() {
  return {
    // 主动模式：自定义工具
    tool: {
      memory_recall: {
        description: "Reconstruct CURRENT governed memory state (not retrieve past).",
        parameters: { type: "object", properties: { t: { type: "string" } } },
        execute: async (args: any) => JSON.stringify(traceCall("memory_recall", args || {})),
      },
      memory_observe: {
        description: "Record an experience into the TRACE governed ledger.",
        parameters: { type: "object",
          properties: { episode: { type: "string" }, t: { type: "string" } },
          required: ["episode"] },
        execute: async (args: any) => JSON.stringify(traceCall("memory_observe", args || {})),
      },
    },
    // 被动模式：每轮把治理态注入 system prompt（防错兜底）
    experimental: {
      chat: {
        system: {
          transform: async (output: { system: string[] }) => {
            const ctx = renderState();
            if (ctx) output.system.push(ctx);
            return output;
          },
        },
      },
    },
  };
}
