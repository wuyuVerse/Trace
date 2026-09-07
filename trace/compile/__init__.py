"""SRG 写入侧（E 部件）：把经历编译成状态转移。

  llm_compiler.py   —— 真实 LLM 编译器（自由文本 → 转移）
  deterministic.py  —— gold → 转移的纯映射（隔离变量 / 上界参照）
  write_gate.py     —— over-memory 抑制
"""

from trace.compile.deterministic import compile_case, contract_state, slot_key, op_for
from trace.compile.llm_compiler import compile_episode, compile_stream
from trace.compile.write_gate import gate, write_stats

__all__ = [
    "compile_case", "contract_state", "slot_key", "op_for",
    "compile_episode", "compile_stream",
    "gate", "write_stats",
]
