"""TRACE — Recall-as-Execution: a replayable, bitemporal ledger memory.

Top-level entry points::

    from trace import TraceMemory                              # protocol-agnostic facade
    from trace.core import execute, Transition, Principal      # the sole engine (recall == execute)
    from trace.integrations.amb_solver import solve            # benchmark adapter

Layers: ``core`` (the deterministic ledger algebra + state VM) · ``compile``
(episode -> transitions) · ``verbalize`` (state -> text) · ``runtime``
(LLM client) · ``integrations`` (AMB/MCP/agent adapters) · ``experiments`` · ``theory``.
"""

from trace.memory import TraceMemory

__all__ = ["TraceMemory"]
__version__ = "0.1.0"
