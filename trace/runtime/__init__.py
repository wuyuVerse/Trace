"""SRG 运行时基础设施（LLM 客户端等）。"""
from trace.runtime.llm_client import chat, extract_json, DEFAULT_MODEL

__all__ = ["chat", "extract_json", "DEFAULT_MODEL"]
