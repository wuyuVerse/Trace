"""Minimal OpenAI-compatible chat client (stdlib-only, no third-party deps).

Reads endpoint / key / model from env vars so the same code works against any
OpenAI-compatible server (OpenAI, Azure, vLLM, Ollama, TGI, an in-house proxy).

Environment (TRACE_* takes priority; SRG_* aliases kept for backward compat):

    TRACE_LLM_BASE_URL / SRG_LLM_BASE_URL   default: https://api.openai.com/v1
    TRACE_LLM_API_KEY  / SRG_LLM_API_KEY    default: $OPENAI_API_KEY
    TRACE_LLM_MODEL    / SRG_LLM_MODEL      default: gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


BASE_URL = _env("TRACE_LLM_BASE_URL", "SRG_LLM_BASE_URL",
                default="https://api.openai.com/v1")
API_KEY = _env("TRACE_LLM_API_KEY", "SRG_LLM_API_KEY", "OPENAI_API_KEY",
               default="")
DEFAULT_MODEL = _env("TRACE_LLM_MODEL", "SRG_LLM_MODEL",
                     default="gpt-4o-mini")


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    retries: int = 5,
    timeout: float = 240.0,
) -> str:
    """One chat.completions round-trip; retries with exponential backoff.

    Returns the assistant message ``content``. If ``content`` is empty and a
    reasoning-style ``reasoning_content`` field is present (some providers emit
    the answer there when ``max_tokens`` is tight), that string is returned.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            time.sleep(min(30.0, 3.0 * (2 ** attempt)))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def extract_json(text: str):
    """Extract the first JSON object/array from ``text`` (tolerates ``` fences)."""
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                s = p
                break
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        i = s.find(open_ch)
        if i == -1:
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == open_ch:
                depth += 1
            elif s[j] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j + 1])
                    except json.JSONDecodeError:
                        break
    return None


if __name__ == "__main__":
    print("model:", DEFAULT_MODEL)
    print("reply:", chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16))
