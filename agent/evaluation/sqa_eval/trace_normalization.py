from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class NormalizedTrace:
    """
    Canonical view of "where to find snippet blocks" regardless of backend.

    We only care about text blobs that may contain:
      <snippets id="..."> ... </snippets>
      <snippet id="..."> ... </snippet>
      <webpage id="..."> ... </webpage>
    """

    text_blobs: List[str]


def _walk(obj: Any) -> Iterable[Any]:
    """Yield all nested values from dict/list trees."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield v
            yield from _walk(v)


def _collect_string_fields(obj: Any, keys: set[str]) -> List[str]:
    out: List[str] = []
    if not isinstance(obj, dict):
        return out
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def normalize_full_traces(full_traces: Any) -> NormalizedTrace:
    """
    Normalize multiple possible trace formats into a list of text blobs to scan for snippet tags.

    Supported (best-effort):
    - OSS workflow traces: {"generated_text": "...", "tool_calls": [{"generated_text": "..."}]}
    - OpenAI-style responses (via model_dump): {"choices":[{"message":{"content": "...", "tool_calls": ...}}], ...}
    - Generic message logs: {"messages":[{"role":"tool","content": ...}, ...], ...}
    """
    blobs: List[str] = []

    if not full_traces:
        return NormalizedTrace(text_blobs=blobs)

    # 1) OSS workflow style (strongly typed)
    if isinstance(full_traces, dict):
        blobs.extend(_collect_string_fields(full_traces, {"generated_text"}))
        tool_calls = full_traces.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    # Common OSS keys
                    blobs.extend(_collect_string_fields(tc, {"generated_text", "tool_output", "output"}))

        # 2) OpenAI style: choices[].message.content and any tool messages
        choices = full_traces.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message")
                if isinstance(msg, dict):
                    blobs.extend(_collect_string_fields(msg, {"content"}))
                    # tool call arguments sometimes include raw snippets
                    tcalls = msg.get("tool_calls")
                    if isinstance(tcalls, list):
                        for tc in tcalls:
                            if isinstance(tc, dict):
                                # OpenAI tool_call schema: {function:{arguments,name}}
                                func = tc.get("function")
                                if isinstance(func, dict):
                                    blobs.extend(_collect_string_fields(func, {"arguments"}))

        # 3) Generic message logs
        messages = full_traces.get("messages")
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    blobs.extend(_collect_string_fields(m, {"content", "text"}))

    # 4) Fallback: scan the entire trace tree for any string fields that look relevant.
    #    This is intentionally broad to handle MiniMax / other vendors that log differently.
    for node in _walk(full_traces):
        if isinstance(node, dict):
            blobs.extend(_collect_string_fields(node, {"generated_text", "content", "text", "tool_output", "output"}))

    # de-dupe while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for b in blobs:
        key = b.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(b)

    return NormalizedTrace(text_blobs=deduped)




