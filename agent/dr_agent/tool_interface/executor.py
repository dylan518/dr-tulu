import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseTool
from .data_types import ToolOutput
from .tool_parsers import ToolCallInfo


def _stable_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(obj), ensure_ascii=False)


def _fresh_call_id() -> str:
    return str(uuid.uuid4())[:8]


@dataclass(frozen=True)
class ToolRequest:
    tool: BaseTool
    tool_input: Any
    call_info: Optional[ToolCallInfo] = None
    cache_key: Optional[str] = None

    def key(self) -> str:
        if self.cache_key is not None:
            return self.cache_key

        if self.call_info is not None:
            payload = {
                "content": self.call_info.content,
                "parameters": self.call_info.parameters or {},
            }
        else:
            payload = {"tool_input": self.tool_input}

        return f"{self.tool.name}:{_stable_json_dumps(payload)}"


class AsyncToolExecutor:
    """
    Execute tool calls with high concurrency + in-flight de-duplication + optional caching.

    This is intentionally NOT rate-limited here; concurrency is controlled by the caller.
    """

    def __init__(self, enable_cache: bool = True, enable_inflight_dedupe: bool = True):
        self._enable_cache = enable_cache
        self._enable_inflight_dedupe = enable_inflight_dedupe
        self._cache: Dict[str, ToolOutput] = {}
        self._inflight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    def _copy_for_reuse(self, out: ToolOutput) -> ToolOutput:
        # ToolOutput is a Pydantic model; use model_copy when available.
        if hasattr(out, "model_copy"):
            return out.model_copy(update={"call_id": _fresh_call_id(), "runtime": 0.0})
        # Fallback (older pydantic): construct a new object via dict.
        d = out.dict()
        d["call_id"] = _fresh_call_id()
        d["runtime"] = 0.0
        return ToolOutput(**d)

    async def execute(self, req: ToolRequest) -> ToolOutput:
        key = req.key()

        if self._enable_cache:
            cached = self._cache.get(key)
            if cached is not None:
                return self._copy_for_reuse(cached)

        if not self._enable_inflight_dedupe:
            return await req.tool(req.tool_input)

        async with self._lock:
            if self._enable_cache:
                cached = self._cache.get(key)
                if cached is not None:
                    return self._copy_for_reuse(cached)

            fut = self._inflight.get(key)
            if fut is None:
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = fut
                creator = True
            else:
                creator = False

        if not creator:
            out = await fut
            # If the shared future carried an exception, raise it to the caller.
            if isinstance(out, Exception):
                raise out
            if isinstance(out, ToolOutput) and self._enable_cache:
                return self._copy_for_reuse(out)
            return out

        try:
            out = await req.tool(req.tool_input)
        except Exception as e:
            out = e

        async with self._lock:
            fut = self._inflight.pop(key, None)
            if fut is not None and not fut.done():
                fut.set_result(out)
            if self._enable_cache and isinstance(out, ToolOutput):
                self._cache[key] = out

        if isinstance(out, Exception):
            raise out
        return out


