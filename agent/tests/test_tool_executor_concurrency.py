import asyncio
import time

import pytest

from dr_agent.client import LLMToolClient
from dr_agent.tool_interface.base import BaseTool
from dr_agent.tool_interface.data_types import ToolOutput
from dr_agent.tool_interface.executor import AsyncToolExecutor, ToolRequest


class _SleepTool(BaseTool):
    def __init__(self, name: str, delay_s: float = 0.15):
        super().__init__(tool_parser="v20250824", name=name, timeout=30)
        self.delay_s = delay_s
        self.calls = 0

    async def __call__(self, tool_input):  # type: ignore[override]
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        content = self.extract_tool_input(tool_input) or ""
        return ToolOutput(
            tool_name=self.name,
            output=f"{self.name}:{content}",
            called=True,
            timeout=False,
            runtime=self.delay_s,
            call_id="",
            raw_output=None,
            error=None,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }


@pytest.mark.asyncio
async def test_async_tool_executor_runs_concurrently():
    t1 = _SleepTool("t1", delay_s=0.2)
    t2 = _SleepTool("t2", delay_s=0.2)
    ex = AsyncToolExecutor(enable_cache=False, enable_inflight_dedupe=False)

    start = time.time()
    outs = await asyncio.gather(
        ex.execute(ToolRequest(tool=t1, tool_input='<call_tool name="t1">a</call_tool>')),
        ex.execute(ToolRequest(tool=t2, tool_input='<call_tool name="t2">b</call_tool>')),
    )
    elapsed = time.time() - start

    assert len(outs) == 2
    # If serial, we'd be near 0.4s; with concurrency we should be near 0.2s.
    assert elapsed < 0.32, f"expected concurrent execution, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_async_tool_executor_inflight_dedupe():
    t = _SleepTool("t", delay_s=0.15)
    ex = AsyncToolExecutor(enable_cache=False, enable_inflight_dedupe=True)
    req = ToolRequest(tool=t, tool_input='<call_tool name="t">same</call_tool>')

    await asyncio.gather(ex.execute(req), ex.execute(req), ex.execute(req))
    assert t.calls == 1


def test_client_extracts_multiple_tool_calls_in_order():
    t1 = _SleepTool("t1")
    t2 = _SleepTool("t2")
    c = LLMToolClient(model_name="dummy", tools=[t1, t2])

    text = (
        '<call_tool name="t2">second</call_tool>\n'
        '<call_tool name="t1">first</call_tool>\n'
        '<call_tool name="t2">third</call_tool>'
    )
    calls = c._extract_all_tool_calls_from_text(text)
    assert [tool.name for (tool, _info, _call_text) in calls] == ["t2", "t1", "t2"]


