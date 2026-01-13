import types

import pytest

import dr_agent.client as client_mod


class _Msg:
    def __init__(self, content=None, reasoning_content=None, provider_specific_fields=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.provider_specific_fields = provider_specific_fields or {}


class _Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, message, finish_reason="stop"):
        self.choices = [_Choice(message=message, finish_reason=finish_reason)]


@pytest.mark.asyncio
async def test_commercial_api_uses_reasoning_content_as_content_when_content_empty(monkeypatch):
    async def fake_acompletion(**kwargs):
        msg = _Msg(content=None, reasoning_content="FINAL ANSWER")
        return _Resp(msg, finish_reason="stop")

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion)

    c = client_mod.LLMToolClient(model_name="openai/gpt-oss-120b", base_url="http://x", api_key="k")
    out = await c._generate_single_response_commercial_api(
        messages=[{"role": "user", "content": "hi"}],
        include_reasoning=True,
        stop_sequences=None,
        verbose=False,
    )
    assert out.strip() == "FINAL ANSWER"


@pytest.mark.asyncio
async def test_native_tool_calling_stores_reasoning_content_in_assistant_message(monkeypatch):
    async def fake_call_litellm_with_tools(self, messages, tools, **kwargs):
        msg = _Msg(content=None, reasoning_content="FINAL ANSWER")
        # Add minimal tool_calls attr so getattr works
        msg.tool_calls = None
        return _Resp(msg, finish_reason="stop")

    monkeypatch.setattr(client_mod.LLMToolClient, "_call_litellm_with_tools", fake_call_litellm_with_tools)

    c = client_mod.LLMToolClient(
        model_name="openai/gpt-oss-120b",
        base_url="http://x",
        api_key="k",
        tools=[],
    )
    res = await c.generate_with_tools(
        prompt_or_messages=[{"role": "user", "content": "hi"}],
        tool_calling_mode="native",
        max_tool_calls=1,
        max_tokens=256,
        include_tool_results=True,
        verbose=False,
    )
    assert "FINAL ANSWER" in (res.generated_text or "")


@pytest.mark.asyncio
async def test_trace_model_input_includes_messages_when_enabled(monkeypatch):
    async def fake_call_litellm_with_tools(self, messages, tools, **kwargs):
        msg = _Msg(content="OK", reasoning_content=None)
        msg.tool_calls = None
        return _Resp(msg, finish_reason="stop")

    monkeypatch.setattr(client_mod.LLMToolClient, "_call_litellm_with_tools", fake_call_litellm_with_tools)
    monkeypatch.setenv("DR_TRACE_MODEL_INPUT", "1")

    c = client_mod.LLMToolClient(
        model_name="openai/gpt-oss-120b",
        base_url="http://x",
        api_key="k",
        tools=[],
    )
    res = await c.generate_with_tools(
        prompt_or_messages=[{"role": "user", "content": "hi"}],
        tool_calling_mode="native",
        max_tool_calls=1,
        max_tokens=64,
        include_tool_results=True,
        verbose=False,
    )
    assert isinstance(res.model_input, dict)
    assert "messages" in res.model_input


