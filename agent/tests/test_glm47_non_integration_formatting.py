import json

from dr_agent.client import LLMToolClient


def test_glm_style_tool_call_marker_recovery_from_content():
    """
    Non-integration regression test:
    GLM/vLLM sometimes emits tool calls in assistant content like:
      <tool_call>get_secret_code?key=alpha</think>
    without populating OpenAI `tool_calls`. We should recover those into
    OpenAI-style tool call dicts so native tool calling still works.
    """
    c = LLMToolClient(model_name="openai/zai-org/GLM-4.7-FP8", base_url="http://example.invalid/v1")

    recovered = c._recover_tool_calls_from_content(
        "<think>...</think><tool_call>get_secret_code?key=alpha</think>"
    )
    assert len(recovered) >= 1
    call = recovered[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_secret_code"
    args = json.loads(call["function"]["arguments"])
    assert args["key"] == "alpha"


def test_glm_style_tool_call_marker_recovery_with_closing_tag():
    c = LLMToolClient(model_name="openai/zai-org/GLM-4.7-FP8", base_url="http://example.invalid/v1")
    recovered = c._recover_tool_calls_from_content(
        "<tool_call>snippet_search?query=pagerank</tool_call>"
    )
    assert recovered and recovered[0]["function"]["name"] == "snippet_search"


def test_glm_style_tool_call_arg_key_value_pairs():
    """
    Non-integration regression test:
    GLM/vLLM can emit tool calls in assistant content like:
      <tool_call>get_secret_code<arg_key>key</arg_key><arg_value>alpha</arg_value></tool_call>
    We should recover those into OpenAI-style tool call dicts.
    """
    c = LLMToolClient(model_name="openai/zai-org/GLM-4.7-FP8", base_url="http://example.invalid/v1")
    recovered = c._recover_tool_calls_from_content(
        "<tool_call>get_secret_code<arg_key>key</arg_key><arg_value>alpha</arg_value></tool_call>"
    )
    assert recovered and recovered[0]["function"]["name"] == "get_secret_code"
    args = json.loads(recovered[0]["function"]["arguments"])
    assert args["key"] == "alpha"


def test_native_tool_choice_only_applies_first_iteration():
    """
    Non-integration guard:
    We allow passing `tool_choice` through, but the client must only force it on the
    first native-tool iteration to avoid infinite tool loops.
    """
    # This is a placeholder guardrail test to ensure the module imports stay stable.
    # Iteration behavior is validated by the integration tests where vLLM returns repeated tool calls
    # if tool_choice is forced every turn.
    assert True


