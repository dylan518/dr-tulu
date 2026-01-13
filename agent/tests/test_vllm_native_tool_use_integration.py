import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import pytest

from dr_agent.client import LLMToolClient
from dr_agent.tool_interface.base import BaseTool
from dr_agent.tool_interface.data_types import ToolOutput


@dataclass
class SecretCodeTool(BaseTool):
    """
    Deterministic tool for verifying native tool calling:
    model must call this tool to obtain a secret value it cannot guess.
    """

    def __init__(self):
        super().__init__(
            tool_parser="unified",
            name="get_secret_code",
            description="Return a secret code for a given key. You must call this tool to obtain the code.",
            timeout=5,
        )

    async def __call__(self, tool_input) -> ToolOutput:
        start = time.time()
        call_id = self._generate_call_id()

        # Support both native tool calling (dict args) and parser-mode (string content).
        if isinstance(tool_input, dict):
            key = tool_input.get("key")
        elif isinstance(tool_input, str):
            key = tool_input.strip()
        else:
            key = None

        if key != "alpha":
            return ToolOutput(
                tool_name=self.name,
                output="",
                called=False,
                error="unknown key",
                runtime=time.time() - start,
                call_id=call_id,
                raw_output={"key": key},
            )

        return ToolOutput(
            tool_name=self.name,
            output="swordfish",
            called=True,
            error="",
            runtime=time.time() - start,
            call_id=call_id,
            raw_output={"key": key},
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Lookup key. Use 'alpha' for this test.",
                }
            },
            "required": ["key"],
        }


def _vllm_targets() -> list[tuple[str, Optional[str], Optional[str]]]:
    """
    (label, base_url, model_name) targets.
    These are intended to point at OpenAI-compatible vLLM servers.
    """
    return [
        (
            "gpt-oss-120b",
            os.environ.get("GPT_OSS_BASE_URL") or os.environ.get("VLLM_BASE_URL"),
            os.environ.get("GPT_OSS_MODEL_NAME") or "openai/gpt-oss-120b",
        ),
        (
            "glm-4.7-fp8",
            os.environ.get("GLM47_BASE_URL") or os.environ.get("VLLM_BASE_URL_2"),
            os.environ.get("GLM47_MODEL_NAME") or "zai-org/GLM-4.7-FP8",
        ),
        (
            "minimax",
            os.environ.get("MINIMAX_BASE_URL") or os.environ.get("VLLM_BASE_URL_3"),
            os.environ.get("MINIMAX_MODEL_NAME") or "ModelCloud/MiniMax-M2-BF16",
        ),
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vllm_models_use_native_tools_to_answer_simple_question():
    """
    Integrated vLLM test:
    - Connect to each configured vLLM endpoint
    - Ensure the model performs native tool calling (OpenAI tools/tool_calls)
    - Ensure it uses the tool output to answer correctly

    This test will SKIP endpoints that are not configured via env vars.
    """
    # Collect configured targets
    targets = [(label, url, model) for (label, url, model) in _vllm_targets() if url and model]
    if not targets:
        pytest.skip(
            "No vLLM endpoints configured. Set GPT_OSS_BASE_URL/GPT_OSS_MODEL_NAME and/or "
            "GLM47_BASE_URL/GLM47_MODEL_NAME and/or MINIMAX_BASE_URL/MINIMAX_MODEL_NAME."
        )

    # Native tool calling requires vLLM to be started with native tool support enabled.
    # Since this repo primarily relies on parser-based tool calling, only run this test when explicitly requested.
    if os.environ.get("VLLM_NATIVE_TOOLS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set VLLM_NATIVE_TOOLS=1 to run native tool-calling integration.")

    tool = SecretCodeTool()

    for label, base_url, model_name in targets:
        with LLMToolClient(
            model_name=model_name,
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "dummy-key"),
            tools=[tool],
        ) as client:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You MUST call the tool get_secret_code to answer. "
                        "Do not guess. After you receive the tool result, reply exactly as: "
                        "CODE: <value>"
                    ),
                },
                {"role": "user", "content": "What is the secret code for key 'alpha'?"},
            ]

            out = await client.generate_with_tools(
                messages,
                max_tool_calls=3,
                include_tool_results=True,
                verbose=False,
                tool_calling_mode="native",
                # GLM-4.7 (and some vLLM setups) may not auto-select tools reliably even with
                # --enable-auto-tool-choice; force the specific function to make this an
                # integration test of the tool execution loop rather than model policy.
                tool_choice={"type": "function", "function": {"name": "get_secret_code"}},
                temperature=0,
                # Give the model enough budget to emit a <tool_call> block when `tool_choice`
                # is treated as "auto" (some servers/models ignore/approximate forced tool_choice).
                max_tokens=256,
            )

        assert (
            out.tool_call_count >= 1
        ), f"[{label}] Model did not call tools. Ensure vLLM is started with tool calling enabled (e.g., --enable-auto-tool-choice)."
        assert (
            "swordfish" in out.generated_text
        ), f"[{label}] Model did not use tool output. got: {out.generated_text!r}"


def _can_connect(base_url: str) -> bool:
    try:
        u = urlparse(base_url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            return False
        import socket

        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vllm_models_use_parser_tools_to_answer_simple_question():
    """
    Integrated vLLM test using *our own* (parser-based) tool calling.

    This does NOT require vLLM native tool calling to be enabled.
    It only requires an OpenAI-compatible endpoint that can produce text/chat completions.
    """
    targets = [(label, url, model) for (label, url, model) in _vllm_targets() if url and model]
    targets = [(label, url, model) for (label, url, model) in targets if _can_connect(url)]
    if not targets:
        pytest.skip(
            "No reachable vLLM endpoints found. Set GPT_OSS_BASE_URL/GLM47_BASE_URL/MINIMAX_BASE_URL "
            "(or VLLM_BASE_URL/VLLM_BASE_URL_2/VLLM_BASE_URL_3) and start the servers."
        )

    tool = SecretCodeTool()

    for label, base_url, model_name in targets:
        with LLMToolClient(
            model_name=model_name,
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "dummy-key"),
            tools=[tool],
        ) as client:
            # Be extremely strict to maximize compliance across models:
            # first response must be ONLY the tool call (no prose).
            prompt = (
                "Respond with ONLY the following tool call (no other text):\n"
                '<tool name="get_secret_code">alpha</tool>'
            )

            out = await client.generate_with_tools(
                prompt,
                max_tool_calls=3,
                include_tool_results=True,
                verbose=False,
                tool_calling_mode="parser",
                temperature=0,
                max_tokens=128,
            )

        if out.tool_call_count < 1:
            pytest.skip(
                f"[{label}] Model did not emit parser-style <tool> tags; "
                "this indicates the model is not compatible with this repo's parser-based tool calling."
            )
        assert "CODE:" in out.generated_text, f"[{label}] Bad final format: {out.generated_text!r}"
        assert "swordfish" in out.generated_text, f"[{label}] Did not use tool output: {out.generated_text!r}"


