import os
import asyncio
import socket
import importlib.util
import time
from pathlib import Path
from urllib.parse import urlparse

import dotenv
import pytest

_AUTO_SEARCH_PATH = (Path(__file__).parent.parent / "workflows" / "auto_search_sft.py").resolve()
_spec = importlib.util.spec_from_file_location("auto_search_sft", _AUTO_SEARCH_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
AutoReasonSearchWorkflow = _mod.AutoReasonSearchWorkflow


REPO_ROOT = Path(__file__).parent.parent.parent
AGENT_DIR = Path(__file__).parent.parent
FIXTURE_SQA = Path(__file__).parent / "fixtures" / "scholarqa_cs2_sample.jsonl"


def _load_env_for_tests() -> None:
    override = os.environ.get("DR_TULU_ENV_FILE") or os.environ.get("DR_AGENT_ENV_FILE")
    if override:
        dotenv.load_dotenv(override)
        return
    dotenv.load_dotenv(REPO_ROOT / ".env")


def _can_connect(base_url: str) -> bool:
    try:
        u = urlparse(base_url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


def _wait_for_connect(base_url: str, timeout_s: int) -> bool:
    """Wait for an endpoint to become reachable (useful while vLLM is still loading)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _can_connect(base_url):
            return True
        time.sleep(2)
    return False


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_relpath",
    [
        "workflows/auto_search_sft-gpt-oss.yaml",
        "workflows/auto_search_sft-glm47.yaml",
        "workflows/auto_search_sft-minimax.yaml",
    ],
)
async def test_deep_research_workflow_runs_with_tools(config_relpath: str):
    """
    End-to-end deep research integration test.

    Requirements:
    - MCP server running (default port 8000; your env already runs it)
    - A reachable OpenAI-compatible vLLM endpoint for the configured workflow
      (defaults are 127.0.0.1:30001/30002/30003 in the YAMLs)
    - Tool keys present in env (SERPER/JINA/S2) so snippet_search/browse work

    The test:
    - runs the workflow on 1 ScholarQA-CS2-style question
    - asserts we made at least one search tool call (searched_links non-empty)
    - asserts a non-empty final response was produced
    """
    _load_env_for_tests()
    assert FIXTURE_SQA.exists()

    # Ensure dataset path is set for the loader (even though this test directly passes a prompt,
    # downstream code may rely on dataset_name for instruction selection).
    os.environ.setdefault("SCHOLARQA_CS2_PATH", str(FIXTURE_SQA))

    cfg_path = (AGENT_DIR / config_relpath).resolve()
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    # Load workflow but skip interactive service checks (tests are non-interactive).
    workflow = AutoReasonSearchWorkflow(configuration=str(cfg_path), skip_service_check=True)

    # If the configured endpoint isn't reachable, skip with guidance.
    base_url = getattr(workflow.configuration, "search_agent_base_url", None)
    # Keep this short by default; the intended workflow is "start vLLM first, then run tests".
    wait_s = int(os.environ.get("VLLM_STARTUP_WAIT_S", "30"))
    if not base_url or not _wait_for_connect(base_url, timeout_s=wait_s):
        pytest.skip(
            f"vLLM endpoint not reachable for {cfg_path.name}. "
            f"Expected an OpenAI-compatible server at: {base_url!r}"
        )

    # Pick a "deep research" style question from the fixture (paper/web search style).
    question = (
        "Give a short, well-cited explanation of what PageRank is and why it works. "
        "Use web/paper search tools as needed."
    )

    # Hard cap so this test never "stalls" indefinitely.
    # Local vLLM on large models + tool calling can be slow; keep a generous default.
    timeout_s = int(os.environ.get("DEEP_RESEARCH_TIMEOUT_S", "300"))
    try:
        result = await asyncio.wait_for(
            workflow(problem=question, dataset_name="scholarqa_cs2", verbose=False),
            timeout=timeout_s,
        )
    except TimeoutError:
        pytest.fail(
            f"Deep research workflow timed out after {timeout_s}s. "
            "Increase DEEP_RESEARCH_TIMEOUT_S or reduce search/browse timeouts."
        )

    assert isinstance(result, dict)
    assert result.get("final_response"), "Expected a non-empty final_response"

    # For native tool calling, searched_links can be empty (e.g., some MCP tools don't return URLs
    # or we didn't collect them). Use total_tool_calls as the ground-truth indicator that tools ran.
    searched_links = result.get("searched_links") or []
    total_tool_calls = int(result.get("total_tool_calls") or 0)
    if len(searched_links) == 0 and total_tool_calls == 0:
        pytest.fail(
            "Deep research did not invoke any tools (searched_links empty and total_tool_calls=0). "
            "This usually indicates tool calling is misconfigured OR required API keys are missing. "
            "If you're using native tool calling on GLM/vLLM, ensure GLM47_TOOL_CALLING_MODE=native and "
            "that your search/browse tools are configured with valid keys."
        )


