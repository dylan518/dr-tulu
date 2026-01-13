import os
from pathlib import Path

import pytest

import dotenv

from dr_agent.tool_interface.tool_parsers import create_tool_parser
from dr_agent.workflow import load_config
from dr_agent.client import LLMToolClient
from dr_agent.dataset_utils.load_dataset import load_dataset


WORKFLOW_DIR = Path(__file__).parent.parent / "workflows"
REPO_ROOT = Path(__file__).parent.parent.parent


def _load_env_for_tests() -> None:
    """
    Load env vars for integration tests without requiring the caller to `source` anything.
    Priority:
      1) DR_TULU_ENV_FILE / DR_AGENT_ENV_FILE
      2) repo root `.env` (../.. / ".env")
    """
    override = os.environ.get("DR_TULU_ENV_FILE") or os.environ.get("DR_AGENT_ENV_FILE")
    if override:
        dotenv.load_dotenv(override)
        return
    dotenv.load_dotenv(REPO_ROOT / ".env")


@pytest.mark.parametrize(
    "config_name, expected_tool_parser",
    [
        ("auto_search_sft-minimax.yaml", "minimax_xml"),
        ("auto_search_sft-gpt-oss.yaml", "v20250824"),
        ("auto_search_sft-glm47.yaml", "v20250824"),
    ],
)
def test_workflow_config_loads_and_parser_exists(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_tool_parser: str,
):
    """
    Offline smoke test:
    - YAML resolves (including ${oc.env:...} interpolation)
    - tool parser is registered and instantiable
    - LLMToolClient can be constructed (no network calls)
    """
    _load_env_for_tests()
    # Ensure any env-interpolated fields resolve to something predictable for the test.
    # (We don't care about correctness of the URLs here, just that config loading works.)
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MINIMAX_API_KEY", "dummy")
    monkeypatch.setenv("MINIMAX_MODEL_NAME", "minimax-m2.1")

    monkeypatch.setenv("GPT_OSS_BASE_URL", "http://localhost:30001/v1")
    monkeypatch.setenv("GPT_OSS_API_KEY", "dummy")
    monkeypatch.setenv("GPT_OSS_MODEL_NAME", "gpt-oss")

    monkeypatch.setenv("GLM47_BASE_URL", "http://localhost:30002/v1")
    monkeypatch.setenv("GLM47_API_KEY", "dummy")
    monkeypatch.setenv("GLM47_MODEL_NAME", "glm-4.7")

    cfg_path = WORKFLOW_DIR / config_name
    assert cfg_path.exists(), f"Missing workflow config: {cfg_path}"

    cfg = load_config(str(cfg_path))
    assert cfg["tool_parser"] == expected_tool_parser

    # Verify parser registration
    parser = create_tool_parser(cfg["tool_parser"])
    assert parser is not None

    # Verify client constructability
    with LLMToolClient(
        model_name=cfg.get("search_agent_model_name", "dummy-model"),
        tokenizer_name=cfg.get("search_agent_tokenizer_name"),
        base_url=cfg.get("search_agent_base_url"),
        api_key=cfg.get("search_agent_api_key"),
        tools=[],
    ) as client:
        assert client.model_name


def test_scholarqa_cs2_loader_from_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "scholarqa_cs2_sample.jsonl"
    assert fixture_path.exists()

    examples = load_dataset(
        {
            "name": "scholarqa_cs2",
            "num_examples": 2,
            "local_path": str(fixture_path),
        }
    )
    assert len(examples) == 2
    assert all("id" in ex and "problem" in ex for ex in examples)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gpt_oss_chat_completion_if_configured(monkeypatch: pytest.MonkeyPatch):
    """
    Online integration smoke test (skipped unless env vars are set):
    validates we can get a single chat-completion response.
    """
    _load_env_for_tests()
    base_url = os.environ.get("GPT_OSS_BASE_URL")
    model = os.environ.get("GPT_OSS_MODEL_NAME")
    api_key = os.environ.get("GPT_OSS_API_KEY", "dummy")
    if not base_url or not model:
        pytest.skip("Set GPT_OSS_BASE_URL and GPT_OSS_MODEL_NAME to run this integration test.")

    with LLMToolClient(model_name=model, base_url=base_url, api_key=api_key) as client:
        out = await client._generate_single_response_commercial_api(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            temperature=0,
        )
    assert "OK" in out


@pytest.mark.integration
@pytest.mark.asyncio
async def test_glm47_chat_completion_if_configured(monkeypatch: pytest.MonkeyPatch):
    _load_env_for_tests()
    base_url = os.environ.get("GLM47_BASE_URL")
    model = os.environ.get("GLM47_MODEL_NAME")
    api_key = os.environ.get("GLM47_API_KEY", "dummy")
    if not base_url or not model:
        pytest.skip("Set GLM47_BASE_URL and GLM47_MODEL_NAME to run this integration test.")

    with LLMToolClient(model_name=model, base_url=base_url, api_key=api_key) as client:
        out = await client._generate_single_response_commercial_api(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            temperature=0,
        )
    assert "OK" in out


@pytest.mark.integration
def test_huggingface_token_if_configured():
    """
    Online integration smoke test (skipped unless token is set):
    validates HF token is usable for authenticated calls (useful for pulling gated models/datasets).
    """
    _load_env_for_tests()
    token = (
        os.environ.get("HF_KEY")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        pytest.skip(
            "Set HF_KEY or HUGGINGFACE_HUB_TOKEN (or HF_TOKEN/HUGGINGFACE_TOKEN) to run this integration test."
        )

    from huggingface_hub import HfApi

    info = HfApi().whoami(token=token)
    assert isinstance(info, dict)
    assert "name" in info or "fullname" in info or "orgs" in info


@pytest.mark.integration
def test_hf_model_repos_accessible_if_token_configured():
    """
    Integration smoke test that does NOT require any model endpoint.
    It verifies that the intended HF model repos exist and are accessible with the configured token.

    This is useful when the desired setup is "HF weights -> local vLLM/SGLang server",
    but the server isn't running yet.
    """
    _load_env_for_tests()
    token = (
        os.environ.get("HF_KEY")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        pytest.skip("No HF token configured.")

    from huggingface_hub import HfApi

    api = HfApi()

    # Requested targets: GPT-OSS 120B and GLM-4.7 FP8
    for repo_id in ["openai/gpt-oss-120b", "zai-org/GLM-4.7-FP8"]:
        info = api.model_info(repo_id=repo_id, token=token)
        assert info is not None
        assert getattr(info, "id", None) == repo_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_minimax_endpoint_if_configured():
    """
    Online integration smoke test (skipped unless env vars are set):
    validates we can get a single chat-completion response from a Minimax-served endpoint.

    Note: this assumes you're serving Minimax weights via an OpenAI-compatible server
    (vLLM/SGLang/etc.). If you're using the vendor API, you'll still need the vendor
    API base_url + key and LiteLLM support for that provider.
    """
    _load_env_for_tests()
    base_url = os.environ.get("MINIMAX_BASE_URL")
    model = os.environ.get("MINIMAX_MODEL_NAME")
    api_key = os.environ.get("MINIMAX_API_KEY", "dummy")
    if not base_url or not model:
        pytest.skip("Set MINIMAX_BASE_URL and MINIMAX_MODEL_NAME to run this integration test.")

    with LLMToolClient(model_name=model, base_url=base_url, api_key=api_key) as client:
        out = await client._generate_single_response_commercial_api(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            temperature=0,
        )
    assert "OK" in out


