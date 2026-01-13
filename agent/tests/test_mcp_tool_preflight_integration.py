import os
import socket
from urllib.parse import urlparse

import dotenv
import pytest

from dr_agent.tool_interface.mcp_tools import JinaBrowseTool, SerperSearchTool


REPO_ROOT = (__import__("pathlib").Path(__file__).parent.parent.parent).resolve()


def _load_env_for_tests() -> None:
    override = os.environ.get("DR_TULU_ENV_FILE") or os.environ.get("DR_AGENT_ENV_FILE")
    if override:
        dotenv.load_dotenv(override)
        return
    dotenv.load_dotenv(REPO_ROOT / ".env")


def _can_connect_url(url: str) -> bool:
    try:
        u = urlparse(url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


@pytest.mark.integration
def test_mcp_server_reachable():
    """
    Preflight: MCP server should be reachable before we attempt tool calls.
    Keep this lightweight (socket check only).
    """
    _load_env_for_tests()
    host = os.environ.get("MCP_TRANSPORT_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_TRANSPORT_PORT", "8000"))
    url = f"http://{host}:{port}/health"
    if not _can_connect_url(url):
        pytest.skip(f"MCP server not reachable at {url}. Start MCP server to run tool preflight.")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_serper_search_tool_works_if_key_configured():
    """
    Preflight: if SERPER_API_KEY is set, verify Serper MCP search works end-to-end.
    This catches missing/invalid keys early and produces a concrete error message.
    """
    _load_env_for_tests()
    if not os.environ.get("SERPER_API_KEY"):
        pytest.skip("SERPER_API_KEY not set; skipping Serper tool preflight.")

    tool = SerperSearchTool(
        tool_parser="v20250824",
        number_documents_to_search=3,
        timeout=15,
        name="snippet_search",
        transport_type=os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport"),
        mcp_port=int(os.environ.get("MCP_TRANSPORT_PORT", "8000")),
        mcp_host=os.environ.get("MCP_TRANSPORT_HOST", "127.0.0.1"),
    )

    out = await tool({"query": "PageRank algorithm", "num_results": 3})
    assert out.error == "", f"Serper tool call failed: {out.error}"
    assert out.documents, "Serper tool returned no documents."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_jina_browse_tool_works_if_key_configured():
    """
    Preflight: if JINA_API_KEY is set, verify Jina Reader MCP browse works.
    """
    _load_env_for_tests()
    if not os.environ.get("JINA_API_KEY"):
        pytest.skip("JINA_API_KEY not set; skipping Jina browse tool preflight.")

    tool = JinaBrowseTool(
        tool_parser="v20250824",
        timeout=20,
        request_timeout=15,
        name="browse_webpage",
        transport_type=os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport"),
        mcp_port=int(os.environ.get("MCP_TRANSPORT_PORT", "8000")),
        mcp_host=os.environ.get("MCP_TRANSPORT_HOST", "127.0.0.1"),
    )

    out = await tool({"url": "https://en.wikipedia.org/wiki/PageRank"})
    assert out.error == "", f"Jina browse tool call failed: {out.error}"
    assert out.documents, "Jina browse returned no documents."




