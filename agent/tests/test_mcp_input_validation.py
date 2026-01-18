import pytest


def test_serper_rejects_empty_query_fast() -> None:
    # Import inside test so we don't require API keys for import time.
    from dr_agent.mcp_backend.apis.serper_apis import search_serper

    with pytest.raises(ValueError, match="non-empty"):
        search_serper("   ")


def test_jina_strips_view_source_prefix() -> None:
    from dr_agent.mcp_backend.apis.jina_apis import sanitize_web_url

    assert sanitize_web_url("view-source:https://example.com/a") == "https://example.com/a"
    assert (
        sanitize_web_url("view-source:view-source:https://example.com/a")
        == "https://example.com/a"
    )


def test_jina_rejects_non_http_scheme() -> None:
    from dr_agent.mcp_backend.apis.jina_apis import sanitize_web_url

    with pytest.raises(ValueError, match="Only http/https"):
        sanitize_web_url("javascript:alert(1)")


def test_mcp_serper_tool_wrapper_returns_clean_error_on_empty_query() -> None:
    # The MCP tool wrapper should not throw; it should return a clean structured error.
    from dr_agent.mcp_backend.main import serper_google_webpage_search

    # fastmcp decorates tool functions into a FunctionTool object; call the underlying function.
    out = serper_google_webpage_search.fn("   ")
    assert out["success"] is False
    assert "Missing required parameter" in out["error"]
    assert out["organic"] == []


def test_mcp_jina_tool_wrapper_strips_view_source_then_fails_cleanly_without_key() -> None:
    # We don't need a real API key here; we just want to ensure "view-source:" is unwrapped
    # before the call and that failures are returned as success=False.
    from dr_agent.mcp_backend.main import jina_fetch_webpage_content

    # fastmcp decorates tool functions into a FunctionTool object; call the underlying function.
    out = jina_fetch_webpage_content.fn("view-source:https://example.com/a", timeout=1)
    # URL should be sanitized even if the downstream call fails (e.g., missing API key).
    assert out["url"] == "https://example.com/a"
    assert out["success"] is False
    assert isinstance(out.get("error", ""), str) and out["error"]

