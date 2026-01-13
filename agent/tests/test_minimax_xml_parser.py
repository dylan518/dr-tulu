import pytest


from dr_agent.tool_interface.tool_parsers import create_tool_parser


def test_minimax_xml_has_calls_and_parse_call_query_and_params():
    parser = create_tool_parser("minimax_xml")

    text = """
Some reasoning...
<minimax:tool_call>
  <invoke name="google_search">
    <parameter name="query">qwen3 moe architecture</parameter>
    <parameter name="num_results">5</parameter>
    <parameter name="gl">us</parameter>
  </invoke>
</minimax:tool_call>
More text...
""".strip()

    assert parser.has_calls(text, "google_search")

    call = parser.parse_call(text, "google_search")
    assert call is not None
    assert call.content == "qwen3 moe architecture"
    # query becomes content; the rest remain parameters
    assert call.parameters == {"num_results": "5", "gl": "us"}
    assert call.start_pos >= 0
    assert call.end_pos > call.start_pos


def test_minimax_xml_parse_call_url_style_content_key():
    parser = create_tool_parser("minimax_xml")

    text = """
<minimax:tool_call>
  <invoke name="browse_webpage">
    <parameter name="webpage_url">https://example.com/paper</parameter>
    <parameter name="include_markdown">true</parameter>
  </invoke>
</minimax:tool_call>
""".strip()

    call = parser.parse_call(text, "browse_webpage")
    assert call is not None
    assert call.content == "https://example.com/paper"
    assert call.parameters == {"include_markdown": "true"}


def test_minimax_xml_no_match_wrong_tool_name():
    parser = create_tool_parser("minimax_xml")

    text = """
<minimax:tool_call>
  <invoke name="google_search">
    <parameter name="query">hello</parameter>
  </invoke>
</minimax:tool_call>
""".strip()

    assert not parser.has_calls(text, "snippet_search")
    assert parser.parse_call(text, "snippet_search") is None




