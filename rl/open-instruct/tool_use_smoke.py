#!/usr/bin/env python3
"""
One-off smoke test for MCP tool-use formatting/execution with a given model.

Why this exists:
- vLLM uses multiprocessing (spawn) and cannot run reliably from stdin (`python - <<PY ...`),
  so we keep this as a real file entrypoint.

Example:
  cd rl/open-instruct
  OPEN_INSTRUCT_MCP_NORMALIZE_TOOL_CALLS=0 uv run --extra compile python tool_use_smoke.py \
    --model rl-research/DR-Tulu-SFT-8B \
    --prompt "Find me similar models to Citroën Ami in Turkey."
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import List, Tuple

import yaml
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.grpo_fast import launch_mcp_subprocess
from open_instruct.search_utils.mcp_tools import MCPTool
from open_instruct.tool_utils.tool_vllm import ToolUseLLM


def _redact_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "<think>[REDACTED]</think>", text, flags=re.DOTALL)


def _extract_tool_calls(text: str) -> List[Tuple[str, str, str]]:
    calls: List[Tuple[str, str, str]] = []
    for m in re.finditer(
        r'<call_tool\s+name="([^"]+)"[^>]*>(.*?)</call_tool>',
        text,
        flags=re.DOTALL,
    ):
        calls.append(("call_tool", m.group(1), m.group(2).strip()))
    for m in re.finditer(
        r'<tool\s+name="([^"]+)"[^>]*>(.*?)</tool>',
        text,
        flags=re.DOTALL,
    ):
        calls.append(("tool", m.group(1), m.group(2).strip()))
    return calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument(
        "--system_prompt_yaml",
        default="open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907.yaml",
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mcp_port", type=int, default=int(os.environ.get("MCP_TRANSPORT_PORT", "8003")))
    ap.add_argument("--mcp_host", default=os.environ.get("MCP_TRANSPORT_HOST", "0.0.0.0"))
    ap.add_argument("--mcp_path", default="/mcp")
    ap.add_argument("--mcp_parser_name", default="v20250824")
    ap.add_argument("--mcp_tool_names", default="google_search")
    ap.add_argument("--num_docs", type=int, default=5)
    ap.add_argument("--mcp_timeout", type=int, default=60)
    ap.add_argument("--max_tool_calls", type=int, default=6)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--mcp_server_command",
        default=None,
        help="If set, launch this MCP server subprocess. If not set, assumes one is already running.",
    )
    args = ap.parse_args()

    if not os.environ.get("SERPER_API_KEY") and "google_search" in args.mcp_tool_names:
        print("WARNING: SERPER_API_KEY is not set; google_search tool calls will likely error.")

    sys_cfg = yaml.safe_load(open(args.system_prompt_yaml, "r", encoding="utf-8"))
    system_prompt = sys_cfg["system_prompt"]

    mcp_process = None
    if args.mcp_server_command:
        print(f"Launching MCP server: {args.mcp_server_command}")
        mcp_process = launch_mcp_subprocess(args.mcp_server_command, "./mcp_logs_smoke")
        time.sleep(3)

    # MCP tool wrapper
    tool_names = [t.strip() for t in args.mcp_tool_names.split(",") if t.strip()]
    mcp_tool = MCPTool(
        mcp_tool_names=tool_names,
        mcp_parser_name=args.mcp_parser_name,
        base_url=f"http://127.0.0.1:{args.mcp_port}{args.mcp_path}",
        number_documents_to_search=args.num_docs,
        mcp_timeout=args.mcp_timeout,
        mcp_host=args.mcp_host,
        mcp_port=args.mcp_port,
    )
    tool_objects = {end_str: mcp_tool for end_str in mcp_tool.get_stop_strings()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    llm = ToolUseLLM(
        model=args.model,
        tools=tool_objects,
        max_tool_calls=args.max_tool_calls,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        include_stop_str_in_output=True,
        stop=list(tool_objects.keys()),
        n=1,
    )

    out = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sp)[0]
    text = tokenizer.decode(out.outputs[0].token_ids, skip_special_tokens=False)

    calls = _extract_tool_calls(text)
    print("==== tool_use_smoke summary ====")
    print("model:", args.model)
    print("OPEN_INSTRUCT_MCP_NORMALIZE_TOOL_CALLS:", os.environ.get("OPEN_INSTRUCT_MCP_NORMALIZE_TOOL_CALLS", "1"))
    print("num_tool_call_tags:", len(calls))
    if calls:
        print("tool_call_tags:", [(t, n, q[:120]) for (t, n, q) in calls])
    print("has_answer_tags:", bool(re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL)))
    print("==== output (think redacted) ====")
    print(_redact_think(text))

    if mcp_process is not None:
        try:
            mcp_process.terminate()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


