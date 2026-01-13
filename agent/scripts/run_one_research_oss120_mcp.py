import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Make `workflows/` importable when running from `agent/scripts/`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.auto_search_sft import AutoReasonSearchWorkflow  # noqa: E402


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


async def _run_one(
    question: str,
    config_path: str,
    dataset_name: str,
    verbose: bool,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wf = AutoReasonSearchWorkflow(configuration=config_path, **(overrides or {}))
    wf.setup_components()
    return await wf(problem=question, dataset_name=dataset_name, verbose=verbose)


def _extract_messages(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Preferred: stitched transcript returned by the workflow when DR_OUTPUT_MESSAGES_SCHEMA=1
    msgs = result.get("messages")
    if isinstance(msgs, list):
        return msgs

    ft = result.get("full_traces") or {}
    if hasattr(ft, "model_dump"):
        ft = ft.model_dump()
    if isinstance(ft, dict):
        model_input = ft.get("model_input") or {}
        msgs = model_input.get("messages")
        if isinstance(msgs, list):
            return msgs
    return []


def _json_default(obj: Any) -> Any:
    # OpenAI / pydantic objects (e.g., ChatCompletionMessageToolCall) usually implement model_dump().
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return obj.model_dump()
    # Some older objects may expose dict()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return obj.dict()
    # Last resort: stringize
    return str(obj)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure messages are JSON-serializable (tool_calls may contain OpenAI SDK objects).
    """
    normalized: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            m = json.loads(json.dumps(m, default=_json_default, ensure_ascii=False))
            normalized.append(m)
            continue

        m2 = dict(m)
        if "tool_calls" in m2 and m2["tool_calls"] is not None:
            m2["tool_calls"] = json.loads(
                json.dumps(m2["tool_calls"], default=_json_default, ensure_ascii=False)
            )
        normalized.append(m2)
    return normalized


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--question",
        default=(
            "How does speculative decoding impact end-to-end latency and throughput in "
            "large-language-model serving systems (e.g., vLLM-style engines)? "
            "Explain mechanisms, tradeoffs, and when it helps vs hurts."
        ),
    )
    ap.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent.parent
            / "workflows"
            / "auto_search_sft-gpt-oss.yaml"
        ),
    )
    ap.add_argument("--dataset-name", default="scholarqa_cs2")
    ap.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parent.parent
            / "eval_output"
            / "oss_one_research_demo"
            / "oss120_one_research.jsonl"
        ),
    )
    ap.add_argument("--example-id", default="oss120_demo_0")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--no-browse",
        action="store_true",
        help="Disable browse_webpage tool (search-only). This avoids models hallucinating page-local tools.",
    )
    ap.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Override search_agent_max_tool_calls for this run.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override search_agent_temperature for this run.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override search_agent_max_tokens for this run.",
    )
    ap.add_argument(
        "--trace-messages",
        action="store_true",
        help="Force DR_TRACE_MODEL_INPUT=1 and DR_OUTPUT_MESSAGES_SCHEMA=1 for a full messages trace.",
    )
    args = ap.parse_args()

    if args.trace_messages:
        os.environ["DR_TRACE_MODEL_INPUT"] = "1"
        os.environ["DR_OUTPUT_MESSAGES_SCHEMA"] = "1"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overrides: Dict[str, Any] = {}
    if args.no_browse:
        overrides["browse_tool_name"] = None
    if isinstance(args.max_tool_calls, int):
        overrides["search_agent_max_tool_calls"] = args.max_tool_calls
    if isinstance(args.temperature, float):
        overrides["search_agent_temperature"] = args.temperature
    if isinstance(args.max_tokens, int):
        overrides["search_agent_max_tokens"] = args.max_tokens

    result = asyncio.run(
        _run_one(
            question=args.question,
            config_path=args.config,
            dataset_name=args.dataset_name,
            verbose=args.verbose,
            overrides=overrides or None,
        )
    )

    messages = _normalize_messages(_extract_messages(result))
    final_response = result.get("final_response")

    row: Dict[str, Any] = {
        "example_id": args.example_id,
        "problem": args.question,
        "messages": messages,
    }
    # Helpful for eyeballing without re-parsing the last assistant message.
    if isinstance(final_response, str) and final_response.strip():
        row["final_response"] = final_response

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print a quick summary for terminal use.
    print(f"Wrote: {out_path}")
    print(
        f"messages={len(messages)} has_tool_calls={any(m.get('role')=='assistant' and m.get('tool_calls') for m in messages)} "
        f"has_tool_msgs={any(m.get('role')=='tool' for m in messages)}"
    )
    if _truthy(os.environ.get("DR_OUTPUT_MESSAGES_SCHEMA")):
        # Show the last assistant message preview
        last_asst = None
        for m in reversed(messages):
            if m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls")):
                last_asst = m
                break
        if last_asst:
            content_preview = (last_asst.get("content") or "")[:240].replace("\n", " ")
            print(f"last_assistant_preview={content_preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


