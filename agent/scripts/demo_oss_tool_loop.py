import argparse
import json
from typing import Any, Dict, List, Optional

import requests


def get_secret_code(key: str) -> str:
    if key != "alpha":
        raise ValueError("unknown key")
    return "swordfish"


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--force-tool", action="store_true")
    args = ap.parse_args()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_secret_code",
                "description": "Return a secret code for a given key. You must call this tool to obtain the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Lookup key. Use 'alpha' for this test.",
                        }
                    },
                    "required": ["key"],
                },
            },
        }
    ]

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You MUST call get_secret_code(key='alpha') to answer. Do not guess. "
                "After you receive the tool result, reply exactly as: CODE: <value>"
            ),
        },
        {"role": "user", "content": "What is the secret code for key 'alpha'?"},
    ]

    req1: Dict[str, Any] = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 200,
        "messages": messages,
        "tools": tools,
        "tool_choice": (
            {"type": "function", "function": {"name": "get_secret_code"}}
            if args.force_tool
            else "auto"
        ),
    }

    r1 = _post_json(f"{args.base_url}/chat/completions", req1)
    msg1 = r1["choices"][0]["message"]
    tool_calls = msg1.get("tool_calls") or []
    if not tool_calls:
        print("No tool_calls returned. Full response:")
        print(json.dumps(r1, indent=2))
        return 2

    print("Step 1 tool_calls:")
    print(json.dumps(tool_calls, indent=2))

    # Execute tools and build tool messages
    tool_messages: List[Dict[str, Any]] = []
    for tc in tool_calls:
        fn = (tc.get("function") or {})
        name = fn.get("name")
        args_str = fn.get("arguments") or "{}"
        call_id = tc.get("id") or ""
        try:
            fn_args = json.loads(args_str)
        except Exception:
            fn_args = {"key": args_str}

        if name != "get_secret_code":
            out = f"ERROR: unknown tool {name}"
        else:
            out = get_secret_code(fn_args.get("key"))

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": out,
            }
        )

    # Step 2: send tool results back
    req2 = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": messages
        + [
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
        ]
        + tool_messages,
    }

    r2 = _post_json(f"{args.base_url}/chat/completions", req2)
    msg2 = r2["choices"][0]["message"]
    print("\nStep 2 final assistant message:")
    print(json.dumps(msg2, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


