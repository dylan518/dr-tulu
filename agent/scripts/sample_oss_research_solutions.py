import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v is not None and str(v).strip() != "" else default


async def _one_completion(
    *,
    model: str,
    api_base: str,
    api_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    resp = await litellm.acompletion(
        model=model,
        custom_llm_provider="openai",
        api_base=api_base,
        api_key=api_key,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    msg = resp.choices[0].message
    # Some OpenAI-compatible OSS servers populate `reasoning_content` but leave `content` empty/None.
    text = (getattr(msg, "content", None) or "") if msg is not None else ""
    if not text:
        text = (
            getattr(msg, "reasoning_content", None)
            or (
                isinstance(getattr(msg, "provider_specific_fields", None), dict)
                and msg.provider_specific_fields.get("reasoning_content")
            )
            or ""
        )
    usage = getattr(resp, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
    return {
        "text": text,
        "seed": seed,
        "usage": usage_dict,
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else None,
    }


async def _one_text_completion(
    *,
    model: str,
    api_base: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    resp = await litellm.atext_completion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    text = ""
    try:
        text = resp.choices[0].text or ""
    except Exception:
        text = ""
    usage = getattr(resp, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
    return {
        "text": text,
        "seed": seed,
        "usage": usage_dict,
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    api_base = args.base_url
    model = args.model
    api_key = args.api_key or "dummy"

    system = args.system
    user = args.question
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = f"{system}\n\n{user}\n"

    # vLLM /completions generally expects the SERVED model name.
    # If the user passes the HF repo ID, map it to the served name used by our vLLM script.
    served_model = model
    if served_model.strip() == "openai/gpt-oss-120b":
        served_model = "gpt-oss-120b"
    if args.mode == "completion" and not served_model.startswith("hosted_vllm/"):
        served_model = f"hosted_vllm/{served_model}"

    sem = asyncio.Semaphore(args.max_concurrent)

    async def run_i(i: int) -> Dict[str, Any]:
        async with sem:
            seed = (args.seed_base + i) if args.seed_base is not None else None
            t0 = datetime.utcnow().isoformat()
            if args.mode == "chat":
                out = await _one_completion(
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                )
            else:
                out = await _one_text_completion(
                    model=served_model,
                    api_base=api_base,
                    api_key=api_key,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                )
            t1 = datetime.utcnow().isoformat()
            return {
                "id": f"sample_{i}",
                "started_at": t0,
                "finished_at": t1,
                "model": model,
                "served_model": served_model,
                "base_url": api_base,
                "question": user,
                **out,
            }

    results = await asyncio.gather(*(run_i(i) for i in range(args.n)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} samples -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        default=_env("GPT_OSS_BASE_URL", "http://127.0.0.1:30001/v1"),
        help="OpenAI-compatible base URL for OSS vLLM server (e.g., http://127.0.0.1:30001/v1)",
    )
    p.add_argument(
        "--model",
        default=_env("GPT_OSS_MODEL_NAME", "openai/gpt-oss-120b"),
        help="Model name exposed by the server (e.g., openai/gpt-oss-120b)",
    )
    p.add_argument(
        "--mode",
        choices=["completion", "chat"],
        default="completion",
        help="Use vLLM /completions (recommended) or /chat/completions",
    )
    p.add_argument(
        "--api-key",
        default=_env("GPT_OSS_API_KEY", "dummy"),
        help="API key (not used by local vLLM, but required by clients)",
    )
    p.add_argument("--n", type=int, default=10, help="Number of samples to generate")
    p.add_argument("--max-concurrent", type=int, default=10, help="Max in-flight requests")
    p.add_argument("--max-tokens", type=int, default=4000, help="Completion token budget")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="If set, uses seed=seed_base+i for reproducible sampling.",
    )
    p.add_argument(
        "--output",
        default="/home/ubuntu/dr-tulu/agent/eval_output/oss_test_research_q10/oss_samples.jsonl",
    )
    p.add_argument(
        "--system",
        default=(
            "Write ONLY the final report. Do not include planning, meta commentary, or hidden reasoning. "
            "If you cite sources, include URLs in parentheses."
        ),
    )
    p.add_argument(
        "--question",
        default=(
            "Research question: How does speculative decoding impact end-to-end latency and throughput "
            "in large-language-model serving systems (e.g., vLLM-style engines)? "
            "Write a concise technical report with mechanisms, tradeoffs, and include citations with URLs."
        ),
    )
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


