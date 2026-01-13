#!/usr/bin/env python3
"""
Benchmark *decode* throughput (tokens/sec) against an OpenAI-compatible vLLM server.

Goal: measure "max TPS" by saturating vLLM with a deep queue (high concurrency) and
long, tool-free generations.

Modes:
- fixed-requests: run N requests total (simple)
- duration: run a worker loop for D seconds (best for saturation TPS)

Examples:

Duration-based saturation test (recommended):
  uv run python scripts/benchmark_decode_tps.py \
    --base-url http://127.0.0.1:30002/v1 \
    --model zai-org/GLM-4.7-FP8 \
    --max-tokens 1024 \
    --duration-s 90 \
    --concurrency 64

Fixed-requests:
  uv run python scripts/benchmark_decode_tps.py \
    --base-url http://127.0.0.1:30002/v1 \
    --model zai-org/GLM-4.7-FP8 \
    --max-tokens 2048 \
    --requests 128 \
    --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _now() -> float:
    return time.perf_counter()


def _make_prompt(min_output_tokens: int) -> str:
    # Keep prompt short to minimize prefill; we want to stress decode throughput.
    # Also avoid requesting hidden chain-of-thought; we just want long outputs.
    return (
        "Write a long technical essay on GPU inference throughput benchmarking.\n"
        f"Output length: at least ~{min_output_tokens} tokens (keep going until you reach it).\n"
        "No tools. No browsing. No meta-commentary.\n"
    )


@dataclass
class OneResult:
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: Optional[str] = None


async def _one_request(
    client: Any,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
    timeout_s: float,
) -> OneResult:
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # Helps debugging server logs (vLLM accepts extra fields).
        "user": f"tpsbench-{uuid.uuid4().hex[:8]}",
    }

    start = _now()
    try:
        resp = await client.post(url, json=payload, timeout=timeout_s)
        latency = _now() - start
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))
        return OneResult(
            ok=True,
            latency_s=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
        )
    except Exception as e:
        latency = _now() - start
        return OneResult(
            ok=False,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=f"{type(e).__name__}: {e}",
        )


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:30002/v1")
    ap.add_argument("--model", required=True, help="Model name as served by vLLM.")
    ap.add_argument("--max-tokens", type=int, default=2048, help="max_tokens for the completion.")
    ap.add_argument("--requests", type=int, default=128, help="Total number of requests to run (fixed-requests mode).")
    ap.add_argument("--duration-s", type=float, default=0.0, help="If >0, run worker loop for this many seconds (duration mode).")
    ap.add_argument("--concurrency", type=int, default=4, help="Number of concurrent in-flight requests.")
    ap.add_argument("--timeout-s", type=float, default=600.0, help="Per-request timeout in seconds.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--prompt", default=None, help="Optional explicit prompt; otherwise a default long-output prompt is used.")
    args = ap.parse_args()

    api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy-key"
    chat_url = args.base_url.rstrip("/") + "/chat/completions"

    prompt = args.prompt or _make_prompt(args.max_tokens)

    try:
        import httpx  # type: ignore
    except Exception:
        raise SystemExit("Missing dependency: httpx. Install via `uv pip install httpx`.")

    results: list[OneResult] = []

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}) as client:
        async def worker(stop_at: float) -> None:
            while True:
                if _now() >= stop_at:
                    return
                r = await _one_request(
                    client=client,
                    url=chat_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    prompt=prompt,
                    timeout_s=args.timeout_s,
                )
                results.append(r)

        t0 = _now()
        if args.duration_s and args.duration_s > 0:
            stop_at = t0 + float(args.duration_s)
            tasks = [asyncio.create_task(worker(stop_at)) for _ in range(max(1, args.concurrency))]
            await asyncio.gather(*tasks)
        else:
            # fixed-requests mode: run N total requests with at most `concurrency` in flight
            sem = asyncio.Semaphore(max(1, args.concurrency))

            async def run_one() -> None:
                async with sem:
                    r = await _one_request(
                        client=client,
                        url=chat_url,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        prompt=prompt,
                        timeout_s=args.timeout_s,
                    )
                    results.append(r)

            tasks = [asyncio.create_task(run_one()) for _ in range(max(1, args.requests))]
            await asyncio.gather(*tasks)
        t1 = _now()

    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    total_wall = max(1e-9, t1 - t0)
    sum_completion = sum(r.completion_tokens for r in ok)
    sum_prompt = sum(r.prompt_tokens for r in ok)
    agg_tps = sum_completion / total_wall
    avg_latency = sum(r.latency_s for r in ok) / max(1, len(ok))
    p50 = sorted((r.latency_s for r in ok))[len(ok) // 2] if ok else None

    print("=== decode TPS benchmark ===")
    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print(f"max_tokens: {args.max_tokens}")
    if args.duration_s and args.duration_s > 0:
        print(f"duration_s: {args.duration_s}")
    else:
        print(f"requests: {args.requests}")
    print(f"concurrency: {args.concurrency}")
    print(f"ok: {len(ok)}  errors: {len(bad)}")
    print(f"sum_prompt_tokens: {sum_prompt}")
    print(f"sum_completion_tokens: {sum_completion}")
    print(f"wall_time_s: {total_wall:.3f}")
    print(f"aggregate_decode_tps: {agg_tps:.2f} tokens/s")
    if ok:
        print(f"avg_latency_s: {avg_latency:.3f}  p50_latency_s: {p50:.3f}")
        # Per-request decode TPS is noisy but helpful.
        per_req = [r.completion_tokens / max(1e-9, r.latency_s) for r in ok if r.completion_tokens > 0]
        if per_req:
            per_req_sorted = sorted(per_req)
            pr50 = per_req_sorted[len(per_req_sorted) // 2]
            pr90 = per_req_sorted[int(len(per_req_sorted) * 0.9)]
            print(f"per_request_decode_tps: p50={pr50:.2f}  p90={pr90:.2f}")

    if bad:
        print("\n=== errors (first 5) ===")
        for r in bad[:5]:
            print(r.error)

    # Also emit a machine-readable summary line.
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "ok": len(ok),
        "errors": len(bad),
        "sum_prompt_tokens": sum_prompt,
        "sum_completion_tokens": sum_completion,
        "wall_time_s": total_wall,
        "aggregate_decode_tps": agg_tps,
    }
    print("\nJSON_SUMMARY:", json.dumps(summary, sort_keys=True))

    return 0 if not bad else 2


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()


