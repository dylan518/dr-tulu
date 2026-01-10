import json
import asyncio
import weakref
import logging
import os
import re
import atexit
import time
from collections import deque
from typing import Any, Dict, Optional, List

import jsonlines
import litellm
from openai import AzureOpenAI

# Configure LiteLLM to drop unsupported parameters instead of raising errors
litellm.drop_params = True

LOGGER = logging.getLogger(__name__)

# When the Python interpreter is shutting down (Ray worker teardown, SIGTERM, Ctrl-C during shutdown),
# LiteLLM's async implementation may try to schedule work on an executor that has already been
# torn down, producing:
#   RuntimeError: cannot schedule new futures after interpreter shutdown
# Treat this as a shutdown condition and stop issuing further LiteLLM requests.
_LITELLM_SHUTTING_DOWN = False


def _mark_litellm_shutting_down() -> None:
    global _LITELLM_SHUTTING_DOWN
    _LITELLM_SHUTTING_DOWN = True


atexit.register(_mark_litellm_shutting_down)


def _maybe_enable_litellm_debug() -> None:
    """Enable LiteLLM debug logging if requested via env var.

    LiteLLM debug can be very noisy; we gate it behind `LITELLM_DEBUG=1`.
    """
    if os.environ.get("LITELLM_DEBUG") == "1":
        try:
            litellm._turn_on_debug()
        except Exception as e:
            LOGGER.warning(f"Failed to enable LiteLLM debug: {e}")


def _enforce_disallow_openai(model_name: str) -> None:
    """Fail fast if the caller attempted to use OpenAI/Azure models.

    Set `DISALLOW_OPENAI=1` to make any such usage an immediate error rather than
    a silent fallback / accidental spend.
    """
    if os.environ.get("DISALLOW_OPENAI") != "1":
        return
    mn = (model_name or "").strip()
    if mn.startswith("openai/") or mn.startswith("gpt-") or mn.startswith("azure/"):
        raise RuntimeError(f"DISALLOW_OPENAI=1 but model_name={mn!r} looks like an OpenAI/Azure model")


_maybe_enable_litellm_debug()


# Per-event-loop concurrency control for LiteLLM async calls to avoid event loop binding issues
_LITELLM_SEMAPHORES = weakref.WeakKeyDictionary()

# Optional token usage logging (helps estimate TPM / quota pressure)
_LITELLM_USAGE_WINDOW_START = None  # type: Optional[float]
_LITELLM_USAGE_TOTAL_TOKENS = 0
_LITELLM_USAGE_NUM_CALLS = 0


def _get_litellm_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore limiting concurrent LiteLLM async requests.

    Limit can be configured with env var `LITELLM_MAX_CONCURRENT_CALLS` (default 256).
    """
    loop = asyncio.get_running_loop()
    sem = _LITELLM_SEMAPHORES.get(loop)
    if sem is None:
        max_concurrent = int(os.environ.get("LITELLM_MAX_CONCURRENT_CALLS", "256"))
        sem = asyncio.Semaphore(max_concurrent)
        _LITELLM_SEMAPHORES[loop] = sem
    return sem


class _TpmLimiter:
    """Sliding-window TPM limiter (async).

    This is best-effort and uses an estimated token count per request to decide when to wait.
    For safety (avoid 429), the estimate is intentionally conservative.
    """

    def __init__(self, tpm_limit: int, window_seconds: float = 60.0):
        self.tpm_limit = int(tpm_limit)
        self.window_seconds = float(window_seconds)
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _sum_tokens(self) -> int:
        return sum(t for _, t in self._events)

    async def acquire(self, tokens: int) -> None:
        if tokens <= 0:
            return
        while True:
            async with self._lock:
                now = time.time()
                self._prune(now)
                used = self._sum_tokens()
                if used + tokens <= self.tpm_limit:
                    self._events.append((now, tokens))
                    return

                # Wait until enough old tokens expire.
                target = self.tpm_limit - tokens
                running = used
                wait_until = now + self.window_seconds
                for ts, t in self._events:
                    running -= t
                    if running <= target:
                        wait_until = ts + self.window_seconds
                        break
                sleep_s = max(0.01, wait_until - now)
            await asyncio.sleep(sleep_s)


_LITELLM_TPM_LIMITERS = weakref.WeakKeyDictionary()


def _get_litellm_tpm_limiter() -> Optional[_TpmLimiter]:
    """Return a per-event-loop TPM limiter if enabled via env var `LITELLM_MAX_TPM`."""
    tpm = os.environ.get("LITELLM_MAX_TPM")
    if not tpm:
        return None
    try:
        tpm_limit = int(float(tpm))
    except Exception:
        return None
    if tpm_limit <= 0:
        return None

    loop = asyncio.get_running_loop()
    limiter = _LITELLM_TPM_LIMITERS.get(loop)
    if limiter is None:
        window_s = float(os.environ.get("LITELLM_TPM_WINDOW_SECONDS", "60"))
        limiter = _TpmLimiter(tpm_limit=tpm_limit, window_seconds=window_s)
        _LITELLM_TPM_LIMITERS[loop] = limiter
    return limiter


def _estimate_request_tokens_for_tpm(messages: List[Dict[str, str]], chat_kwargs: Dict[str, Any]) -> int:
    """Conservative token estimate for TPM limiting (Gemini tokenizer not available here)."""
    prompt_chars = 0
    for m in messages or []:
        prompt_chars += len(m.get("content") or "")

    # Conservative: assume ~2 chars per token
    prompt_tokens_est = max(1, prompt_chars // 2)

    max_out = chat_kwargs.get("max_tokens")
    if max_out is None:
        max_out = chat_kwargs.get("max_completion_tokens")
    try:
        max_out_i = int(max_out) if max_out is not None else 0
    except Exception:
        max_out_i = 0

    return int(prompt_tokens_est + max(0, max_out_i))


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON-like dict from an LLM response.

    Gemini sometimes returns "almost JSON" such as:
      ```json
      { score: 1 }
      ```
    which is not strict JSON (bare keys). We try strict JSON first, then apply safe
    normalizations (strip code fences, quote bare keys, remove trailing commas),
    and finally fall back to yaml.safe_load (YAML is a superset that can parse bare keys).
    """
    if not response:
        return None

    # Strip Markdown code fences if present (```json ... ```).
    text = response.strip()
    if text.startswith("```"):
        # remove leading ```lang and trailing ```
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    json_end = text.rfind("}") + 1
    if json_end == 0:
        return None

    # Try to find valid JSON by testing different starting positions
    json_start = text.find("{")
    while json_start != -1 and json_start < json_end:
        candidate = text[json_start:json_end]

        def _postprocess_obj(o: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(o, dict):
                return None
            # Normalize keys like score" (Gemini sometimes emits `"score":` missing the first quote -> `score":`)
            fixed: Dict[str, Any] = {}
            for k, v in o.items():
                if isinstance(k, str):
                    kk = k.strip()
                    # Strip any surrounding quotes
                    if (kk.startswith('"') and kk.endswith('"')) or (kk.startswith("'") and kk.endswith("'")):
                        kk = kk[1:-1].strip()
                    # Strip dangling quotes on one side (e.g., score")
                    kk = kk.strip('"').strip("'").strip()
                    fixed[kk] = v
                else:
                    fixed[k] = v
            return fixed

        def _try_json(s: str) -> Optional[Dict[str, Any]]:
            try:
                obj = json.loads(s)
                return _postprocess_obj(obj)
            except json.JSONDecodeError:
                return None

        # 1) Strict JSON
        obj = _try_json(candidate)
        if obj is not None:
            return obj

        # 2) Clean BOM/whitespace and retry
        cleaned = candidate.strip().encode("utf-8").decode("utf-8-sig")
        obj = _try_json(cleaned)
        if obj is not None:
            return obj

        # 3) Fix doubled braces
        fixed_braces = cleaned.replace("{{", "{").replace("}}", "}")
        obj = _try_json(fixed_braces)
        if obj is not None:
            return obj

        # 4) Normalize "JSON-ish" to JSON:
        #    - quote bare keys: { score: 1 } -> { "score": 1 }
        #    - remove trailing commas
        #    - normalize Python literals to JSON
        normalized = fixed_braces
        # Fix cases like: { score": 2 } (missing leading quote)
        normalized = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\"(\s*:)", r"\1\2\3", normalized)
        normalized = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', normalized)
        normalized = re.sub(r",(\s*[}\]])", r"\1", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        normalized = re.sub(r"\bTrue\b", "true", normalized)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        obj = _try_json(normalized)
        if obj is not None:
            return obj

        # 5) YAML fallback (safe) – handles bare keys and some minor formatting issues.
        try:
            import yaml  # type: ignore

            yobj = yaml.safe_load(candidate)
            yfixed = _postprocess_obj(yobj)
            if yfixed is not None:
                return yfixed
        except Exception:
            pass

        # Try next { position
        json_start = text.find("{", json_start + 1)

    LOGGER.warning(f"Could not decode JSON from response: {repr(response)}")
    return None


def run_chatopenai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_mode: bool = False,
    **chat_kwargs,
) -> str:
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    if json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )
    resp = litellm.completion(
        model=model_name,
        messages=msgs,
        **chat_kwargs,
    )

    return resp.choices[0].message.content


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode="w") as writer:
        writer.write_all(data)


def run_azure_openai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    **chat_kwargs,
) -> str:
    """
    Run Azure OpenAI model with the given prompts.

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt
        user_prompt: User prompt
        endpoint: Azure OpenAI endpoint URL
        deployment: Azure OpenAI deployment name (defaults to model_name if not specified)
        api_version: Azure OpenAI API version
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """
    # Get Azure credentials from environment
    subscription_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not subscription_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

    # Use provided endpoint or get from environment
    if endpoint is None:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint must be provided or set in AZURE_OPENAI_ENDPOINT environment variable"
            )

    # Use provided deployment or default to model_name
    if deployment is None:
        deployment = model_name

    # Set default parameters
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_completion_tokens"] = chat_kwargs.get("max_completion_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)

    # Create Azure OpenAI client
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    # Prepare messages
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    # Create chat completion
    response = client.chat.completions.create(
        messages=msgs,
        model=deployment,
        **chat_kwargs,
    )

    return response.choices[0].message.content


def run_litellm(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    **chat_kwargs,
) -> str:
    """
    Run litellm for the given model.
    matches api for the run_azure_openai function.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt (defaults to None)
        user_prompt: User prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """

    _enforce_disallow_openai(model_name)

    # Set default parameters
    # GPT-5 family enforces temperature=1 on OpenAI; setting 0 will error.
    if "gpt-5" in (model_name or ""):
        chat_kwargs["temperature"] = chat_kwargs.get("temperature", 1)
        if chat_kwargs.get("temperature", 1) == 0:
            chat_kwargs["temperature"] = 1
    else:
        chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    # OpenAI GPT-5 family uses `max_completion_tokens` (and rejects `max_tokens`).
    # Keep backward compatibility for other providers/models.
    if "gpt-5" in (model_name or ""):
        chat_kwargs["max_completion_tokens"] = chat_kwargs.get("max_completion_tokens", chat_kwargs.get("max_tokens", 800))
        chat_kwargs.pop("max_tokens", None)
    else:
        chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)
    chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 5)
    # Default to no fallbacks to avoid silently routing requests to a different provider/model.
    # (Callers can still explicitly pass `fallbacks=[...]` if desired.)
    chat_kwargs["fallbacks"] = chat_kwargs.get("fallbacks", [])

    # Prepare messages
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    # Create chat completion
    try:
        response = litellm.completion(
            messages=msgs,
            model=model_name,
            **chat_kwargs,
        )
    except:
        # if we get an error, return an empty string
        return ""

    return response.choices[0].message.content


async def run_litellm_async(
    model_name: str,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **chat_kwargs,
) -> str:
    """
    Async version of run_litellm for the given model.
    matches api for the run_azure_openai function.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt (defaults to None)
        user_prompt: User prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """

    _enforce_disallow_openai(model_name)

    # Don't try to issue new requests while the worker is exiting.
    if _LITELLM_SHUTTING_DOWN:
        return ""

    # Set default parameters
    # GPT-5 family enforces temperature=1 on OpenAI; setting 0 will error.
    if "gpt-5" in (model_name or ""):
        chat_kwargs["temperature"] = chat_kwargs.get("temperature", 1)
        if chat_kwargs.get("temperature", 1) == 0:
            chat_kwargs["temperature"] = 1
    else:
        chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    # OpenAI GPT-5 family uses `max_completion_tokens` (and rejects `max_tokens`).
    if "gpt-5" in (model_name or ""):
        chat_kwargs["max_completion_tokens"] = chat_kwargs.get(
            "max_completion_tokens", chat_kwargs.get("max_tokens", 16384)
        )
        chat_kwargs.pop("max_tokens", None)
    else:
        chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 16384)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)
    chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 5)
    chat_kwargs["fallbacks"] = chat_kwargs.get("fallbacks", [])

    # Prepare messages
    if messages is not None:
        msgs = messages
    else:
        msgs = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if system_prompt is not None
            else [{"role": "user", "content": user_prompt}]
        )

    # Apply default timeout if not provided
    chat_kwargs["timeout"] = chat_kwargs.get(
        "timeout", float(os.environ.get("LITELLM_DEFAULT_TIMEOUT", "600"))
    )

    # Guard concurrent calls with a global semaphore
    try:
        # Optional strict TPM throttle (primarily for Gemini / Vertex quotas).
        # Enable with:
        #   export LITELLM_MAX_TPM=350000
        # Optionally tune:
        #   export LITELLM_TPM_WINDOW_SECONDS=60
        #
        # This uses a conservative estimate, so it should keep you under the limit, but may
        # underutilize slightly. For *closest* adherence, also set:
        #   export LITELLM_MAX_CONCURRENT_CALLS=1
        if os.environ.get("LITELLM_MAX_TPM") and "gemini" in (model_name or "").lower():
            limiter = _get_litellm_tpm_limiter()
            if limiter is not None:
                await limiter.acquire(_estimate_request_tokens_for_tpm(msgs, chat_kwargs))

        semaphore = _get_litellm_semaphore()
        async with semaphore:
            # Create chat completion
            response = await litellm.acompletion(
                messages=msgs,
                model=model_name,
                **chat_kwargs,
            )

            # Optional: print a rolling TPM estimate (very useful for diagnosing 429/RESOURCE_EXHAUSTED).
            # Enable with: LITELLM_LOG_USAGE=1
            if os.environ.get("LITELLM_LOG_USAGE") == "1":
                global _LITELLM_USAGE_WINDOW_START, _LITELLM_USAGE_TOTAL_TOKENS, _LITELLM_USAGE_NUM_CALLS
                if _LITELLM_USAGE_WINDOW_START is None:
                    _LITELLM_USAGE_WINDOW_START = time.time()

                usage = getattr(response, "usage", None)
                prompt_tokens = completion_tokens = total_tokens = None
                if usage is not None:
                    # LiteLLM may return usage as dict-like or object-like.
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                    else:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None)

                if total_tokens is None:
                    # Fallback: at least count something to avoid divide-by-zero; if usage is missing,
                    # we don't know exact tokens so we skip accounting.
                    total_tokens = 0

                _LITELLM_USAGE_TOTAL_TOKENS += int(total_tokens or 0)
                _LITELLM_USAGE_NUM_CALLS += 1

                now = time.time()
                elapsed = max(1e-6, now - (_LITELLM_USAGE_WINDOW_START or now))
                tpm_est = int((_LITELLM_USAGE_TOTAL_TOKENS / elapsed) * 60.0)

                every_n = int(os.environ.get("LITELLM_LOG_USAGE_EVERY", "50"))
                if _LITELLM_USAGE_NUM_CALLS % max(1, every_n) == 0:
                    rpm_est = int((_LITELLM_USAGE_NUM_CALLS / elapsed) * 60.0)
                    avg_tokens_per_call = (
                        (_LITELLM_USAGE_TOTAL_TOKENS / max(1, _LITELLM_USAGE_NUM_CALLS))
                        if _LITELLM_USAGE_NUM_CALLS > 0
                        else 0.0
                    )
                    LOGGER.warning(
                        f"[LiteLLM usage] model={model_name!r} calls={_LITELLM_USAGE_NUM_CALLS} "
                        f"window_tokens={_LITELLM_USAGE_TOTAL_TOKENS} elapsed_s={elapsed:.1f} "
                        f"est_rpm={rpm_est} est_tpm={tpm_est} avg_tokens_per_call={avg_tokens_per_call:.0f}"
                    )
    except Exception as e:
        msg = str(e)
        # Shutdown race: not a real connectivity problem; happens when the worker is exiting.
        if (
            "cannot schedule new futures after interpreter shutdown" in msg
            or "Event loop is closed" in msg
        ):
            _mark_litellm_shutting_down()
            return ""

        # Otherwise, return empty string (callers treat this as a failed judge call).
        print(f"Error in run_litellm_async: {e}")
        return ""

    return response.choices[0].message.content


if __name__ == "__main__":
    # Simple test case for run_chatopenai function
    def test_run_chatopenai():
        """Test the run_chatopenai function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_chatopenai function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # Note: This will require a valid model name and API credentials
            # Uncomment the line below to actually test (requires OPENAI_API_KEY)
            response = run_chatopenai("gpt-3.5-turbo", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    def test_run_azure():
        """Test the run_azure_openai function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_azure_openai function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # Note: This will require valid Azure OpenAI credentials
            # Use the correct deployment name that matches test_azure_api.py
            response = run_azure_openai(
                "gpt-4.5-preview", system_prompt, user_prompt, deployment="gpt-4.5-preview-standard"
            )
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    def test_run_litellm():
        """Test the run_litellm function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_litellm function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # assert env vars are set
            # for now, azure
            assert os.environ.get("HOSTED_VLLM_API_BASE") is not None

            response = run_litellm("hosted_vllm/Qwen/QwQ-32B-Preview", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    async def test_run_litellm_async():
        """Test the run_litellm_async function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_litellm_async function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # assert env vars are set
            # for now, azure
            assert os.environ.get("HOSTED_VLLM_API_BASE") is not None

            response = await run_litellm_async("hosted_vllm/Qwen/QwQ-32B-Preview", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    # test_run_chatopenai()
    # test_run_azure()
    # test_run_litellm()
    
    # Run async test
    import asyncio
    asyncio.run(test_run_litellm_async())
