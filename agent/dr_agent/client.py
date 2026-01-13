import asyncio
import json
import os
import re
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs

import litellm
from litellm.utils import token_counter
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Optional transformers import for vLLM models only
try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

from .tool_interface.base import BaseTool
from .tool_interface.data_types import DocumentToolOutput, ToolInput, ToolOutput
from .tool_interface.tool_parsers import ToolCallInfo
from .tool_interface.executor import AsyncToolExecutor, ToolRequest

# Context variable to store the current client
_llm_tool_client_context: ContextVar[Optional["LLMToolClient"]] = ContextVar(
    "llm_tool_client", default=None
)

DEFAULT_LLM_MAX_CONCURRENT_CALLS = 20


class GenerateWithToolsOutput(BaseModel):
    """Output type for generate_with_tools method"""

    generated_text: str
    total_tokens: int
    tool_call_count: int
    stopped_reason: str
    tool_calls: List[Union[ToolOutput, DocumentToolOutput]]
    # Debug-only: capture the exact messages/prompt that were sent to the model.
    # This can be large; enable via DR_TRACE_MODEL_INPUT=1
    model_input: Optional[Dict[str, Any]] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation"""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32768
    repetition_penalty: float = 1.0
    top_k: int = 1
    min_p: float = 0.0
    retry_limit: int = 3
    timeout: int = 3600
    top_k: int = 1
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


class LLMToolClient:
    def _recover_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Recover tool calls from assistant text for models/servers that do not populate
        OpenAI-style message.tool_calls but still emit tool call markers in content.

        Known observed pattern (GLM-style):
          <tool_call>tool_name?arg1=val&arg2=val</think>

        Returns OpenAI-style tool_call dicts:
          {"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...json...}"}}
        """
        if not content:
            return []

        recovered: List[Dict[str, Any]] = []

        def _mk(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": f"recovered_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }

        # 1) Prefer well-formed blocks: <tool_call> ... </tool_call>
        for m in re.finditer(
            r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", content, flags=re.IGNORECASE
        ):
            inner = m.group(1)
            inner = re.sub(r"</?think>", "", inner, flags=re.IGNORECASE).strip()
            recovered.extend(self._recover_tool_calls_from_compact(inner, _mk))
            recovered.extend(self._recover_tool_calls_from_arg_tags(inner, _mk))

        # 2) Handle common malformed pattern: <tool_call>... (no closing tag)
        # Capture up to the next '<' (e.g., before </think>) or end of string.
        for m in re.finditer(r"<tool_call>\s*([^<\n]+)", content, flags=re.IGNORECASE):
            inner = m.group(1)
            inner = re.sub(r"</?think>", "", inner, flags=re.IGNORECASE).strip()
            recovered.extend(self._recover_tool_calls_from_compact(inner, _mk))
            recovered.extend(self._recover_tool_calls_from_arg_tags(inner, _mk))

        # De-dupe identical recovered calls (best-effort) while preserving order
        seen = set()
        out: List[Dict[str, Any]] = []
        for c in recovered:
            key = (c.get("function", {}).get("name"), c.get("function", {}).get("arguments"))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _recover_tool_calls_from_compact(
        self, compact: str, mk: Callable[[str, Dict[str, Any]], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse a compact tool call encoding like:
          tool_name?arg1=val&arg2=val
        """
        if not compact:
            return []

        # Drop any trailing thought markers if present.
        compact = compact.strip().strip('"').strip()

        # tool?querystring
        if "?" in compact:
            name, qs = compact.split("?", 1)
            name = name.strip()
            qs = qs.strip()
            if not name:
                return []
            parsed = parse_qs(qs, keep_blank_values=True)
            args: Dict[str, Any] = {k: (v[-1] if isinstance(v, list) and v else "") for k, v in parsed.items()}
            return [mk(name, args)]

        # If it's just a name with no args, still return an empty-args call.
        if re.fullmatch(r"[A-Za-z0-9_\-]+", compact):
            return [mk(compact, {})]

        return []

    def _recover_tool_calls_from_arg_tags(
        self, inner: str, mk: Callable[[str, Dict[str, Any]], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse GLM-style tool call encoding like:
          get_secret_code<arg_key>key</arg_key><arg_value>alpha</arg_value>

        Some vLLM+GLM setups emit this inside <tool_call>...</tool_call> but do not populate
        OpenAI `tool_calls`.
        """
        if not inner:
            return []

        # Extract tool name: prefix before the first tag.
        name = inner.split("<", 1)[0].strip()
        if not name:
            return []

        # Extract key/value pairs.
        keys = re.findall(r"<arg_key>\s*([\s\S]*?)\s*</arg_key>", inner, flags=re.IGNORECASE)
        vals = re.findall(r"<arg_value>\s*([\s\S]*?)\s*</arg_value>", inner, flags=re.IGNORECASE)
        if not keys or not vals:
            return []

        args: Dict[str, Any] = {}
        for k, v in zip(keys, vals):
            k = (k or "").strip()
            v = (v or "").strip()
            if not k:
                continue
            args[k] = v

        if not args:
            return []

        return [mk(name, args)]
    """LLM Client with integrated tool calling"""

    # Global semaphore for controlling concurrent generation calls
    _global_semaphore = None
    _max_concurrent_calls = None

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        # Initialize global semaphore if not already done
        if LLMToolClient._global_semaphore is None:
            LLMToolClient._max_concurrent_calls = int(
                os.environ.get(
                    "LLM_MAX_CONCURRENT_CALLS", DEFAULT_LLM_MAX_CONCURRENT_CALLS
                )
            )
            LLMToolClient._global_semaphore = asyncio.Semaphore(
                LLMToolClient._max_concurrent_calls
            )

        # Store connection parameters for LiteLLM
        self.api_key = api_key
        self.base_url = base_url
        self.client = client  # For custom client if provided

        self.tools = tools or []
        # High-concurrency tool execution (no internal rate limiting).
        self._tool_executor = AsyncToolExecutor(
            enable_cache=os.environ.get("TOOL_EXECUTOR_CACHE", "1").lower()
            in ("1", "true", "yes"),
            enable_inflight_dedupe=os.environ.get("TOOL_EXECUTOR_INFLIGHT_DEDUPE", "1")
            .lower()
            in ("1", "true", "yes"),
        )
        self.model_name = model_name
        self.generation_config = generation_config or GenerationConfig()

        # Validate tool parser types
        self._validate_tool_parser_types()

        # Detect if this is a commercial API model or self-hosted model
        self.is_commercial_api_model = self._is_commercial_api_model(model_name)

        # Initialize tokenizer only for self-hosted models
        if not self.is_commercial_api_model and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name if tokenizer_name else model_name
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def _validate_tool_parser_types(self):
        """Validate that all tools use the same type of tool parser"""
        if len(self.tools) <= 1:
            return  # No validation needed for 0 or 1 tools

        # Get the parser type of the first tool
        first_tool_parser_type = type(self.tools[0].tool_parser).__name__

        # Check that all other tools have the same parser type
        for i, tool in enumerate(self.tools[1:], 1):
            current_parser_type = type(tool.tool_parser).__name__
            if current_parser_type != first_tool_parser_type:
                raise ValueError(
                    f"All tools must use the same parser type. Tool '{self.tools[0].name}' "
                    f"uses {first_tool_parser_type}, but tool '{tool.name}' uses {current_parser_type}."
                )

    def _get_tool_parser_type(self) -> Optional[str]:
        """Get the parser type used by tools. Returns 'legacy', 'unified', or None if no tools."""
        if not self.tools:
            return None

        parser_class_name = type(self.tools[0].tool_parser).__name__
        if parser_class_name == "LegacyToolCallParser":
            return "legacy"
        elif parser_class_name == "UnifiedToolCallParser":
            return "unified"
        else:
            return None

    def _is_commercial_api_model(self, model_name: str) -> bool:
        """Detect if this is a commercial API model vs a self-hosted model

        Commercial API models (OpenAI, Claude, etc.) use chat completion APIs and don't need tokenizers.
        Self-hosted models (vLLM) use text completion APIs and benefit from tokenizers.
        """
        # OpenAI model patterns
        openai_patterns = [
            "gpt-",
            "o1-",
            "text-",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "chatgpt",
            "gpt4",
            "turbo",
        ]

        # Anthropic Claude patterns
        claude_patterns = ["claude-", "sonnet", "haiku", "opus"]

        # Other commercial API patterns
        commercial_patterns = [
            "gemini",
            "palm",  # Google
            "command",
            "coral",  # Cohere
        ]

        all_commercial_patterns = (
            openai_patterns + claude_patterns + commercial_patterns
        )
        model_lower = model_name.lower()

        return any(pattern in model_lower for pattern in all_commercial_patterns)

    def _supports_openai_style_tool_calling(self) -> bool:
        """
        Whether we can use OpenAI-style native tool calling (tools/tool_calls) for this client.

        This is true for:
        - Commercial API models handled via LiteLLM (OpenAI/Claude/etc.)
        - Self-hosted models behind an OpenAI-compatible API (e.g., vLLM / TGI) when base_url is provided
        """
        return bool(self.base_url) or self._is_commercial_api_model(self.model_name)

    def add_tool(self, tool: BaseTool):
        """Add a tool to the client"""
        # Validate parser type consistency if we already have tools
        if self.tools:
            existing_parser_type = type(self.tools[0].tool_parser).__name__
            new_parser_type = type(tool.tool_parser).__name__
            if existing_parser_type != new_parser_type:
                raise ValueError(
                    f"Cannot add tool '{tool.name}' with parser type {new_parser_type}. "
                    f"All tools must use the same parser type (existing tools use {existing_parser_type})."
                )

        self.tools.append(tool)

    def _get_all_stop_sequences(self) -> List[str]:
        """Get all tool stop sequences"""
        all_sequences = []
        for tool in self.tools:
            sequences = tool.stop_sequences
            for seq in sequences:
                if seq not in all_sequences:
                    all_sequences.append(seq)
        return all_sequences

    def _find_first_tool_call(self, text: str) -> Optional[BaseTool]:
        """Find the first tool call in the text using the new clean interface
        Returns: (tool, call_info) or None
        """

        for tool in self.tools:
            if tool.has_calls(text):
                return tool
        return None

    def _extract_all_tool_calls_from_text(
        self, text: str
    ) -> List[Tuple[BaseTool, ToolCallInfo, str]]:
        """
        Extract ALL tool calls in left-to-right order from `text`.

        This is used for parser-mode + vLLM/text-mode tool calling, where models may emit
        multiple tool calls in a single assistant chunk.
        """
        results: List[Tuple[BaseTool, ToolCallInfo, str]] = []
        cursor = 0
        while cursor < len(text):
            remaining = text[cursor:]
            best_tool: Optional[BaseTool] = None
            best_info: Optional[ToolCallInfo] = None
            best_abs_start: Optional[int] = None
            best_abs_end: Optional[int] = None

            for tool in self.tools:
                info = tool.parse_call(remaining)
                if info is None:
                    continue
                abs_start = cursor + info.start_pos
                abs_end = cursor + info.end_pos
                if best_abs_start is None or abs_start < best_abs_start:
                    best_tool = tool
                    best_info = info
                    best_abs_start = abs_start
                    best_abs_end = abs_end

            if best_tool is None or best_info is None:
                break

            call_text = text[best_abs_start:best_abs_end]
            results.append((best_tool, best_info, call_text))
            cursor = best_abs_end

        return results

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text"""
        if self.is_commercial_api_model:
            # For commercial API models, use litellm.utils.token_counter with model name
            try:
                return token_counter(model=self.model_name, text=text)
            except Exception as e:
                print(f"Warning: Could not count tokens with litellm: {e}")
                # Fallback: rough estimate (4 characters per token)
                return len(text) // 4
        elif self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                print(f"Warning: Could not count tokens: {e}")
                # Fallback: rough estimate (4 characters per token)
                return len(text) // 4
        else:
            # Fallback: rough estimate (4 characters per token)
            return len(text) // 4

    def _count_tokens_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens for a list of messages"""
        if self.is_commercial_api_model:
            # For commercial API models, use litellm.utils.token_counter with messages
            try:
                return token_counter(model=self.model_name, messages=messages)
            except Exception as e:
                print(f"Warning: Could not count message tokens with litellm: {e}")
                # Fallback: count by concatenating content
                return self._count_tokens("".join(msg["content"] for msg in messages))
        else:
            # For self-hosted models, concatenate and count
            return self._count_tokens("".join(msg["content"] for msg in messages))

    def _calculate_dynamic_max_tokens(
        self, current_context: str, base_max_tokens: int
    ) -> int:
        """Calculate remaining max tokens based on current context length"""
        if self.is_commercial_api_model:
            # For commercial API models, don't do dynamic calculation - use fixed max_tokens
            return base_max_tokens

        current_token_count = self._count_tokens(current_context)
        remaining_tokens = base_max_tokens - current_token_count

        # Ensure we have at least some tokens for generation (minimum 100)
        return max(100, remaining_tokens)

    def _calculate_dynamic_max_tokens_messages(
        self, current_messages: List[Dict[str, str]], base_max_tokens: int
    ) -> int:
        """Calculate remaining max tokens based on current messages length for commercial API models"""
        current_token_count = self._count_tokens_messages(current_messages)
        # IMPORTANT:
        # - For OpenAI-compatible chat APIs, `max_tokens` is a COMPLETION budget, but the provider
        #   enforces a TOTAL context limit (prompt + completion <= context_length).
        # - Token counting can differ between LiteLLM/token_counter and the server-side tokenizer.
        #   Apply a safety buffer so we don't exceed the provider's context window.
        safety_buffer = int(os.environ.get("DR_TOKEN_SAFETY_BUFFER", "512"))
        remaining_tokens = base_max_tokens - current_token_count - safety_buffer

        # Ensure we have at least some tokens for generation (minimum 64)
        return max(64, remaining_tokens)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string using tokenizer chat template"""
        if self.is_commercial_api_model:
            # For commercial API models, return messages as-is since we'll use chat completion API
            raise NotImplementedError(
                "Commercial API models should use chat completion API directly"
            )

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Could not apply chat template: {e}")
                # Fallback to simple concatenation
                return self._fallback_messages_to_prompt(messages)
        else:
            return self._fallback_messages_to_prompt(messages)

    def _fallback_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Fallback method to convert messages to prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add assistant prefix for generation
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    async def generate_with_tools(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_tool_calls: int = 10,
        include_tool_results: bool = True,
        verbose: bool = False,
        generation_prefix: Optional[str] = None,
        include_reasoning: bool = True,  # New parameter for reasoning content
        tool_calling_mode: str = "parser",  # "parser" or "native"
        # Overridable generation parameters
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        on_step_callback: Optional[Callable[[str, List[ToolOutput]], None]] = None,
        **kwargs,
    ) -> GenerateWithToolsOutput:
        """Generate response with automatic tool calling

        Args:
            prompt_or_messages: Either a string prompt or list of message dictionaries
            max_tool_calls: Maximum number of tool calls to make (default: 10)
            include_tool_results: Whether to include tool results in the context (default: True)
            verbose: Whether to print debug information (default: False)
            generation_prefix: Optional prefix to add to generation (default: None)
            include_reasoning: Whether to include reasoning content from models that support it,
                             wrapped in <think>...</think> tags (default: False, only for commercial APIs)
            tool_calling_mode: Mode for tool calling - "parser" uses custom XML parsing,
                             "native" uses OpenAI-style native tool calling (default: "parser")
            temperature: Sampling temperature (default: from config)
            top_p: Nucleus sampling parameter (default: from config)
            max_tokens: Maximum tokens to generate (default: from config)
            top_k: Top-k sampling parameter (default: from config, vLLM only)
            repetition_penalty: Repetition penalty (default: from config, vLLM only)
            seed: Random seed for reproducible generation (default: from config)
            stop: Additional stop sequences (default: tool end tags)
            on_step_callback: Optional callback called after each generation step with (generated_text, tool_calls)
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            GenerateWithToolsOutput: Contains tool calls, generated text, token counts, etc.
        """

        async with LLMToolClient._global_semaphore:
            # Route to native tool calling if requested
            # print(f"tool_calling_mode: {tool_calling_mode}")
            if tool_calling_mode == "native":
                return await self._generate_with_tools_native(
                    prompt_or_messages,
                    max_tool_calls,
                    include_tool_results,
                    verbose,
                    temperature,
                    top_p,
                    max_tokens,
                    seed,
                    **kwargs,
                )
            elif tool_calling_mode == "parser":
                # Original parser-based tool calling
                if self.is_commercial_api_model:
                    return await self._generate_with_tools_commercial_api(
                        prompt_or_messages,
                        max_tool_calls,
                        include_tool_results,
                        verbose,
                        generation_prefix,
                        include_reasoning,
                        temperature,
                        top_p,
                        max_tokens,
                        top_k,
                        repetition_penalty,
                        seed,
                        stop,
                        on_step_callback=on_step_callback,
                        **kwargs,
                    )
                else:
                    return await self._generate_with_tools_vllm(
                        prompt_or_messages,
                        max_tool_calls,
                        include_tool_results,
                        verbose,
                        generation_prefix,
                        temperature,
                        top_p,
                        max_tokens,
                        top_k,
                        repetition_penalty,
                        seed,
                        stop,
                        on_step_callback=on_step_callback,
                        **kwargs,
                    )
            else:
                raise ValueError(
                    f"Invalid tool_calling_mode: {tool_calling_mode}. Must be 'parser' or 'native'."
                )

    async def _generate_with_tools_commercial_api(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_tool_calls: int,
        include_tool_results: bool,
        verbose: bool,
        generation_prefix: Optional[str],
        include_reasoning: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        seed: Optional[int],
        stop: Optional[List[str]],
        on_step_callback: Optional[Callable[[str, List[ToolOutput]], None]] = None,
        **kwargs,
    ) -> GenerateWithToolsOutput:
        """Generate response for commercial API models using chat completion API"""

        # Convert to messages format if needed
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages.copy()

        # Add generation prefix to the last user message if provided
        if generation_prefix:
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += generation_prefix
            else:
                warnings.warn(
                    "Generation prefix provided but no user message found. Ignoring prefix."
                )

        # Track results
        tool_calls: List[ToolOutput] = []
        current_messages = messages.copy()
        tool_call_count = 0
        iteration = 0
        original_message_count = len(messages)

        # Get base max tokens from parameter or config
        base_max_tokens = (
            max_tokens if max_tokens is not None else self.generation_config.max_tokens
        )

        while True:
            iteration += 1

            if verbose:
                print(f"\n--- Commercial API Iteration {iteration} ---")
                print(f"Tool calls made so far: {tool_call_count}")
                print(f"Current messages count: {len(current_messages)}")
                print(
                    f"Current message tokens: {self._count_tokens_messages(current_messages)}"
                )

            # Calculate dynamic max tokens for this generation step
            dynamic_max_tokens = self._calculate_dynamic_max_tokens_messages(
                current_messages, base_max_tokens
            )

            if verbose:
                print(f"Dynamic max tokens for this step: {dynamic_max_tokens}")

            # Check if we've hit the token limit before generation
            current_token_count = self._count_tokens_messages(current_messages)
            if current_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit before generation ({current_token_count}/{base_max_tokens}), stopping."
                    )
                break

            # Generate response using chat completion
            response_content = await self._generate_single_response_commercial_api(
                current_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=dynamic_max_tokens,  # Use dynamic max tokens
                seed=seed,
                stop_sequences=stop or self._get_all_stop_sequences(),
                verbose=verbose,
                include_reasoning=include_reasoning,
                **kwargs,
            )

            # Call on_step_callback with the new generation
            if on_step_callback:
                # Check if callback is async (either a coroutine function or has async __call__)
                is_async = asyncio.iscoroutinefunction(on_step_callback) or (
                    callable(on_step_callback)
                    and asyncio.iscoroutinefunction(
                        getattr(on_step_callback, "__call__", None)
                    )
                )
                if is_async:
                    await on_step_callback(response_content, [])
                else:
                    on_step_callback(response_content, [])

            # Add assistant response to messages
            current_messages.append({"role": "assistant", "content": response_content})

            # Check if we've hit the token limit after generation
            new_token_count = self._count_tokens_messages(current_messages)
            if new_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit after generation ({new_token_count}/{base_max_tokens}), stopping."
                    )
                break

            # Check for tool calls (extract ALL tool calls in the chunk)
            tool_parse_text = (
                generation_prefix + response_content
                if (generation_prefix and iteration == 1)
                else response_content
            )
            extracted_calls = self._extract_all_tool_calls_from_text(tool_parse_text)

            if not extracted_calls:
                # No tool calls found, we're done
                if verbose:
                    print("No tool calls found, finishing.")
                break

            # Check if we've exceeded the maximum number of tool calls
            if tool_call_count + len(extracted_calls) > max_tool_calls:
                if verbose:
                    print(
                        f"Exceeded maximum tool calls ({max_tool_calls}), creating error output."
                    )

                tool0 = extracted_calls[0][0]
                error_output = tool0._create_error_output(
                    error_msg="Exceed allowed tool call requests",
                    call_id="",
                    runtime=0,
                    output="Exceed allowed tool call requests.",
                )
                tool_calls.append(error_output)

                if include_tool_results:
                    error_formatted = tool0.format_result(error_output)
                    current_messages.append({"role": "user", "content": error_formatted})
                break

            if verbose:
                tool_names = [t.name for (t, _, _) in extracted_calls]
                print(f"Found {len(extracted_calls)} tool call(s): {tool_names}")

            # Execute all tools concurrently via the shared executor
            reqs = [
                ToolRequest(tool=tool, tool_input=call_text, call_info=call_info)
                for (tool, call_info, call_text) in extracted_calls
            ]
            tool_outputs = await asyncio.gather(
                *(self._tool_executor.execute(r) for r in reqs), return_exceptions=True
            )

            realized_outputs: List[ToolOutput] = []
            for (tool, _call_info, _call_text), tool_output in zip(
                extracted_calls, tool_outputs
            ):
                if isinstance(tool_output, Exception):
                    realized_outputs.append(
                        tool._create_error_output(
                            error_msg=str(tool_output),
                            call_id="",
                            runtime=0,
                            output=f"Error executing tool: {str(tool_output)}",
                        )
                    )
                else:
                    realized_outputs.append(tool_output)

            tool_call_count += len(realized_outputs)
            tool_calls.extend(realized_outputs)

            if on_step_callback:
                is_async = asyncio.iscoroutinefunction(on_step_callback) or (
                    callable(on_step_callback)
                    and asyncio.iscoroutinefunction(
                        getattr(on_step_callback, "__call__", None)
                    )
                )
                if is_async:
                    await on_step_callback("", realized_outputs)
                else:
                    on_step_callback("", realized_outputs)

            if include_tool_results:
                for (tool, _call_info, _call_text), out in zip(
                    extracted_calls, realized_outputs
                ):
                    if out.called:
                        result_formatted = tool.format_result(out)
                        current_messages.append(
                            {"role": "user", "content": result_formatted}
                        )
                        if verbose:
                            print(f"Tool '{tool.name}' output: {out.output[:200]}...")

            # Check token limit again after adding tool results
            final_token_count = self._count_tokens_messages(current_messages)
            if final_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit after tool execution ({final_token_count}/{base_max_tokens}), stopping."
                    )
                break

        # Calculate generated text (everything after original messages)
        generated_text = ""
        for msg in current_messages[original_message_count:]:
            generated_text += msg["content"]

        final_token_count = self._count_tokens_messages(current_messages)
        trace_model_input = str(os.environ.get("DR_TRACE_MODEL_INPUT", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        return GenerateWithToolsOutput(
            tool_calls=tool_calls,
            generated_text=generated_text,
            total_tokens=final_token_count,
            tool_call_count=tool_call_count,
            stopped_reason=(
                "max_tokens" if final_token_count >= base_max_tokens else "natural"
            ),
            model_input=({"messages": current_messages} if trace_model_input else None),
        )

    async def _generate_with_tools_vllm(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_tool_calls: int,
        include_tool_results: bool,
        verbose: bool,
        generation_prefix: Optional[str],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        seed: Optional[int],
        stop: Optional[List[str]],
        on_step_callback: Optional[Callable[[str, List[ToolOutput]], None]] = None,
        **kwargs,
    ) -> GenerateWithToolsOutput:
        """Generate response for self-hosted models (vLLM) using text completion API"""

        if isinstance(prompt_or_messages, str):
            prompt = prompt_or_messages
        else:
            messages = prompt_or_messages
            prompt = self._messages_to_prompt(messages)

        # Track results
        tool_calls: List[ToolOutput] = []
        current_context = prompt
        tool_call_count = 0

        # Get base max tokens from parameter or config
        base_max_tokens = (
            max_tokens if max_tokens is not None else self.generation_config.max_tokens
        )
        iteration = 0
        while True:
            iteration += 1

            if verbose:
                print(f"\n--- vLLM Iteration {iteration} ---")
                print(f"Tool calls made so far: {tool_call_count}")
                print(f"Current context tokens: {self._count_tokens(current_context)}")
                print(f"Current context: ...{current_context[-1000:]}")

            # Calculate dynamic max tokens for this generation step
            dynamic_max_tokens = self._calculate_dynamic_max_tokens(
                current_context, base_max_tokens
            )
            # print(
            #     f"Dynamic max tokens for this step: {dynamic_max_tokens}; base max tokens: {base_max_tokens}"
            # )

            if verbose:
                print(f"Dynamic max tokens for this step: {dynamic_max_tokens}")

            if generation_prefix and iteration == 1:
                current_context = current_context + generation_prefix

            # Generate response
            response = await self._generate_single_response_vllm(
                current_context,
                stop_sequences=stop or self._get_all_stop_sequences(),
                temperature=temperature,
                top_p=top_p,
                max_tokens=dynamic_max_tokens,  # Use dynamic max tokens
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
                **kwargs,
            )

            # Append response to context
            current_context += response

            # Call on_step_callback with the new generation
            if on_step_callback:
                is_async = asyncio.iscoroutinefunction(on_step_callback) or (
                    callable(on_step_callback)
                    and asyncio.iscoroutinefunction(
                        getattr(on_step_callback, "__call__", None)
                    )
                )
                if is_async:
                    await on_step_callback(response, [])
                else:
                    on_step_callback(response, [])

            # Check if we've hit the token limit after generation
            new_token_count = self._count_tokens(current_context)
            if new_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit after generation ({new_token_count}/{base_max_tokens}), stopping."
                    )
                break

            # Check for tool calls (extract ALL tool calls in the chunk)
            tool_parse_text = (
                generation_prefix + response
                if (generation_prefix and iteration == 1)
                else response
            )
            extracted_calls = self._extract_all_tool_calls_from_text(tool_parse_text)

            if not extracted_calls:
                # No tool calls found, we're done
                if verbose:
                    print("No tool calls found, finishing.")
                break

            # Check if we've exceeded the maximum number of tool calls
            if tool_call_count + len(extracted_calls) > max_tool_calls:
                if verbose:
                    print(
                        f"Exceeded maximum tool calls ({max_tool_calls}), creating error output."
                    )

                tool0 = extracted_calls[0][0]
                error_output = tool0._create_error_output(
                    error_msg="exceed allowed tool call requests",
                    call_id="",
                    runtime=0,
                    output="Exceed allowed tool call requests.",
                )

                tool_calls.append(error_output)

                if include_tool_results:
                    error_formatted = tool0.format_result(error_output)
                    current_context += "\n" + error_formatted

            else:
                if verbose:
                    tool_names = [t.name for (t, _, _) in extracted_calls]
                    print(f"Found {len(extracted_calls)} tool call(s): {tool_names}")

                reqs = [
                    ToolRequest(tool=tool, tool_input=call_text, call_info=call_info)
                    for (tool, call_info, call_text) in extracted_calls
                ]
                tool_outputs = await asyncio.gather(
                    *(self._tool_executor.execute(r) for r in reqs),
                    return_exceptions=True,
                )

                realized_outputs: List[ToolOutput] = []
                for (tool, _call_info, _call_text), tool_output in zip(
                    extracted_calls, tool_outputs
                ):
                    if isinstance(tool_output, Exception):
                        realized_outputs.append(
                            tool._create_error_output(
                                error_msg=str(tool_output),
                                call_id="",
                                runtime=0,
                                output=f"Error executing tool: {str(tool_output)}",
                            )
                        )
                    else:
                        realized_outputs.append(tool_output)

                tool_call_count += len(realized_outputs)
                tool_calls.extend(realized_outputs)

                if on_step_callback:
                    is_async = asyncio.iscoroutinefunction(on_step_callback) or (
                        callable(on_step_callback)
                        and asyncio.iscoroutinefunction(
                            getattr(on_step_callback, "__call__", None)
                        )
                    )
                    if is_async:
                        await on_step_callback("", realized_outputs)
                    else:
                        on_step_callback("", realized_outputs)

                if include_tool_results:
                    for (tool, _call_info, _call_text), out in zip(
                        extracted_calls, realized_outputs
                    ):
                        if out.called:
                            result_formatted = tool.format_result(out)
                            current_context += result_formatted
                            if verbose:
                                print(f"Tool '{tool.name}' output: {out.output[:200]}...")

            # Check token limit again after adding tool results
            final_token_count = self._count_tokens(current_context)
            if final_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit after tool execution ({final_token_count}/{base_max_tokens}), stopping."
                    )
                break

        trace_model_input = str(os.environ.get("DR_TRACE_MODEL_INPUT", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        return GenerateWithToolsOutput(
            tool_calls=tool_calls,
            generated_text=current_context[len(prompt) :],
            total_tokens=self._count_tokens(current_context),
            tool_call_count=tool_call_count,
            stopped_reason=(
                "max_tokens"
                if self._count_tokens(current_context) >= base_max_tokens
                else "natural"
            ),
            model_input=({"prompt": current_context} if trace_model_input else None),
        )

    async def _generate_with_tools_native(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        max_tool_calls: int,
        include_tool_results: bool,
        verbose: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
        **kwargs,
    ) -> GenerateWithToolsOutput:
        """Generate response using native OpenAI-style tool calling via litellm"""

        # Validate model supports OpenAI-style native tool calling (tools/tool_calls).
        # This includes commercial APIs and self-hosted OpenAI-compatible endpoints (when base_url is set).
        if not self._supports_openai_style_tool_calling():
            raise ValueError(
                "Native tool calling requires an OpenAI-style tools/tool_calls API. "
                f"Model '{self.model_name}' does not look like a supported commercial model and no base_url was provided. "
                "Either set base_url to your OpenAI-compatible server or use tool_calling_mode='parser'."
            )

        # Convert to messages format if needed
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages.copy()

        # Build OpenAI tool schemas
        tools = [tool.to_openai_tool_schema() for tool in self.tools]

        # Create tool lookup dict for O(1) access
        tools_by_name = {tool.name: tool for tool in self.tools}

        # Track results
        tool_calls_made: List[ToolOutput] = []
        current_messages = messages.copy()
        tool_call_count = 0
        iteration = 0
        original_message_count = len(messages)

        # Get base max tokens from parameter or config
        base_max_tokens = (
            max_tokens if max_tokens is not None else self.generation_config.max_tokens
        )

        # If caller forces a tool_choice, only apply it on the FIRST iteration.
        # Forcing tool_choice on every iteration can cause infinite tool loops (e.g., vLLM will keep
        # returning tool_calls even after tool results are provided).
        forced_tool_choice = kwargs.pop("tool_choice", "auto")

        while True:
            iteration += 1

            if verbose:
                print(f"\n--- Native Tool Calling Iteration {iteration} ---")
                print(f"Tool calls made so far: {tool_call_count}")
                print(f"Current messages count: {len(current_messages)}")

            # Calculate dynamic max tokens
            dynamic_max_tokens = self._calculate_dynamic_max_tokens_messages(
                current_messages, base_max_tokens
            )

            if verbose:
                print(f"Dynamic max tokens for this step: {dynamic_max_tokens}")

            # Check token limit before generation
            current_token_count = self._count_tokens_messages(current_messages)
            if current_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit before generation ({current_token_count}/{base_max_tokens}), stopping."
                    )
                break

            # Call litellm with tools
            response = await self._call_litellm_with_tools(
                current_messages,
                tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=dynamic_max_tokens,
                seed=seed,
                verbose=verbose,
                tool_choice=(forced_tool_choice if iteration == 1 else "auto"),
                **kwargs,
            )

            # Extract response message
            response_message = response.choices[0].message
            # OSS vLLM may return the completion in reasoning_content and leave content empty/None.
            rm_content = (response_message.content or "") if response_message else ""
            if not rm_content:
                rm_content = (
                    getattr(response_message, "reasoning_content", None)
                    or (
                        isinstance(
                            getattr(response_message, "provider_specific_fields", None), dict
                        )
                        and response_message.provider_specific_fields.get("reasoning_content")
                    )
                    or ""
                )

            # Append assistant's response to messages (contains tool_calls if any)
            current_messages.append(
                {
                    "role": "assistant",
                    "content": rm_content,
                    "tool_calls": getattr(response_message, "tool_calls", None),
                }
            )

            # Check if we've hit the token limit after generation
            new_token_count = self._count_tokens_messages(current_messages)
            if new_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit after generation ({new_token_count}/{base_max_tokens}), stopping."
                    )
                break

            # Check for tool calls
            tool_calls = getattr(response_message, "tool_calls", None)

            # Some OpenAI-compatible servers/models emit tool calls in the assistant content
            # (e.g., "<tool_call>tool?arg=val</tool_call>") but do not populate tool_calls.
            # Recover them so GLM-style tool usage still works.
            if not tool_calls:
                recovered = self._recover_tool_calls_from_content(
                    rm_content
                )
                if recovered:
                    tool_calls = recovered
                    # Update the last assistant message in-place so downstream token counting / logging
                    # sees the recovered tool calls.
                    current_messages[-1]["tool_calls"] = tool_calls
            else:
                # Normalize tool_calls into JSON-serializable dicts and ensure `function.arguments` is valid JSON.
                # This matters for some vLLM+GLM setups where tool_choice-forced calls return non-JSON arguments,
                # and vLLM later fails when parsing the chat history.
                normalized_calls: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        call = tc
                    else:
                        call = {
                            "id": getattr(tc, "id", "") or "",
                            "type": getattr(tc, "type", "function") or "function",
                            "function": {
                                "name": getattr(getattr(tc, "function", None), "name", "") or "",
                                "arguments": getattr(getattr(tc, "function", None), "arguments", "") or "",
                            },
                        }

                    fn = call.get("function", {}) or {}
                    fn_name = (fn.get("name") or "").strip()
                    arg_str = fn.get("arguments")
                    arg_str = "" if arg_str is None else str(arg_str)

                    def _coerce_to_json_args(tool_name: str, raw: str) -> str:
                        raw = (raw or "").strip()
                        if not raw:
                            return "{}"
                        try:
                            json.loads(raw)
                            return raw
                        except Exception:
                            pass

                        # Try to extract values for required keys from the raw string using the tool schema.
                        args: Dict[str, Any] = {}
                        tool = tools_by_name.get(tool_name)
                        required: List[str] = []
                        if tool is not None:
                            try:
                                schema = tool.to_openai_tool_schema()
                                required = (
                                    schema.get("function", {})
                                    .get("parameters", {})
                                    .get("required", [])
                                    or []
                                )
                            except Exception:
                                required = []

                        # Common fallback keys if schema isn't available.
                        if not required:
                            required = ["query", "url", "key", "input", "text", "content"]

                        # If the model returned a bare string (e.g., "alpha") and the tool has
                        # exactly one required field, treat the whole string as the value for that field.
                        # This is common in some vLLM/OpenAI-compatible servers when tool_choice is forced.
                        if len(required) == 1 and raw and all(k not in raw for k in [":", "=", "<arg_key>", "<arg_value>"]):
                            return json.dumps({required[0]: raw.strip().strip('"')}, ensure_ascii=False)

                        for k in required:
                            # Match patterns like: key 'alpha' / key="alpha" / key: alpha
                            m = re.findall(
                                rf"{re.escape(k)}[^A-Za-z0-9]+['\"]([^'\"]+)['\"]",
                                raw,
                                flags=re.IGNORECASE,
                            )
                            if not m:
                                m = re.findall(
                                    rf"{re.escape(k)}\\s*[:=]\\s*([A-Za-z0-9_\\-\\.]+)",
                                    raw,
                                    flags=re.IGNORECASE,
                                )
                            if m:
                                args[k] = m[-1].strip()

                        if not args:
                            # Last resort: keep the raw string under "query" so downstream still sees it.
                            args = {"query": raw}

                        return json.dumps(args, ensure_ascii=False)

                    if fn_name:
                        fn["arguments"] = _coerce_to_json_args(fn_name, arg_str)
                        call["function"] = fn
                    normalized_calls.append(call)

                tool_calls = normalized_calls
                current_messages[-1]["tool_calls"] = tool_calls

            if not tool_calls:
                # No tool calls, we're done
                if verbose:
                    print("No tool calls found, finishing.")
                break

            # Check if we've exceeded the maximum number of tool calls
            if tool_call_count + len(tool_calls) > max_tool_calls:
                if verbose:
                    print(
                        f"Exceeded maximum tool calls ({max_tool_calls}), creating error output."
                    )

                # Create error output for exceeding tool call limit
                error_output = ToolOutput(
                    output="Exceeded allowed tool call requests. Please stop calling tools and provide the final answer.",
                    error="Exceed allowed tool call requests. Please stop calling tools and provide the final answer.",
                    called=False,
                    timeout=False,
                    runtime=0,
                    call_id="",
                    raw_output=None,
                    tool_name="",
                )
                tool_calls_made.append(error_output)

                # Respond to all tool calls with error messages to maintain valid conversation format
                if include_tool_results:
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id", "")
                            tool_name = tool_call.get("function", {}).get("name", "")
                        else:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": error_output.output,
                            }
                        )
            else:
                if verbose:
                    print(f"Found {len(tool_calls)} tool call(s)")

                # Execute all tool calls in parallel
                tool_execution_tasks = []
                scheduled_calls: List[Tuple[Any, str, BaseTool]] = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_name = tool_call.get("function", {}).get("name", "")
                    else:
                        function_name = tool_call.function.name

                    if function_name not in tools_by_name:
                        if verbose:
                            print(
                                f"Warning: Tool '{function_name}' not found in available tools"
                            )
                        continue

                    tool = tools_by_name[function_name]

                    # Parse arguments
                    try:
                        if isinstance(tool_call, dict):
                            function_args = json.loads(
                                tool_call.get("function", {}).get("arguments", "{}")
                            )
                        else:
                            function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"Error parsing tool arguments: {e}")
                        continue

                    # Execute tool (via shared executor for concurrency + in-flight de-dupe + caching)
                    scheduled_calls.append((tool_call, function_name, tool))
                    tool_execution_tasks.append(
                        self._tool_executor.execute(
                            ToolRequest(tool=tool, tool_input=function_args)
                        )
                    )

                # Execute all tools in parallel
                tool_outputs = await asyncio.gather(
                    *tool_execution_tasks, return_exceptions=True
                )

                # Process tool outputs and add to messages
                for (tool_call, function_name, tool), tool_output in zip(
                    scheduled_calls, tool_outputs
                ):
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id", "")
                    else:
                        tool_call_id = tool_call.id
                    # Handle exceptions from tool execution
                    if isinstance(tool_output, Exception):
                        tool_output_str = f"Error executing tool: {str(tool_output)}"
                        error_output = ToolOutput(
                            output=tool_output_str,
                            error=str(tool_output),
                            called=False,
                            timeout=False,
                            runtime=0,
                            call_id=tool_call_id,
                            raw_output=None,
                            tool_name=function_name,
                        )
                        tool_calls_made.append(error_output)
                    else:
                        # Track successful tool call
                        tool_calls_made.append(tool_output)
                        tool_call_count += 1

                        # Get output content
                        # If you change this line, please do check the following lines that
                        # collectes and generates the generated_text.
                        tool_output_str = tool._format_output(tool_output)

                    # Add tool result to messages if include_tool_results
                    if include_tool_results:
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": tool_output_str,
                            }
                        )

                        if verbose:
                            print(
                                f"Tool '{function_name}' output: {tool_output_str[:200]}..."
                            )

            # Check token limit again after adding tool results
            final_token_count = self._count_tokens_messages(current_messages)
            if final_token_count >= base_max_tokens:
                if verbose:
                    print(
                        f"Hit token limit after tool execution ({final_token_count}/{base_max_tokens}), stopping."
                    )
                break

        # print(current_messages)
        # Create mapping from call_id to ToolOutput for efficient lookup
        tool_outputs_by_call_id = {
            output.call_id: output for output in tool_calls_made if output.call_id
        }

        # Calculate generated text (everything after original messages)
        generated_text = ""
        for msg in current_messages[original_message_count:]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # If content is empty but there are tool_calls, serialize them using tool parser format
                if not content and msg.get("tool_calls"):
                    tool_calls = msg.get("tool_calls", [])
                    serialized_calls = []
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            function = tool_call.get("function", {})
                            tool_name = function.get("name", "")
                            arguments_str = function.get("arguments", "")

                            # Parse JSON arguments
                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                arguments = {"query": arguments_str}

                            # Find the tool and use its parser to format the call
                            if tool_name in tools_by_name:
                                tool = tools_by_name[tool_name]
                                # Parser will auto-detect main parameter from schema
                                formatted_call = tool.tool_parser.format_tool_call(
                                    tool_name, arguments
                                )
                                serialized_calls.append(formatted_call)
                            else:
                                # Fallback to simple format if tool not found
                                serialized_calls.append(
                                    f'<tool name="{tool_name}">{arguments_str}</tool>'
                                )
                    content = "\n".join(serialized_calls)
                generated_text += content

            elif msg.get("role") == "tool":
                # Format tool output using tool parser
                tool_name = msg.get("name", "")
                content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id", "")

                # Find the tool and use its parser to format the result
                if tool_name in tools_by_name:
                    tool = tools_by_name[tool_name]
                    # Extract the actual ToolOutput from the saved tool calls
                    tool_output = tool_outputs_by_call_id.get(tool_call_id)
                    if tool_output is not None:
                        formatted_output = tool.format_result(tool_output)
                    else:
                        formatted_output = content
                    # They are effectively the same in this stage.
                    # formatted_output = content
                    generated_text += f"\n{formatted_output}"
                else:
                    # Fallback to simple format
                    generated_text += f"\n<tool_output>{content}</tool_output>"

        final_token_count = self._count_tokens_messages(current_messages)
        trace_model_input = str(os.environ.get("DR_TRACE_MODEL_INPUT", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        return GenerateWithToolsOutput(
            tool_calls=tool_calls_made,
            generated_text=generated_text,
            total_tokens=final_token_count,
            tool_call_count=tool_call_count,
            stopped_reason=(
                "max_tokens" if final_token_count >= base_max_tokens else "natural"
            ),
            model_input=({"messages": current_messages, "tools": tools} if trace_model_input else None),
        )

    async def _call_litellm_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
        verbose: bool,
        **kwargs,
    ) -> Any:
        """Call litellm completion API with tools"""
        config = self.generation_config

        # Build parameters
        tool_choice = kwargs.pop("tool_choice", "auto")
        params = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": (
                temperature if temperature is not None else config.temperature
            ),
            "top_p": top_p if top_p is not None else config.top_p,
            "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        }

        # Add seed if provided
        if seed is not None or config.seed is not None:
            params["seed"] = seed if seed is not None else config.seed

        # Add API credentials if available
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url
            # Force LiteLLM to treat this as an OpenAI-compatible endpoint without rewriting the model
            # name (vLLM expects the served model name verbatim).
            params["custom_llm_provider"] = "openai"

        # Add any additional kwargs
        params.update(kwargs)

        if verbose:
            print(f"Calling litellm with {len(tools)} tools")

        response = await litellm.acompletion(**params)
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
    )
    async def _generate_single_response_commercial_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        verbose: bool = False,
        include_reasoning: bool = False,
        **kwargs,
    ) -> str:
        """Generate a single response from commercial API models using chat completion API with tenacity retry

        This method includes special handling for:
        1. Stop tokens: When the API stops due to hitting a stop token, it typically doesn't include that
           stop token in the response content. However, tool calling logic often needs to see those stop
           tokens (e.g., tool end tags like '</search>'). This method automatically detects when generation
           stopped due to a stop token and adds the appropriate stop token back to the content.
        2. Reasoning content: For models that support reasoning (like OpenAI o1), this method can optionally
           include the model's internal reasoning process wrapped in <think>...</think> tags.
        """

        config = self.generation_config

        # Build parameters for LiteLLM
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else config.temperature
            ),
            "top_p": top_p if top_p is not None else config.top_p,
            "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
            "stop": stop_sequences,
        }

        # Add seed if provided
        if seed is not None or config.seed is not None:
            params["seed"] = seed if seed is not None else config.seed

        # Add API credentials if available
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url
            # Force LiteLLM to treat this as an OpenAI-compatible endpoint without rewriting the model
            # name (vLLM expects the served model name verbatim).
            params["custom_llm_provider"] = "openai"

        # Safety: this "commercial api" path does NOT pass OpenAI-native tools.
        # If callers accidentally pass tool_choice/tools through kwargs (e.g., via shared AgentInterface),
        # OpenAI-compatible servers will reject the request: "When using tool_choice, tools must be set."
        kwargs.pop("tool_choice", None)
        kwargs.pop("tools", None)

        # Add any additional kwargs
        params.update(kwargs)
        # print(params)

        try:
            response = await litellm.acompletion(**params)
            original_content = response.choices[0].message.content or ""
            content = original_content

            # Extract reasoning content if available and requested
            reasoning_content = None
            if include_reasoning:
                message = response.choices[0].message
                # Try to get reasoning content from different possible locations
                reasoning_content = getattr(message, "reasoning_content", None) or (
                    hasattr(message, "provider_specific_fields")
                    and isinstance(message.provider_specific_fields, dict)
                    and message.provider_specific_fields.get("reasoning_content")
                )

                if reasoning_content and verbose:
                    print(
                        f"Found reasoning content: {repr(reasoning_content[:100])}..."
                    )

            # Some OpenAI-compatible OSS servers put the *actual completion* in reasoning_content
            # while leaving message.content empty. In that case, treat reasoning_content as content
            # (do NOT wrap it in <think>...</think>, because downstream postprocessing strips <think>).
            if reasoning_content and not original_content.strip():
                content = str(reasoning_content)
            # Otherwise, if we truly have separate reasoning, optionally prepend it.
            elif reasoning_content:
                content = f"<think>{reasoning_content}</think>\n{content}"

            # If the generation finished due to a stop token, add it back to the content
            # This is important for tool calling logic that expects to see the stop tokens
            finish_reason = getattr(response.choices[0], "finish_reason", None)

            if verbose:
                print(f"API response finish_reason: {finish_reason}")
                print(f"Stop sequences: {stop_sequences}")
                print(f"Original content: {repr(original_content)}")

            if finish_reason == "stop" and stop_sequences and len(stop_sequences) == 1:
                # Simple case: only one stop token, just append it
                stop_token = stop_sequences[0]
                if stop_token and not content.endswith(stop_token):
                    content += stop_token
                    if verbose:
                        print(f"Added single stop token: {repr(stop_token)}")
                        print(f"Updated content: {repr(content)}")

            elif finish_reason == "stop" and stop_sequences and len(stop_sequences) > 1:
                # Multiple stop tokens - handle based on parser type
                parser_type = self._get_tool_parser_type()

                if parser_type == "unified":
                    # For unified parser, simply add the </tool> tag
                    stop_token = "</tool>"
                    if not content.endswith(stop_token):
                        content += stop_token
                        if verbose:
                            print(
                                f"Added unified parser stop token: {repr(stop_token)}"
                            )
                            print(f"Updated content: {repr(content)}")

                elif parser_type == "legacy":
                    # For legacy parser, use the existing detection logic
                    added_stop_token = None
                    for stop_token in stop_sequences:
                        if stop_token and not content.endswith(stop_token):
                            # Check if adding this stop token would create a valid tool call
                            test_content = content + stop_token
                            if self._find_first_tool_call(test_content):
                                content = test_content
                                added_stop_token = stop_token
                                break

                    if verbose and added_stop_token:
                        print(f"Added detected stop token: {repr(added_stop_token)}")
                        print(f"Updated content: {repr(content)}")
                    elif verbose:
                        print("Could not determine which stop token was hit")

                else:
                    # Fallback to existing logic for unknown parser types
                    if verbose:
                        print(
                            f"Unknown parser type: {parser_type}, using fallback logic"
                        )
                    added_stop_token = None
                    for stop_token in stop_sequences:
                        if stop_token and not content.endswith(stop_token):
                            test_content = content + stop_token
                            if self._find_first_tool_call(test_content):
                                content = test_content
                                added_stop_token = stop_token
                                break

            return content
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Request timed out after {config.timeout} seconds"
            )
        except Exception as e:
            print(f"API call failed: {e}")
            raise Exception(f"API call failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
    )
    async def _generate_single_response_vllm(
        self,
        prompt: str,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a single response from vLLM models using completion API with tenacity retry"""

        config = self.generation_config

        # Use provided parameters or fall back to config defaults
        params = {
            "model": (
                f"hosted_vllm/{self.model_name}"
                if not self.model_name.startswith("hosted_vllm/")
                else self.model_name
            ),
            "prompt": prompt,
            "temperature": (
                temperature if temperature is not None else config.temperature
            ),
            "top_p": top_p if top_p is not None else config.top_p,
            "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
            "stop": stop_sequences,
        }

        # Add vLLM-specific parameters
        extra_body = {}
        if top_k is not None or config.top_k != 1:
            extra_body["top_k"] = top_k if top_k is not None else config.top_k
        if repetition_penalty is not None or config.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = (
                repetition_penalty
                if repetition_penalty is not None
                else config.repetition_penalty
            )

        if extra_body:
            params["extra_body"] = extra_body

        # Add seed if provided
        if seed is not None or config.seed is not None:
            params["seed"] = seed if seed is not None else config.seed

        # Add API credentials if available
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        # Add any additional kwargs
        params.update(kwargs)
        params["include_stop_str_in_output"] = True

        try:
            response = await litellm.atext_completion(**params)
            return (
                response.choices[0].text
                if hasattr(response.choices[0], "text")
                else response.choices[0].message.content or ""
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Request timed out after {config.timeout} seconds"
            )
        except Exception as e:
            print(f"API call failed: {e}")
            raise Exception(f"API call failed: {e}")

    # Keep the old method name for backward compatibility, but delegate to vLLM implementation
    async def _generate_single_response(
        self,
        prompt: str,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a single response from the model using completion API with tenacity retry"""

        return await self._generate_single_response_vllm(
            prompt,
            stop_sequences,
            temperature,
            top_p,
            max_tokens,
            top_k,
            repetition_penalty,
            seed,
            **kwargs,
        )

    def __enter__(self):
        """Enter the context and set this client as the current one"""
        self._context_token = _llm_tool_client_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and reset to previous client"""
        _llm_tool_client_context.reset(self._context_token)
        # Optionally cleanup resources here if needed
        return False  # Don't suppress exceptions
