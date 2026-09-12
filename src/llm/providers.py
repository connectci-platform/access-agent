"""LLM provider abstraction supporting OpenAI, vLLM, and custom endpoints."""

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import settings

# Wire-protocol sentinel, NOT user-facing text. Stands in for a response whose
# visible content stripped to nothing, so the assistant turn is non-empty: UKY's
# vLLM rejects a blank assistant turn on the next request in the loop ("No user
# query found in messages"), failing the whole turn with a 400.
#
# tool_calling_loop translates this into a real apology before it can reach a
# user, and src/eval/runner.py treats it as a failed turn so a reasoning-only
# response stays countable instead of being judged as a bad answer. Anything
# that reads a final answer must recognise it — grep before changing the value.
EMPTY_ANSWER_SENTINEL = "\u0000__no_answer__"


def is_empty_answer(text: str | None) -> bool:
    """Whether a final answer is the sentinel rather than real content.

    Not an equality test: the streaming path forwards post-</think> whitespace
    chunks verbatim (inter-word spacing arrives as its own chunk), so a trailing
    "\n\n" before the model stops yields "\n\n" + sentinel. An == check misses
    that and the marker reaches the user.
    """
    return text is not None and text.strip() == EMPTY_ANSWER_SENTINEL


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# Non-greedy paired-block match. DOTALL so newlines inside the trace match.
_PAIRED_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Same match with the trace captured, for reasoning capture.
_PAIRED_THINK_INNER_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _record_model_reasoning(reasoning: str) -> None:
    """Stash one model call's stripped reasoning in turn_capture.

    Import is lazy: the agent package imports this module (nodes → llm), so a
    module-level import of agent.turn_capture would be circular.
    """
    from ..agent.turn_capture import record_model_reasoning

    record_model_reasoning(reasoning)


def _strip_think_block(content: str) -> str:
    """Strip reasoning trace from reasoning-model output (see _split_think_block)."""
    return _split_think_block(content)[0]


def _split_think_block(content: str) -> tuple[str, str]:
    """Split reasoning-model output into (answer, reasoning trace).

    On UKY's vLLM the opening ``<think>`` is consumed by the Qwen chat
    template, so the visible stream is shaped ``[reasoning prose]</think>
    [answer]`` — no leading ``<think>`` tag. Defensive coverage is also
    needed for models that DO emit paired ``<think>...</think>`` blocks
    (e.g. mid-answer reflections) and for the truncation case.

    Order matters:

    1. Strip any properly-paired ``<think>...</think>`` blocks (handles
       models that emit the open tag, or a paired block mid-answer).
    2. If an unpaired ``<think>`` remains, drop everything from there to
       end of content (truncated mid-reasoning).
    3. After (1) and (2), if a bare ``</think>`` remains, partition at
       the first occurrence and return what's after — that's the Qwen
       no-open-tag shape, and it's the most common case in production.
    4. Otherwise content is already a plain answer; return as-is.

    Trade-off: a user who asks about the literal ``</think>`` token and
    gets a response quoting it will see the prose before ``</think>``
    stripped. This false positive is rare in ACCESS-CI question-answering
    vs the 100% true-positive rate of stripping reasoning traces on
    every Qwen response.

    The reasoning side of the split feeds the turn report's
    ``payload.reasoning`` (the dashboard's Reasoning tab); empty string
    when the content carried no trace.
    """
    parts = [p.strip() for p in _PAIRED_THINK_INNER_RE.findall(content) if p.strip()]
    content = _PAIRED_THINK_RE.sub("", content)
    open_idx = content.find(_THINK_OPEN)
    if open_idx != -1:
        tail = content[open_idx + len(_THINK_OPEN) :].strip()
        if tail:
            parts.append(tail)
        content = content[:open_idx]
    if _THINK_CLOSE in content:
        before, _, after = content.partition(_THINK_CLOSE)
        if before.strip():
            parts.append(before.strip())
        return after.lstrip(), "\n".join(parts)
    return content.strip(), "\n".join(parts)


def _strip_generations(result: ChatResult) -> None:
    """Strip ``</think>`` blocks from each generation's message in place.

    The stripped reasoning is recorded into turn_capture so the turn report
    can carry it (one entry per model call).
    """
    for gen in result.generations:
        msg = getattr(gen, "message", None)
        if msg is None:
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            answer, reasoning = _split_think_block(content)
            # A reasoning-only response ("...thinking...</think>" with nothing
            # after) strips to "". Assigning that leaves a blank assistant turn
            # in the thread, and UKY's vLLM rejects the next request with
            # "No user query found in messages" — killing the whole turn with a
            # 400 rather than degrading. Substitute a neutral placeholder so the
            # turn stays valid.
            #
            # The reasoning itself is NOT used as the fallback. A response
            # truncated mid-trace (max_tokens hit before </think>) also strips
            # to "", and handing that buffer to the user is the chain-of-thought
            # leak closed in 1d66cff. The two cases are not reliably
            # distinguishable here, so neither one leaks.
            #
            # Tool-calling turns keep their empty content: tool_calls carry the
            # intent, and a placeholder would render as spurious prose.
            if not answer and not getattr(msg, "tool_calls", None):
                msg.content = EMPTY_ANSWER_SENTINEL
            else:
                msg.content = answer
            if reasoning:
                _record_model_reasoning(reasoning)


class _ThinkStripState:
    """Cross-chunk state machine for stripping reasoning traces.

    Mirrors :func:`_strip_think_block` for the streaming path. Buffers
    incoming chunks until either:

    - A paired ``<think>...</think>`` block can be removed via regex
      (rare in Qwen stream but defensive for models that emit the open
      tag, or paired blocks mid-answer).
    - A bare ``</think>`` arrives signaling end of the Qwen-shape
      reasoning trace.

    Once a close has been processed, ``seen_close`` flips to True and
    subsequent chunks pass through unchanged. Truncation handling lives
    in :meth:`flush` — an unpaired ``<think>`` tail is dropped before
    the buffered remainder (if any) is emitted, covering the
    truncation-with-preamble case while still letting non-reasoning
    streams flush their content.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._seen_close = False
        self.reasoning = ""
        # Whether any meaningful answer text has been emitted. A reasoning-only
        # response closes the trace and emits nothing; flush() uses this to
        # emit the sentinel rather than yield an empty message.
        self._emitted_any = False
        # Set when a chunk carries a tool-call delta; suppresses the sentinel.
        self._saw_tool_call = False

    @property
    def seen_close(self) -> bool:
        return self._seen_close

    def mark_tool_call_seen(self) -> None:
        """Record that this response carries a tool call.

        A tool-calling turn legitimately has no prose — tool_calls carry the
        intent — so flush() must not append the sentinel, which would surface as
        spurious text beside the call. Mirrors the tool_calls check the
        non-streaming path makes in _strip_generations.
        """
        self._saw_tool_call = True

    def mark_emitted(self) -> None:
        """Record that answer text reached the consumer.

        Needed because ``_apply_strip_to_chunk`` short-circuits post-``</think>``
        chunks straight to the consumer without calling :meth:`feed`, so the
        state machine would otherwise never learn that an answer was streamed.
        """
        self._emitted_any = True

    def feed(self, content: str) -> str | None:
        """Feed one chunk's text; return what should be emitted now."""
        if self._seen_close:
            if content.strip():
                self._emitted_any = True
            return content
        self._buffer += content
        # Capture paired-block traces before stripping them.
        parts = [p.strip() for p in _PAIRED_THINK_INNER_RE.findall(self._buffer) if p.strip()]
        new_buffer, n_subs = _PAIRED_THINK_RE.subn("", self._buffer)
        has_close = _THINK_CLOSE in new_buffer
        if n_subs == 0 and not has_close:
            # No reasoning-done signal yet — keep buffering.
            return None
        self._seen_close = True
        self._buffer = ""
        if has_close:
            # Bare </think> remains (Qwen no-open-tag case) — partition.
            before, _, after = new_buffer.partition(_THINK_CLOSE)
            if before.strip():
                parts.append(before.strip())
            self.reasoning = "\n".join(parts)
            out = after.lstrip() or None
            if out:
                self._emitted_any = True
            return out
        self.reasoning = "\n".join(parts)
        # Only paired blocks were present. ``lstrip`` only — a trailing
        # space here connects to the next chunk; ``strip`` would eat it.
        out = new_buffer.lstrip() or None
        if out:
            self._emitted_any = True
        return out

    def flush(self) -> str | None:
        """End-of-stream: emit remaining safe content.

        - Already closed → nothing more to emit.
        - No close ever arrived → defer to :func:`_strip_think_block`
          which handles the truncation case (drops any unpaired
          ``<think>`` tail) and the non-reasoning fall-through (the
          buffer IS the answer; emit it). Full ``strip`` is fine here
          — we're at end-of-stream, no more chunks to connect to.
        """
        if self._seen_close:
            # Reasoning-only stream: the trace closed but no answer followed, so
            # nothing was emitted and the aggregated message would be empty.
            # UKY's vLLM rejects a blank assistant turn on the next request with
            # "No user query found in messages", failing the turn with a 400.
            # Emit a neutral placeholder, never the buffered trace — that would
            # be the chain-of-thought leak closed in 1d66cff.
            if not self._emitted_any and not self._saw_tool_call:
                self._emitted_any = True
                return EMPTY_ANSWER_SENTINEL
            return None
        answer, reasoning = _split_think_block(self._buffer)
        if reasoning:
            self.reasoning = reasoning
        self._buffer = ""
        if answer:
            self._emitted_any = True
            return answer
        # Truncated mid-trace (max_tokens hit before </think>): the trace is
        # correctly dropped, leaving nothing. Same empty-turn problem as the
        # reasoning-only case, so the same sentinel — never the buffer.
        if not self._emitted_any and not self._saw_tool_call:
            self._emitted_any = True
            return EMPTY_ANSWER_SENTINEL
        return None


def _replace_chunk_content(chunk: ChatGenerationChunk, new_content: str) -> ChatGenerationChunk:
    """Return a chunk with text content replaced; tool-call deltas et al preserved."""
    msg = chunk.message
    new_msg = AIMessageChunk(
        content=new_content,
        additional_kwargs=dict(getattr(msg, "additional_kwargs", {}) or {}),
        response_metadata=dict(getattr(msg, "response_metadata", {}) or {}),
        tool_call_chunks=list(getattr(msg, "tool_call_chunks", []) or []),
        usage_metadata=getattr(msg, "usage_metadata", None),
        id=getattr(msg, "id", None),
    )
    return ChatGenerationChunk(
        message=new_msg,
        generation_info=chunk.generation_info,
    )


def _chunk_carries_non_text_payload(chunk: ChatGenerationChunk) -> bool:
    """True if the chunk has tool-call deltas or extra_kwargs we must not drop."""
    msg = chunk.message
    if getattr(msg, "tool_call_chunks", None):
        return True
    return bool(getattr(msg, "additional_kwargs", None))


class _StrippingChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that strips reasoning-model ``</think>`` blocks.

    Used by ``OpenAICompatibleProvider`` so every response from a thinking
    model (e.g. Qwen3 via UKY's vLLM endpoint) arrives with the reasoning
    trace removed — keeping eval, telemetry, the loop's own message thread,
    and the frontend on clean content.

    Three paths are overridden so the strip fires regardless of how
    ``BaseChatModel`` decides to serve the request:

    * ``_generate`` / ``_agenerate`` — non-streaming path (sync / async).
    * ``_astream`` — streaming path. With ``streaming=True``,
      ``BaseChatModel._agenerate_with_cache`` routes directly through
      ``_astream`` and bypasses ``_agenerate``; aggregated chunks then become
      the result of ``ainvoke``. Stripping at the chunk level keeps both
      streaming consumers and ``ainvoke`` callers seeing clean content.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        _strip_generations(result)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        _strip_generations(result)
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        state = _ThinkStripState()
        for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield from _apply_strip_to_chunk(chunk, state)
        tail = _flush_strip_state(state)
        if tail is not None:
            yield tail
        if state.reasoning:
            _record_model_reasoning(state.reasoning)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        state = _ThinkStripState()
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            for out in _apply_strip_to_chunk(chunk, state):
                yield out
        tail = _flush_strip_state(state)
        if tail is not None:
            yield tail
        if state.reasoning:
            _record_model_reasoning(state.reasoning)


def _apply_strip_to_chunk(
    chunk: ChatGenerationChunk,
    state: _ThinkStripState,
) -> list[ChatGenerationChunk]:
    """Apply the cross-chunk strip state to one chunk; return chunks to yield.

    Returns 0 or 1 chunks. Drops content-only chunks that fall inside a
    reasoning trace; preserves chunks that carry tool-call deltas or other
    non-text payloads (with their text emptied if still buffered).

    Once the state machine has consumed the first ``</think>`` (or a
    paired ``<think>...</think>`` block), subsequent chunks pass through
    unchanged — Qwen emits one reasoning trace per response, no more.
    """
    msg = chunk.message
    raw = msg.content if isinstance(msg.content, str) else ""

    # Deliberately narrower than _chunk_carries_non_text_payload, which is true
    # for ANY additional_kwargs: only a real tool call should suppress the
    # sentinel. Otherwise an unrelated kwarg (a provider adding "refusal", say)
    # silently disables the empty-turn guard and the 400 comes back.
    if getattr(chunk.message, "tool_call_chunks", None) or (
        getattr(chunk.message, "additional_kwargs", None) or {}
    ).get("function_call"):
        state.mark_tool_call_seen()

    if state.seen_close:
        # Post-trace chunks pass through without re-entering feed(), so record
        # here that real answer text reached the consumer — flush() relies on it
        # to tell "answered normally" from "reasoning only, nothing emitted".
        # strip(): a whitespace-only tail is a blank turn to vLLM, so it must not
        # count as an answer or flush() would skip the sentinel.
        if raw.strip():
            state.mark_emitted()
        return [chunk]

    if not raw:
        # Tool-call-only chunks (empty content) pass through unchanged.
        return [chunk]

    emit = state.feed(raw)
    if emit is None:
        if _chunk_carries_non_text_payload(chunk):
            return [_replace_chunk_content(chunk, "")]
        return []
    return [_replace_chunk_content(chunk, emit)]


def _flush_strip_state(state: _ThinkStripState) -> ChatGenerationChunk | None:
    """Emit a final chunk if the stream ended without ever seeing ``</think>``."""
    leftover = state.flush()
    if leftover is None:
        return None
    return ChatGenerationChunk(message=AIMessageChunk(content=leftover))


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chat_model(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        enable_thinking: bool | None = None,
    ) -> BaseChatModel:
        """Get a chat model instance.

        Args:
            model_name: Override the default model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            enable_thinking: Optional override for reasoning-model thinking mode.
                When set, the value is sent to the server as
                ``chat_template_kwargs.enable_thinking`` in the request body.
                When ``None`` (default), nothing is sent and the server uses
                its own default. Only meaningful for reasoning models served
                via OpenAI-compatible endpoints (e.g. Qwen3 on vLLM); ignored
                by providers that don't support the parameter.

        Returns:
            A LangChain chat model instance.
        """


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API."""

    def __init__(self, api_key: str, default_model: str = "gpt-4o"):
        self.api_key = SecretStr(api_key)
        self.default_model = default_model

    def get_chat_model(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        enable_thinking: bool | None = None,
    ) -> BaseChatModel:
        # enable_thinking is a vLLM/Qwen concept; OpenAI's API doesn't honor it.
        # Accepted for signature parity, silently ignored.
        del enable_thinking
        return ChatOpenAI(
            model=model_name or self.default_model,
            api_key=self.api_key,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs (vLLM, LocalAI, etc.).

    This works with any server that implements the OpenAI chat completions API,
    including:
    - vLLM with --served-model-name
    - LocalAI
    - Ollama with OpenAI compatibility
    - Any custom endpoint
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",  # Many local servers don't require a key
        default_model: str = "default",
    ):
        self.base_url = base_url
        self.api_key = SecretStr(api_key)
        self.default_model = default_model

    def get_chat_model(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        enable_thinking: bool | None = None,
    ) -> BaseChatModel:
        extra_body: dict[str, Any] = {}
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        return _StrippingChatOpenAI(
            model=model_name or self.default_model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            extra_body=extra_body or None,
        )


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider.

    Returns the appropriate provider based on LLM_PROVIDER setting:
    - openai: Uses OpenAI API
    - vllm: Uses vLLM server with OpenAI-compatible API
    - access_ai: Uses ACCESS AI endpoint

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured.
    """
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for openai provider")
        return OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            default_model=settings.OPENAI_MODEL,
        )

    if provider == "vllm":
        return OpenAICompatibleProvider(
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.VLLM_API_KEY or "dummy",
            default_model=settings.VLLM_MODEL_NAME,
        )

    if provider == "access_ai":
        if not settings.ACCESS_AI_API_KEY:
            raise ValueError("ACCESS_AI_API_KEY is required for access_ai provider")
        return OpenAICompatibleProvider(
            base_url=settings.ACCESS_AI_BASE_URL,
            api_key=settings.ACCESS_AI_API_KEY,
            default_model="access-llama",
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def active_model_name() -> str:
    """The model id actually used at runtime, mirroring get_llm_provider's
    provider→model mapping so reporting records the real model, not a default.
    """
    provider = settings.LLM_PROVIDER
    if provider == "openai":
        return settings.OPENAI_MODEL
    if provider == "vllm":
        return settings.VLLM_MODEL_NAME
    if provider == "access_ai":
        return "access-llama"
    return provider or "unknown"


def get_llm(
    model_name: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    enable_thinking: bool | None = None,
) -> BaseChatModel:
    """Get a configured LLM instance.

    Convenience function that gets the provider and returns a model.

    Args:
        model_name: Override the default model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        enable_thinking: Optional override for reasoning-model thinking mode.
            See :meth:`LLMProvider.get_chat_model` for details. No call site
            currently sets this; it is exposed for future fast-path
            experiments where a node may want to opt out of reasoning.

    Returns:
        A LangChain chat model instance.
    """
    provider = get_llm_provider()
    return provider.get_chat_model(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
