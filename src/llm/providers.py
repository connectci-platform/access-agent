"""LLM provider abstraction supporting OpenAI, vLLM, and custom endpoints."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import settings

_THINK_CLOSE = "</think>"


def _strip_think_block(content: str) -> str:
    """Remove a leading reasoning trace terminated by ``</think>``.

    Reasoning models (Qwen3, DeepSeek-R1, etc.) emit chain-of-thought inline
    in ``content`` before the user-visible answer. Their docs say to strip
    the trace before re-sending the assistant message in conversation
    history — replaying the trace back at the model on subsequent turns is
    out-of-distribution input. No-op when ``</think>`` is absent.
    """
    if _THINK_CLOSE not in content:
        return content
    _, _, after = content.partition(_THINK_CLOSE)
    return after.lstrip()


def _strip_generations(result: ChatResult) -> None:
    """Strip ``</think>`` blocks from each generation's message in place."""
    for gen in result.generations:
        msg = getattr(gen, "message", None)
        if msg is None:
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            msg.content = _strip_think_block(content)


class _ThinkStripState:
    """Cross-chunk state machine for stripping a leading ``</think>`` block.

    Reasoning-model output arrives as a stream of text chunks where the
    ``<think>...</think>`` reasoning trace may span multiple chunks (and the
    closing tag itself may be split across two chunks). This holds the buffer
    and the "have we seen the close tag yet" flag across calls.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._seen_close = False

    @property
    def seen_close(self) -> bool:
        return self._seen_close

    def feed(self, content: str) -> str | None:
        """Feed one chunk's text content; return what should be emitted now.

        - ``None`` → suppress this chunk entirely (we're still buffering).
        - ``""``   → emit a placeholder chunk (e.g. so tool-call deltas on the
          same chunk still pass through with empty text).
        - ``str``  → emit this text (the post-``</think>`` remainder).

        Once ``</think>`` is observed the state flips; subsequent calls
        passthrough the input unchanged.
        """
        if self._seen_close:
            return content
        self._buffer += content
        if _THINK_CLOSE not in self._buffer:
            return None
        _, _, after = self._buffer.partition(_THINK_CLOSE)
        self._seen_close = True
        self._buffer = ""
        return after.lstrip()

    def flush(self) -> str | None:
        """End-of-stream: if no ``</think>`` ever arrived, hand back the buffer.

        Models that don't emit a reasoning trace (or emit one without the
        close tag at all) shouldn't have their entire response swallowed.
        """
        if self._seen_close:
            return None
        leftover = self._buffer
        self._buffer = ""
        return leftover or None


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


def _apply_strip_to_chunk(
    chunk: ChatGenerationChunk,
    state: _ThinkStripState,
) -> list[ChatGenerationChunk]:
    """Apply the cross-chunk strip state to one chunk; return chunks to yield.

    Returns 0 or 1 chunks. Drops content-only chunks that fall inside the
    reasoning trace; preserves chunks that carry tool-call deltas or other
    non-text payloads (with their text emptied if still pre-``</think>``).
    """
    msg = chunk.message
    raw = msg.content if isinstance(msg.content, str) else ""

    if state.seen_close:
        return [chunk]

    if not raw:
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
