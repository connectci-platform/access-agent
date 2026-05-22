"""Tests for LLM provider abstraction — Qwen `</think>` strip + enable_thinking knob."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from src.llm.providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    _strip_generations,
    _strip_think_block,
    _StrippingChatOpenAI,
)


def _chunk(content: str = "", **kwargs: Any) -> ChatGenerationChunk:
    """Build a streaming chunk from string content (+ optional AIMessageChunk fields)."""
    return ChatGenerationChunk(message=AIMessageChunk(content=content, **kwargs))


def _fake_astream_factory(chunks: list[ChatGenerationChunk]):
    """Patch target for ``ChatOpenAI._astream`` that yields a fixed chunk sequence."""

    async def _astream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in chunks:
            yield chunk

    return _astream


def _fake_stream_factory(chunks: list[ChatGenerationChunk]):
    """Patch target for ``ChatOpenAI._stream``."""

    def _stream(self: Any, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        yield from chunks

    return _stream


async def _collect_astream(model: _StrippingChatOpenAI) -> list[ChatGenerationChunk]:
    """Run model._astream with throwaway args and return the emitted chunks."""
    out: list[ChatGenerationChunk] = []
    async for chunk in model._astream(messages=[], stop=None, run_manager=None):
        out.append(chunk)
    return out


def _aggregate_text(chunks: list[ChatGenerationChunk]) -> str:
    return "".join(c.message.content for c in chunks if isinstance(c.message.content, str))


class TestStripThinkBlock:
    def test_strips_leading_think_block(self):
        raw = "<think>I should look up X.</think>\n\nThe answer is foo."
        assert _strip_think_block(raw) == "The answer is foo."

    def test_no_think_tag_returns_unchanged(self):
        raw = "The answer is foo."
        assert _strip_think_block(raw) == "The answer is foo."

    def test_empty_content_unchanged(self):
        assert _strip_think_block("") == ""

    def test_strips_at_first_close_tag_when_multiple(self):
        # Defensive: model emits two `</think>` tags; we cut at the first.
        # Anything between the first and second close tag is treated as answer.
        raw = "<think>outer reasoning</think>middle</think>final answer"
        assert _strip_think_block(raw) == "middle</think>final answer"

    def test_strips_when_close_tag_at_start(self):
        # Bare close tag with no open is treated as a degenerate trace.
        raw = "</think>just the answer"
        assert _strip_think_block(raw) == "just the answer"

    def test_preserves_internal_whitespace_after_strip(self):
        raw = "<think>thinking...</think>\n\nLine 1\n\nLine 2"
        assert _strip_think_block(raw) == "Line 1\n\nLine 2"

    def test_prose_mentioning_close_tag_is_not_stripped(self):
        # The user asked what </think> means. The model's prose contains
        # `</think>` mid-message but the message does not open with a
        # think tag — leave it intact.
        raw = "The closing tag is `</think>`, used to terminate reasoning."
        assert _strip_think_block(raw) == raw

    def test_truncated_mid_think_returns_empty(self):
        # max_tokens cut the reasoning off before `</think>` was emitted.
        # Never leak the buffered chain-of-thought.
        raw = "<think>I should consider every possibility before answering, starting with"
        assert _strip_think_block(raw) == ""


class TestStripGenerations:
    def test_strips_each_generation_message(self):
        result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="<think>reason A</think>answer A")),
                ChatGeneration(message=AIMessage(content="<think>reason B</think>answer B")),
            ]
        )
        _strip_generations(result)
        assert result.generations[0].message.content == "answer A"
        assert result.generations[1].message.content == "answer B"

    def test_skips_non_string_content(self):
        # AIMessage.content can be list[str | dict] for multimodal responses; leave alone.
        multimodal: list[str | dict[Any, Any]] = [{"type": "text", "text": "hi"}]
        result = ChatResult(generations=[ChatGeneration(message=AIMessage(content=multimodal))])
        _strip_generations(result)
        assert result.generations[0].message.content == multimodal


class TestOpenAICompatibleProviderEnableThinking:
    def test_enable_thinking_none_omits_extra_body(self):
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = provider.get_chat_model()
        # Default: no thinking override → no extra_body present.
        assert getattr(model, "extra_body", None) is None

    def test_enable_thinking_false_sets_chat_template_kwargs(self):
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model(enable_thinking=False))
        assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_enable_thinking_true_sets_chat_template_kwargs(self):
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model(enable_thinking=True))
        assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_returns_stripping_subclass(self):
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = provider.get_chat_model()
        assert isinstance(model, _StrippingChatOpenAI)


class TestStrippingChatOpenAIAstream:
    """Streaming-path strip: BaseChatModel routes through _astream with streaming=True,
    so the strip has to operate on a chunk sequence rather than a finished ChatResult.
    Each test patches the parent ChatOpenAI._astream to inject a fixture stream and
    verifies that the override emits the post-</think> tail and nothing before it.
    """

    @pytest.mark.asyncio
    async def test_strips_think_block_split_across_chunks(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [
            _chunk("<think>I should look up X."),
            _chunk("\n</think>\n\n"),
            _chunk("The answer is foo."),
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        text = _aggregate_text(chunks)
        assert "</think>" not in text
        assert "I should look up X" not in text
        assert text.strip() == "The answer is foo."

    @pytest.mark.asyncio
    async def test_close_tag_split_across_two_chunks(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [
            _chunk("<think>thinking</thi"),
            _chunk("nk>real answer"),
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        assert _aggregate_text(chunks) == "real answer"

    @pytest.mark.asyncio
    async def test_cuts_at_first_close_tag_when_multiple(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [_chunk("<think>outer reasoning</think>middle</think>final answer")]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        assert _aggregate_text(chunks) == "middle</think>final answer"

    @pytest.mark.asyncio
    async def test_no_think_block_emits_buffer_at_end(self) -> None:
        # Non-thinking model output (or thinking disabled) shouldn't be swallowed.
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [_chunk("just"), _chunk(" the answer")]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        assert _aggregate_text(chunks) == "just the answer"

    @pytest.mark.asyncio
    async def test_tool_call_chunks_pass_through_after_close_tag(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        tool_chunk_msg = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "search_announcements", "args": "{}", "id": "c1", "index": 0}
            ],
        )
        fixture = [
            _chunk("<think>let me look</think>"),
            ChatGenerationChunk(message=tool_chunk_msg),
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        # Tool-call chunk preserved unchanged.
        tool_chunks_seen = [c for c in chunks if getattr(c.message, "tool_call_chunks", None)]
        assert len(tool_chunks_seen) == 1
        assert tool_chunks_seen[0].message.tool_call_chunks[0]["name"] == "search_announcements"
        assert "</think>" not in _aggregate_text(chunks)

    @pytest.mark.asyncio
    async def test_tool_call_chunk_during_buffering_keeps_payload(self) -> None:
        # Defensive: a chunk arrives carrying both pre-</think> text AND a tool-call
        # delta. The text is suppressed but the tool-call payload still surfaces.
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        mixed_msg = AIMessageChunk(
            content="<think>thinking",
            tool_call_chunks=[{"name": "tool_x", "args": "{}", "id": "c1", "index": 0}],
        )
        fixture = [
            ChatGenerationChunk(message=mixed_msg),
            _chunk("</think>answer"),
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        tool_chunks_seen = [c for c in chunks if getattr(c.message, "tool_call_chunks", None)]
        assert len(tool_chunks_seen) == 1
        assert _aggregate_text(chunks) == "answer"


class TestStrippingChatOpenAIStream:
    """Sync streaming path mirrors async — kept narrow because no current consumer
    uses sync streaming, but the override exists for parity with _stream callers."""

    def test_sync_stream_strips_think_block(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [_chunk("<think>r</think>"), _chunk("answer")]
        with patch.object(ChatOpenAI, "_stream", _fake_stream_factory(fixture)):
            out = list(model._stream(messages=[], stop=None, run_manager=None))
        assert "".join(c.message.content for c in out) == "answer"


class TestOpenAIProviderIgnoresEnableThinking:
    def test_enable_thinking_does_not_raise(self):
        # OpenAI's API doesn't honor chat_template_kwargs; the parameter is
        # accepted for signature parity and silently ignored.
        provider = OpenAIProvider(api_key="sk-test")
        model = provider.get_chat_model(enable_thinking=False)
        assert getattr(model, "extra_body", None) is None
