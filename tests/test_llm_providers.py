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
        # Multiple </think> tags: the paired block is stripped first by
        # regex, then the bare </think> partition drops everything before
        # it. Net: only the final answer survives.
        raw = "<think>outer reasoning</think>middle</think>final answer"
        assert _strip_think_block(raw) == "final answer"

    def test_strips_when_close_tag_at_start(self):
        # Bare close tag with no open is treated as a degenerate trace.
        raw = "</think>just the answer"
        assert _strip_think_block(raw) == "just the answer"

    def test_preserves_internal_whitespace_after_strip(self):
        raw = "<think>thinking...</think>\n\nLine 1\n\nLine 2"
        assert _strip_think_block(raw) == "Line 1\n\nLine 2"

    def test_qwen_no_open_tag_stream_shape(self):
        # On UKY's vLLM the chat template consumes the opening <think>,
        # so the visible content is [reasoning prose]</think>[answer]
        # with no leading <think>. The reasoning prose must be stripped.
        raw = (
            "The user is asking about Anvil. From the documentation I can "
            "see it supports Python via modules. Let me confirm and then "
            "explain.\n</think>\n\nAnvil supports Python via the python "
            "and Anaconda modules."
        )
        out = _strip_think_block(raw)
        assert "</think>" not in out
        assert "The user is asking" not in out
        assert out.startswith("Anvil supports Python")

    def test_truncated_mid_think_returns_empty(self):
        # max_tokens cut the reasoning off before `</think>` was emitted,
        # and a <think> open tag was present. Never leak the buffered CoT.
        raw = "<think>I should consider every possibility before answering, starting with"
        assert _strip_think_block(raw) == ""

    def test_strips_think_block_with_preamble(self):
        # Some templates emit a brief preamble before opening the think
        # tag. The paired-block regex removes the trace; the preamble stays.
        raw = "Let me check: <think>reasoning here</think>The actual answer."
        assert _strip_think_block(raw) == "Let me check: The actual answer."

    def test_strips_multiple_paired_blocks(self):
        # Some models emit more than one paired block (e.g. an empty stub
        # followed by real reasoning). All paired blocks should be removed.
        raw = "<think></think><think>real reasoning</think>The answer."
        assert _strip_think_block(raw) == "The answer."

    def test_truncated_mid_think_with_preamble_drops_from_open(self):
        # Preamble before an unpaired <think> open is kept; the truncated
        # trace from the open tag onward is dropped.
        raw = "Looking this up: <think>I should consider"
        assert _strip_think_block(raw) == "Looking this up:"


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
        # Paired block stripped, then bare </think> partition drops "middle".
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [_chunk("<think>outer reasoning</think>middle</think>final answer")]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        assert _aggregate_text(chunks) == "final answer"

    @pytest.mark.asyncio
    async def test_qwen_no_open_tag_stream_shape(self) -> None:
        # Production reality: UKY vLLM emits reasoning prose then </think>
        # then the answer, with no leading <think>. Multiple chunks of
        # reasoning text must all be buffered (dropped), and only the
        # post-</think> answer streams out.
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [
            _chunk("The user is asking about Anvil. "),
            _chunk("From the documentation I can see "),
            _chunk("it supports Python via modules. "),
            _chunk("Let me confirm and then explain."),
            _chunk("\n</think>\n\n"),
            _chunk("Anvil supports Python via the python and Anaconda modules."),
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        text = _aggregate_text(chunks)
        assert "</think>" not in text
        assert "The user is asking" not in text
        assert "From the documentation" not in text
        assert text == "Anvil supports Python via the python and Anaconda modules."

    @pytest.mark.asyncio
    async def test_strips_think_block_with_preamble_in_stream(self) -> None:
        # Regression for the Qwen-emits-preamble-before-open case that leaked
        # </think> into production responses. Preamble chunks must pass
        # through; the trace between paired tags must be hidden.
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [
            _chunk("Let me check: "),
            _chunk("<think>reasoning"),
            _chunk(" continues</think>"),
            _chunk("The actual answer."),
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        text = _aggregate_text(chunks)
        assert "</think>" not in text
        assert "reasoning" not in text
        assert text == "Let me check: The actual answer."

    @pytest.mark.asyncio
    async def test_truncated_mid_think_in_stream_drops_buffer(self) -> None:
        # max_tokens cuts mid-reasoning. Preamble must still emit; the
        # truncated trace must not.
        provider = OpenAICompatibleProvider(
            base_url="http://example/v1", api_key="k", default_model="m"
        )
        model = cast("_StrippingChatOpenAI", provider.get_chat_model())
        fixture = [
            _chunk("Looking up: "),
            _chunk("<think>I should consider every"),
            # stream ends mid-think — no </think>
        ]
        with patch.object(ChatOpenAI, "_astream", _fake_astream_factory(fixture)):
            chunks = await _collect_astream(model)
        text = _aggregate_text(chunks)
        assert "consider" not in text
        assert text.rstrip() == "Looking up:"

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


class TestReasoningCapture:
    """Stripped <think> reasoning is recorded into turn_capture per model call."""

    def setup_method(self):
        from src.agent.turn_capture import reset_turn_capture

        reset_turn_capture()

    def test_split_returns_answer_and_reasoning(self):
        from src.llm.providers import _split_think_block

        answer, reasoning = _split_think_block("I should check the docs.</think>The answer.")
        assert answer == "The answer."
        assert reasoning == "I should check the docs."

    def test_split_no_think_has_empty_reasoning(self):
        from src.llm.providers import _split_think_block

        answer, reasoning = _split_think_block("Just an answer.")
        assert answer == "Just an answer."
        assert reasoning == ""

    def test_split_truncated_mid_think_captures_tail(self):
        from src.llm.providers import _split_think_block

        answer, reasoning = _split_think_block("Preamble.<think>cut off mid-reason")
        assert answer == "Preamble."
        assert reasoning == "cut off mid-reason"

    def test_split_paired_blocks_captured(self):
        from src.llm.providers import _split_think_block

        answer, reasoning = _split_think_block("<think>step one</think>Answer.")
        assert answer == "Answer."
        assert reasoning == "step one"

    def test_strip_generations_records_reasoning(self):
        from src.agent.turn_capture import get_turn_capture

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="thinking hard</think>Final."))]
        )
        _strip_generations(result)
        assert result.generations[0].message.content == "Final."
        assert get_turn_capture()["model_reasoning"] == ["thinking hard"]

    def test_strip_generations_no_reasoning_records_nothing(self):
        from src.agent.turn_capture import get_turn_capture

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Plain answer."))]
        )
        _strip_generations(result)
        assert get_turn_capture()["model_reasoning"] == []

    def test_stream_state_accumulates_reasoning_on_close(self):
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        assert state.feed("step one, ") is None
        assert state.feed("step two</think>Answer") == "Answer"
        assert state.reasoning == "step one, step two"

    def test_stream_state_flush_captures_truncated_reasoning(self):
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        assert state.feed("<think>never closed") is None
        assert state.flush() is None
        assert state.reasoning == "never closed"

    def test_stream_state_no_reasoning_stays_empty(self):
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        state.feed("plain ")
        assert state.flush() == "plain"
        assert state.reasoning == ""
