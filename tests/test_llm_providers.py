"""Tests for LLM provider abstraction — Qwen `</think>` strip + enable_thinking knob."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from src.llm.providers import (
    EMPTY_ANSWER_SENTINEL,
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


class TestEmptyStrippedContentNeverBlankNeverLeaks:
    """A response whose visible content strips to nothing must stay non-empty.

    Observed in production 2026-09-11: 12 of 49 eval questions died with
    ``400 No user query found in messages`` from UKY's vLLM. Qwen had answered
    with reasoning that ended at ``</think>`` and nothing after; stripping set
    content to "", and the blank assistant turn made the NEXT request in the
    loop invalid — so the whole turn failed instead of degrading.

    The substitute is a neutral placeholder, never the buffered reasoning: a
    response truncated mid-trace also strips to "", and emitting that buffer is
    the chain-of-thought leak closed in 1d66cff.
    """

    def test_reasoning_only_gets_placeholder(self):
        result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="deciding which resource</think>"))
            ]
        )
        _strip_generations(result)
        assert result.generations[0].message.content == EMPTY_ANSWER_SENTINEL

    def test_truncated_mid_think_does_not_leak_the_trace(self):
        # The leak guard: an unpaired <think> means max_tokens cut the trace
        # short. The reasoning must never reach the user as the answer.
        secret = "<think>step one is to enumerate every possible resource"
        result = ChatResult(generations=[ChatGeneration(message=AIMessage(content=secret))])
        _strip_generations(result)
        content = result.generations[0].message.content
        assert content == EMPTY_ANSWER_SENTINEL

    def test_keeps_real_answer_when_present(self):
        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="reasoned</think>real answer"))]
        )
        _strip_generations(result)
        assert result.generations[0].message.content == "real answer"

    def test_tool_call_turn_stays_empty(self):
        # A tool-calling turn legitimately has no prose; its tool_calls carry
        # the intent, so a placeholder would render as spurious prose.
        msg = AIMessage(
            content="picking a tool</think>",
            tool_calls=[{"name": "search", "args": {}, "id": "c1"}],
        )
        _strip_generations(ChatResult(generations=[ChatGeneration(message=msg)]))
        assert msg.content == ""

    def test_streaming_reasoning_only_gets_placeholder(self):
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        emitted = [out for c in ["deciding which", " resource</think>"] if (out := state.feed(c))]
        flushed = state.flush()
        if flushed:
            emitted.append(flushed)
        joined = "".join(emitted)
        assert joined == EMPTY_ANSWER_SENTINEL
        assert "deciding which" not in joined

    def test_streaming_keeps_real_answer_and_flushes_nothing_extra(self):
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        emitted = [out for c in ["reasoned", "</think>", "real answer"] if (out := state.feed(c))]
        assert state.flush() is None
        assert "".join(emitted) == "real answer"


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
        from src.llm.providers import EMPTY_ANSWER_SENTINEL, _ThinkStripState

        state = _ThinkStripState()
        assert state.feed("<think>never closed") is None
        # The trace is still never emitted — but the turn cannot be left empty
        # either, so flush yields the sentinel the loop translates. Previously
        # this returned None, which is what produced the blank assistant turn
        # and the vLLM 400.
        flushed = state.flush()
        assert flushed == EMPTY_ANSWER_SENTINEL
        assert "never closed" not in flushed
        assert state.reasoning == "never closed"

    def test_stream_state_no_reasoning_stays_empty(self):
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        state.feed("plain ")
        assert state.flush() == "plain"
        assert state.reasoning == ""


class TestSentinelNeverReachesAUser:
    """The empty-answer sentinel is wire-protocol only.

    It exists so a reasoning-only response leaves a non-empty assistant turn
    (vLLM 400s on a blank one). If it escaped to the chat UI the user would read
    a bare internal marker as their answer, and the eval would judge it as a bad
    answer instead of recording a failed turn.
    """

    def test_sentinel_is_not_text_a_model_could_emit(self):
        from src.llm.providers import EMPTY_ANSWER_SENTINEL

        # NUL-prefixed: no model emits this, so a real answer can never collide
        # with it and be replaced by an apology.
        assert EMPTY_ANSWER_SENTINEL.startswith("\x00")

    def test_loop_translates_it_to_prose_and_tags_the_span(self):
        from unittest.mock import MagicMock

        from src.agent.nodes.tool_calling_loop import _answer_unavailable_message
        from src.llm.providers import EMPTY_ANSWER_SENTINEL

        span = MagicMock()
        msg = _answer_unavailable_message(48, span)
        assert EMPTY_ANSWER_SENTINEL not in msg
        assert "support.access-ci.org" in msg
        # Countable: these turns used to die as an opaque 400 with no signal.
        span.set_attribute.assert_called_once_with("agent.empty_answer", True)

    def test_streaming_suppresses_the_sentinel_on_a_tool_call_turn(self):
        # A tool-calling turn has no prose by design; the sentinel would render
        # as spurious text beside the call. Mirrors the non-streaming behavior.
        from src.llm.providers import _apply_strip_to_chunk, _flush_strip_state, _ThinkStripState

        state = _ThinkStripState()
        for raw in ("picking a tool", "</think>"):
            _apply_strip_to_chunk(ChatGenerationChunk(message=AIMessageChunk(content=raw)), state)
        _apply_strip_to_chunk(
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": "search", "args": "{}", "id": "c1", "index": 0}],
                )
            ),
            state,
        )
        assert _flush_strip_state(state) is None

    def test_whitespace_tail_through_the_chunk_path_is_detected_as_empty(self):
        """The shape an == guard misses.

        A post-</think> whitespace chunk is forwarded verbatim (inter-word
        spacing arrives as its own chunk), so the aggregate is "\n\n" + sentinel,
        not the sentinel. Asserting membership in the emitted LIST passes while
        the joined content still leaks the marker to the user — which is exactly
        what an earlier version of this test did. Assert the aggregate, and
        assert the consumer-side predicate recognises it.
        """
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        from src.llm.providers import (
            EMPTY_ANSWER_SENTINEL,
            _apply_strip_to_chunk,
            _flush_strip_state,
            _ThinkStripState,
            is_empty_answer,
        )

        state = _ThinkStripState()
        emitted: list[str] = []
        for raw in ["Let me think", "</think>", "\n\n"]:
            for out in _apply_strip_to_chunk(
                ChatGenerationChunk(message=AIMessageChunk(content=raw)), state
            ):
                if out.message.content:
                    emitted.append(str(out.message.content))
        tail = _flush_strip_state(state)
        if tail is not None and tail.message.content:
            emitted.append(str(tail.message.content))

        aggregate = "".join(emitted)
        # The raw aggregate is NOT equal to the sentinel — an == guard misses it.
        assert aggregate != EMPTY_ANSWER_SENTINEL
        # But it must still be recognised, or the marker reaches the user.
        assert is_empty_answer(aggregate)


class TestSseTokenGuard:
    """_should_stream_token decides what reaches the browser as a token event.

    qa-bot-core concatenates every `token` event it receives (qa-flow.tsx:
    `if (evType === 'token') collectedTokens += (parsed.content || '')`), so
    whatever this returns True for lands verbatim in the visible answer.
    """

    @staticmethod
    def _loop(content, node="tool_calling_loop", chunk=True):
        from langchain_core.messages import AIMessage, AIMessageChunk

        cls = AIMessageChunk if chunk else AIMessage
        return cls(content=content), {"langgraph_node": node}

    def test_sentinel_is_not_streamed(self):
        # The reason this predicate exists: the sentinel is wire-protocol only,
        # and tool_calling_loop substitutes real prose into final_answer, which
        # the done event carries. Streaming it would paste the marker into the
        # answer text the user reads.
        from src.api.routes import _should_stream_token
        from src.llm.providers import EMPTY_ANSWER_SENTINEL

        assert _should_stream_token(*self._loop(EMPTY_ANSWER_SENTINEL)) is False

    def test_ordinary_content_is_streamed(self):
        # Positive control. Without it the sentinel assertion above can pass
        # vacuously against a predicate that streams nothing at all.
        from src.api.routes import _should_stream_token

        assert _should_stream_token(*self._loop("Anvil supports Python.")) is True

    def test_whitespace_is_streamed(self):
        # Deliberate: inter-word spacing arrives as its own chunk, so dropping
        # whitespace would run words together. Unlike the strip layer, where a
        # whitespace-only RESPONSE is a blank turn, here it is a fragment.
        from src.api.routes import _should_stream_token

        assert _should_stream_token(*self._loop("   ")) is True

    def test_empty_content_is_not_streamed(self):
        from src.api.routes import _should_stream_token

        assert _should_stream_token(*self._loop("")) is False

    def test_other_nodes_are_not_streamed(self):
        from src.api.routes import _should_stream_token

        assert _should_stream_token(*self._loop("internal", node="other")) is False

    def test_complete_messages_are_not_streamed(self):
        # The messages stream carries both AIMessageChunk (tokens) and the
        # complete AIMessage added to state; streaming the latter would emit
        # the entire answer a second time.
        from src.api.routes import _should_stream_token

        assert _should_stream_token(*self._loop("whole answer", chunk=False)) is False


class TestSentinelDoesNotOutliveTheLoop:
    """Findings from the second review round, each with its own failure mode."""

    def test_unrelated_additional_kwargs_do_not_suppress_the_sentinel(self):
        # _chunk_carries_non_text_payload is true for ANY additional_kwargs. If
        # that gated the tool-call check, a provider adding e.g. "refusal" would
        # silently disable the empty-turn guard and the vLLM 400 would return.
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        from src.llm.providers import (
            EMPTY_ANSWER_SENTINEL,
            _apply_strip_to_chunk,
            _flush_strip_state,
            _ThinkStripState,
        )

        state = _ThinkStripState()
        for raw, kwargs in [("reasoning", {}), ("</think>", {"refusal": None})]:
            _apply_strip_to_chunk(
                ChatGenerationChunk(message=AIMessageChunk(content=raw, additional_kwargs=kwargs)),
                state,
            )
        tail = _flush_strip_state(state)
        assert tail is not None
        assert tail.message.content == EMPTY_ANSWER_SENTINEL

    def test_is_empty_answer_tolerates_surrounding_whitespace(self):
        from src.llm.providers import EMPTY_ANSWER_SENTINEL, is_empty_answer

        assert is_empty_answer(EMPTY_ANSWER_SENTINEL)
        assert is_empty_answer(f"\n\n{EMPTY_ANSWER_SENTINEL}")
        assert is_empty_answer(f"  {EMPTY_ANSWER_SENTINEL}  ")
        # Must not swallow a real answer that merely mentions it.
        assert not is_empty_answer(f"the marker is {EMPTY_ANSWER_SENTINEL}")
        assert not is_empty_answer("Anvil supports Python.")
        assert not is_empty_answer("")
        assert not is_empty_answer(None)

    def test_loop_rewrites_the_sentinel_message_not_just_final_answer(self):
        # With checkpointing on, messages are persisted and replayed as prior
        # context. A raw sentinel would come back as an uninterpretable
        # assistant turn and feed into summarization.
        from langchain_core.messages import AIMessage

        from src.agent.nodes.tool_calling_loop import _replace_sentinel_message
        from src.llm.providers import EMPTY_ANSWER_SENTINEL

        messages = [
            AIMessage(content="earlier real answer"),
            AIMessage(content=f"\n\n{EMPTY_ANSWER_SENTINEL}"),
        ]
        _replace_sentinel_message(messages, "sorry, no answer")
        assert messages[1].content == "sorry, no answer"
        assert messages[0].content == "earlier real answer"


class TestEmptyAnswerIntegrationPaths:
    """Cover the integration points the unit tests bypass."""

    def test_truncated_stream_never_seeing_close_yields_the_sentinel(self):
        # flush()'s no-close branch: max_tokens cut the response before
        # </think>, so the trace is dropped and nothing would be emitted.
        from src.llm.providers import EMPTY_ANSWER_SENTINEL, _flush_strip_state, _ThinkStripState

        state = _ThinkStripState()
        assert state.feed("<think>reasoning that never closes") is None
        tail = _flush_strip_state(state)
        assert tail is not None
        assert tail.message.content == EMPTY_ANSWER_SENTINEL
        assert "never closes" not in str(tail.message.content)

    def test_loop_substitutes_prose_for_a_sentinel_final_answer(self):
        # The three lines that turn a sentinel into a user-facing answer, and
        # flag the turn so the eval records a failure rather than judging it.
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage

        from src.agent.nodes.tool_calling_loop import (
            _answer_unavailable_message,
            _replace_sentinel_message,
        )
        from src.llm.providers import EMPTY_ANSWER_SENTINEL, is_empty_answer

        final_answer = f"\n\n{EMPTY_ANSWER_SENTINEL}"
        assert is_empty_answer(final_answer)

        span = MagicMock()
        messages = [AIMessage(content=final_answer)]
        final_answer = _answer_unavailable_message(12, span)
        _replace_sentinel_message(messages, final_answer)

        assert EMPTY_ANSWER_SENTINEL not in final_answer
        assert EMPTY_ANSWER_SENTINEL not in str(messages[0].content)
        assert not is_empty_answer(final_answer)

    def test_stream_events_calls_the_token_predicate(self):
        # routes.py's call site: the predicate must gate the yield, so a
        # sentinel chunk produces no token event while real content does.
        import inspect

        from langchain_core.messages import AIMessageChunk

        from src.api import routes
        from src.llm.providers import EMPTY_ANSWER_SENTINEL

        src = inspect.getsource(routes._stream_events)
        assert "_should_stream_token(msg, metadata)" in src

        loop = {"langgraph_node": "tool_calling_loop"}
        assert not routes._should_stream_token(
            AIMessageChunk(content=f"\n{EMPTY_ANSWER_SENTINEL}"), loop
        )
        assert routes._should_stream_token(AIMessageChunk(content="real"), loop)

    def test_flush_adds_nothing_when_a_tool_call_already_streamed(self):
        # The no-close fall-through: a tool-calling turn that ends without
        # </think> must not get a sentinel appended beside its tool call.
        from src.llm.providers import _ThinkStripState

        state = _ThinkStripState()
        state.mark_tool_call_seen()
        assert state.feed("<think>choosing a tool") is None
        assert state.flush() is None
