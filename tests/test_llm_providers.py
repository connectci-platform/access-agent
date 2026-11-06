"""Tests for LLM provider abstraction — Qwen `</think>` strip + enable_thinking knob."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Self, cast
from unittest.mock import patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from openai.types.chat import ChatCompletionChunk

from src.llm.providers import (
    EMPTY_ANSWER_SENTINEL,
    OpenAICompatibleProvider,
    OpenAIProvider,
    _apply_strip_to_chunk,
    _flush_strip_state,
    _strip_generations,
    _strip_think_block,
    _StrippingChatOpenAI,
    _ThinkStripState,
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
    def _loop(content, node="model", chunk=True):
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

        loop = {"langgraph_node": "model"}
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


REASONING = "SECRET-TRACE the user wants a scheduler"


def _reasoning_chunk(
    reasoning: str, content: str = "", generation_info: dict[str, Any] | None = None
) -> ChatGenerationChunk:
    """A chunk as _convert_chunk_to_generation_chunk leaves it for a reasoning delta."""
    return ChatGenerationChunk(
        message=AIMessageChunk(content=content),
        generation_info={"reasoning_delta": reasoning, **(generation_info or {})},
    )


def _tool_chunk(content: str = "") -> ChatGenerationChunk:
    return _chunk(
        content,
        tool_call_chunks=[{"name": "search", "args": "{}", "id": "c1", "index": 0}],
    )


def _strip_all(
    chunks: list[ChatGenerationChunk],
) -> tuple[list[list[ChatGenerationChunk]], ChatGenerationChunk | None, _ThinkStripState]:
    """Drive the real strip over a script: what each chunk yielded, the tail, the state."""
    state = _ThinkStripState()
    per_chunk = [_apply_strip_to_chunk(chunk, state) for chunk in chunks]
    return per_chunk, _flush_strip_state(state), state


def _flatten(
    per_chunk: list[list[ChatGenerationChunk]], tail: ChatGenerationChunk | None
) -> list[ChatGenerationChunk]:
    out = [chunk for group in per_chunk for chunk in group]
    return [*out, tail] if tail is not None else out


def _wire_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    """One chat.completion.chunk as the OpenAI client dumps it."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _compatible_model() -> _StrippingChatOpenAI:
    provider = OpenAICompatibleProvider(
        base_url="http://example/v1", api_key="k", default_model="m"
    )
    return cast("_StrippingChatOpenAI", provider.get_chat_model())


def _convert(model: _StrippingChatOpenAI, wire: dict[str, Any]) -> ChatGenerationChunk | None:
    return model._convert_chunk_to_generation_chunk(wire, AIMessageChunk, {})


class _FakeCompletionStream:
    """The async context manager and iterator client.create() hands back."""

    def __init__(self, wire_chunks: list[dict[str, Any]]) -> None:
        self._chunks = [ChatCompletionChunk.model_validate(c) for c in wire_chunks]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncCompletions:
    def __init__(self, wire_chunks: list[dict[str, Any]]) -> None:
        self._wire_chunks = wire_chunks

    async def create(self, **payload: Any) -> _FakeCompletionStream:
        return _FakeCompletionStream(self._wire_chunks)


class _TokenSpy(BaseCallbackHandler):
    """Records everything an on_llm_new_token listener gets to see."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def on_llm_new_token(self, token: str, *, chunk: Any = None, **kwargs: Any) -> None:
        self.seen.append(f"{token!r} {chunk.model_dump() if chunk is not None else None}")


class TestReasoningReportedInASeparateField:
    """UKY's vLLM sends reasoning in delta.reasoning_content and clean text in
    delta.content, with no </think>. The field is the evidence that content can
    stream; without it the strip buffers exactly as it did before.
    """

    def setup_method(self):
        from src.agent.turn_capture import reset_turn_capture

        reset_turn_capture()

    def test_reasoning_field_releases_content_chunk_by_chunk(self):
        per_chunk, tail, _ = _strip_all(
            [
                _reasoning_chunk("step one, "),
                _reasoning_chunk("step two"),
                _chunk("\n\n"),
                _chunk("A batch"),
                _chunk(" scheduler"),
            ]
        )

        assert [[c.message.content for c in group] for group in per_chunk] == [
            [],
            [],
            [],
            ["A batch"],
            [" scheduler"],
        ]
        assert tail is None

    def test_spacing_inside_the_answer_survives(self):
        per_chunk, tail, _ = _strip_all(
            [_reasoning_chunk("r"), _chunk("\n\n  First"), _chunk("\n\n"), _chunk(" second ")]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "First\n\n second "

    def test_reasoning_text_never_reaches_the_message(self):
        per_chunk, tail, _ = _strip_all(
            [
                _reasoning_chunk(REASONING),
                _reasoning_chunk(REASONING, generation_info={"finish_reason": "length"}),
                _reasoning_chunk(REASONING, content="Answer"),
                _chunk(" more"),
                _reasoning_chunk(REASONING, generation_info={"finish_reason": "stop"}),
            ]
        )

        out = _flatten(per_chunk, tail)
        assert out
        for chunk in out:
            assert "SECRET-TRACE" not in str(chunk.model_dump())
            assert "reasoning_delta" not in (chunk.generation_info or {})

    def test_logprobs_of_reasoning_tokens_do_not_survive(self):
        logprobs = {"content": [{"token": "SECRET-TRACE", "logprob": -0.1}]}
        chunk = _reasoning_chunk(
            REASONING, generation_info={"logprobs": logprobs, "finish_reason": "length"}
        )

        per_chunk, _, _ = _strip_all([chunk])

        assert "SECRET-TRACE" not in str(per_chunk[0][0].model_dump())

    def test_reasoning_only_chunk_yields_nothing(self):
        per_chunk, _, _ = _strip_all([_reasoning_chunk(REASONING)])

        assert per_chunk == [[]]

    def test_reasoning_chunk_keeps_its_finish_reason(self):
        per_chunk, _, _ = _strip_all(
            [_reasoning_chunk("r", generation_info={"finish_reason": "length"})]
        )

        assert per_chunk[0][0].generation_info == {"finish_reason": "length"}

    def test_reasoning_chunk_keeps_its_usage(self):
        chunk = _reasoning_chunk("r")
        chunk.message.usage_metadata = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}

        per_chunk, _, _ = _strip_all([chunk])

        assert per_chunk[0][0].message.usage_metadata == chunk.message.usage_metadata

    def test_whitespace_chunk_keeps_its_finish_reason(self):
        last = ChatGenerationChunk(
            message=AIMessageChunk(content="\n\n"), generation_info={"finish_reason": "stop"}
        )

        per_chunk, _, _ = _strip_all([_reasoning_chunk("r"), last])

        assert per_chunk[1][0].message.content == ""
        assert per_chunk[1][0].generation_info == {"finish_reason": "stop"}

    def test_reasoning_field_tool_round_has_empty_content(self):
        per_chunk, tail, _ = _strip_all(
            [_reasoning_chunk(REASONING), _chunk("\n\n"), _tool_chunk(), _tool_chunk()]
        )

        out = _flatten(per_chunk, tail)
        assert len(out) == 2
        assert all(c.message.tool_call_chunks for c in out)
        assert _aggregate_text(out) == ""

    def test_whitespace_on_a_tool_call_chunk_is_emptied(self):
        per_chunk, tail, _ = _strip_all([_reasoning_chunk("r"), _tool_chunk("\n\n")])

        assert per_chunk[1][0].message.content == ""
        assert per_chunk[1][0].message.tool_call_chunks
        assert tail is None

    def test_reasoning_only_response_gets_the_sentinel(self):
        per_chunk, tail, _ = _strip_all([_reasoning_chunk(REASONING), _chunk("\n\n")])

        assert _flatten(per_chunk, None) == []
        assert tail is not None
        assert tail.message.content == EMPTY_ANSWER_SENTINEL

    def test_no_reasoning_field_buffers_as_before(self):
        per_chunk, tail, _ = _strip_all([_chunk("A batch"), _chunk(" scheduler")])

        assert per_chunk == [[], []]
        assert tail is not None
        assert tail.message.content == "A batch scheduler"

    def test_content_before_reasoning_field_stays_buffered(self):
        per_chunk, tail, state = _strip_all(
            [_chunk("Maybe a trace"), _reasoning_chunk(REASONING), _chunk(" still held")]
        )

        assert per_chunk == [[], [], []]
        assert not state.seen_close
        assert tail is not None
        assert tail.message.content == "Maybe a trace still held"
        assert "SECRET-TRACE" not in str(tail.model_dump())

    def test_content_before_reasoning_field_is_still_stripped_at_the_close_tag(self):
        per_chunk, tail, state = _strip_all(
            [_chunk("early trace"), _reasoning_chunk("field text"), _chunk("</think>Answer")]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "Answer"
        assert "field text" in state.reasoning
        assert "early trace" in state.reasoning

    def test_truncated_buffer_keeps_the_field_reasoning(self):
        per_chunk, tail, state = _strip_all(
            [_chunk("Plain <think>cut off"), _reasoning_chunk("field text")]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "Plain"
        assert "field text" in state.reasoning
        assert "cut off" in state.reasoning

    def test_whitespace_before_the_reasoning_field_does_not_hold_the_stream(self):
        per_chunk, tail, _ = _strip_all(
            [_chunk("\n"), _reasoning_chunk("r"), _chunk("Answer"), _chunk(" now")]
        )

        assert [[c.message.content for c in group] for group in per_chunk] == [
            [],
            [],
            ["Answer"],
            [" now"],
        ]
        assert tail is None

    def test_field_after_a_close_tag_leaves_that_path_alone(self):
        per_chunk, tail, state = _strip_all(
            [_chunk("trace</think>"), _reasoning_chunk("late"), _chunk("\n\n"), _chunk("Answer")]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "\n\nAnswer"
        assert "late" in state.reasoning

    def test_think_block_in_content_is_still_stripped(self):
        per_chunk, tail, state = _strip_all(
            [_chunk("<think>I should look."), _chunk("</think>\n\nAnswer"), _chunk(" here")]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "Answer here"
        assert state.reasoning == "I should look."

    def test_reasoning_is_recorded(self):
        _, _, state = _strip_all(
            [_reasoning_chunk("step one, "), _reasoning_chunk("step two"), _chunk("Answer")]
        )

        assert state.reasoning == "step one, step two"

    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    def test_convert_carries_the_field_in_generation_info(self, field):
        converted = _convert(_compatible_model(), _wire_chunk({field: "thinking", "content": None}))

        assert converted is not None
        assert converted.generation_info == {"reasoning_delta": "thinking"}
        assert converted.message.content == ""
        assert "thinking" not in str(converted.message.model_dump())

    def test_convert_does_not_double_a_delta_that_carries_both_fields(self):
        converted = _convert(
            _compatible_model(), _wire_chunk({"reasoning_content": "t", "reasoning": "t"})
        )

        assert converted is not None
        assert converted.generation_info == {"reasoning_delta": "t"}

    @pytest.mark.parametrize(
        "delta",
        [
            {"content": "Answer"},
            {"content": "Answer", "reasoning_content": ""},
            {"content": "Answer", "reasoning_content": None, "reasoning": None},
            {"content": "Answer", "reasoning_content": {"not": "text"}},
        ],
    )
    def test_convert_sets_nothing_without_reasoning_text(self, delta):
        converted = _convert(_compatible_model(), _wire_chunk(delta))

        assert converted is not None
        assert "reasoning_delta" not in (converted.generation_info or {})

    def test_convert_keeps_what_the_parent_recorded(self):
        converted = _convert(
            _compatible_model(), _wire_chunk({"reasoning_content": "t"}, finish_reason="length")
        )

        assert converted is not None
        assert converted.generation_info is not None
        assert converted.generation_info["finish_reason"] == "length"
        assert converted.generation_info["reasoning_delta"] == "t"

    def test_convert_does_not_touch_the_shared_base_generation_info(self):
        base = {"headers": {"x": "1"}}

        _compatible_model()._convert_chunk_to_generation_chunk(
            _wire_chunk({"reasoning_content": "t"}), AIMessageChunk, base
        )

        assert base == {"headers": {"x": "1"}}

    def test_convert_passes_through_a_chunk_with_no_choices(self):
        wire = {**_wire_chunk({}), "choices": []}

        converted = _convert(_compatible_model(), wire)

        assert converted is not None
        assert "reasoning_delta" not in (converted.generation_info or {})

    @pytest.mark.parametrize(
        "wire",
        [
            {"type": "content.delta"},
            {"choices": [{"index": 0, "delta": None}]},
        ],
    )
    def test_convert_passes_through_none_from_the_parent(self, wire):
        assert _convert(_compatible_model(), wire) is None

    def test_whitespace_reasoning_delta_is_not_evidence(self):
        per_chunk, tail, state = _strip_all(
            [_reasoning_chunk(" "), _chunk("A batch"), _chunk(" scheduler")]
        )

        assert per_chunk == [[], [], []]
        assert tail is not None
        assert tail.message.content == "A batch scheduler"
        assert state.reasoning == " "

    def test_think_block_after_the_field_is_stripped(self):
        per_chunk, tail, state = _strip_all(
            [_reasoning_chunk("real"), _chunk("<think>SECRET</think>"), _chunk("answer")]
        )

        out = _flatten(per_chunk, tail)
        assert _aggregate_text(out) == "answer"
        assert "SECRET" not in str([c.model_dump() for c in out])
        assert "SECRET" in state.reasoning

    def test_bare_close_after_a_whitespace_delta_is_stripped(self):
        per_chunk, tail, _ = _strip_all([_reasoning_chunk(" "), _chunk("SECRET think</think>")])

        out = _flatten(per_chunk, tail)
        assert "SECRET" not in str([c.model_dump() for c in out])
        assert tail is not None
        assert tail.message.content == EMPTY_ANSWER_SENTINEL

    def test_bare_close_after_the_field_drops_what_came_before_it(self):
        per_chunk, tail, _ = _strip_all(
            [_reasoning_chunk("real"), _chunk("SECRET think</think>Answer")]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "Answer"

    def test_tag_split_across_chunks_is_stripped(self):
        per_chunk, tail, _ = _strip_all(
            [
                _reasoning_chunk("real"),
                _chunk("<th"),
                _chunk("ink>SECRET</think>"),
                _chunk("answer"),
            ]
        )

        out = _flatten(per_chunk, tail)
        assert _aggregate_text(out) == "answer"
        assert "SECRET" not in str([c.model_dump() for c in out])

    def test_think_block_after_visible_text_keeps_the_text_and_strips_the_block(self):
        per_chunk, tail, _ = _strip_all(
            [
                _reasoning_chunk("real"),
                _chunk("Hello "),
                _chunk("<think>SECRET</think>"),
                _chunk("world"),
            ]
        )

        assert _aggregate_text(_flatten(per_chunk, tail)) == "Hello world"

    def test_unclosed_think_after_visible_text_flushes_no_sentinel_and_no_trace(self):
        per_chunk, tail, _ = _strip_all(
            [_reasoning_chunk("real"), _chunk("Hello"), _chunk("<think>SECRET")]
        )

        assert _aggregate_text(_flatten(per_chunk, None)) == "Hello"
        assert tail is None

    def test_think_block_with_no_answer_gets_the_sentinel(self):
        per_chunk, tail, _ = _strip_all([_reasoning_chunk("real"), _chunk("<think>SECRET</think>")])

        assert _flatten(per_chunk, None) == []
        assert tail is not None
        assert tail.message.content == EMPTY_ANSWER_SENTINEL

    def test_field_cannot_reopen_a_stream_a_tag_put_back_into_buffering(self):
        per_chunk, tail, _ = _strip_all(
            [
                _reasoning_chunk("real"),
                _chunk("<think>SECRET"),
                _reasoning_chunk("more"),
                _chunk(" still secret"),
            ]
        )

        out = _flatten(per_chunk, tail)
        assert "ecret" not in _aggregate_text(out).lower()

    @pytest.mark.parametrize(
        ("pieces", "expected"),
        [
            (["a < b"], "a < b"),
            (["a <", " b"], "a < b"),
            (["x </", "div>"], "x </div>"),
            (["<", "b>bold"], "<b>bold"),
            (["ends with <"], "ends with <"),
            (["<"], "<"),
            (["  <", " b"], "< b"),
        ],
    )
    def test_harmless_angle_brackets_arrive_intact(self, pieces, expected):
        per_chunk, tail, _ = _strip_all([_reasoning_chunk("real"), *map(_chunk, pieces)])

        assert _aggregate_text(_flatten(per_chunk, tail)) == expected

    def test_held_tail_goes_out_with_the_next_chunk(self):
        per_chunk, tail, _ = _strip_all([_reasoning_chunk("r"), _chunk("a <"), _chunk(" b")])

        assert [[c.message.content for c in group] for group in per_chunk] == [
            [],
            ["a "],
            ["< b"],
        ]
        assert tail is None

    def test_held_tail_on_a_tool_call_chunk_keeps_the_tool_call(self):
        per_chunk, _, _ = _strip_all([_reasoning_chunk("r"), _tool_chunk("<")])

        assert per_chunk[1][0].message.content == ""
        assert per_chunk[1][0].message.tool_call_chunks

    async def test_real_stream_with_a_think_block_in_content_stays_clean(self):
        model = _compatible_model()
        model.streaming = True
        model.async_client = _FakeAsyncCompletions(
            [
                _wire_chunk({"reasoning_content": "real"}),
                _wire_chunk({"content": "<think>SECRET</think>"}),
                _wire_chunk({"content": "answer"}),
                _wire_chunk({}, finish_reason="stop"),
            ]
        )

        message = await model.ainvoke("hi")

        assert message.content == "answer"
        assert "SECRET" not in str(message.model_dump())

    async def test_nested_graph_message_stream_never_carries_the_reasoning(self):
        model = _compatible_model()
        model.async_client = _FakeAsyncCompletions(
            [
                _wire_chunk({"reasoning_content": REASONING}),
                _wire_chunk({"content": "\n\n"}),
                _wire_chunk({"content": "A batch"}),
                _wire_chunk({"content": " scheduler"}),
                _wire_chunk({}, finish_reason="stop"),
            ]
        )

        async def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        inner = StateGraph(MessagesState)
        inner.add_node("model", call_model)
        inner.add_edge(START, "model")
        outer = StateGraph(MessagesState)
        outer.add_node("inner", inner.compile())
        outer.add_edge(START, "inner")

        events = [
            event
            async for event in outer.compile().astream(
                {"messages": [("user", "hi")]}, stream_mode=["messages"], subgraphs=True
            )
        ]

        tokens = [message.content for _, _, (message, _) in events if message.content]
        assert tokens == ["A batch", " scheduler"]
        assert "SECRET-TRACE" not in str(events)

    async def test_real_stream_through_the_openai_client_shape(self):
        """Real ChatCompletionChunk objects through the real parent _astream."""
        from src.agent.turn_capture import get_turn_capture

        model = _compatible_model()
        model.async_client = _FakeAsyncCompletions(
            [
                _wire_chunk({"role": "assistant", "content": ""}),
                _wire_chunk({"reasoning_content": REASONING}),
                _wire_chunk({"content": "\n\n"}),
                _wire_chunk({"content": "A batch"}),
                _wire_chunk({"content": " scheduler"}),
                _wire_chunk({}, finish_reason="stop"),
            ]
        )
        spy = _TokenSpy()

        chunks = [c async for c in model.astream("hi", config={"callbacks": [spy]})]

        assert "".join(str(c.content) for c in chunks) == "A batch scheduler"
        assert [c.content for c in chunks if c.content] == ["A batch", " scheduler"]
        assert any(c.response_metadata.get("finish_reason") == "stop" for c in chunks)
        assert "SECRET-TRACE" not in str([c.model_dump() for c in chunks])
        assert spy.seen
        assert "SECRET-TRACE" not in " ".join(spy.seen)
        assert "reasoning_delta" not in " ".join(spy.seen)
        assert REASONING in get_turn_capture()["model_reasoning"]

    async def test_real_stream_ainvoke_aggregates_clean(self):
        model = _compatible_model()
        model.streaming = True
        model.async_client = _FakeAsyncCompletions(
            [
                _wire_chunk({"reasoning": REASONING}),
                _wire_chunk({"content": "\n\nAnswer"}),
                _wire_chunk({}, finish_reason="stop"),
            ]
        )

        message = await model.ainvoke("hi")

        assert message.content == "Answer"
        assert "SECRET-TRACE" not in str(message.model_dump())
