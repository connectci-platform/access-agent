"""Tests for LLM provider abstraction — Qwen `</think>` strip + enable_thinking knob."""

from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.llm.providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    _strip_generations,
    _strip_think_block,
    _StrippingChatOpenAI,
)


class TestStripThinkBlock:
    def test_strips_leading_think_block(self):
        raw = "I should look up X.\n</think>\n\nThe answer is foo."
        assert _strip_think_block(raw) == "The answer is foo."

    def test_no_think_tag_returns_unchanged(self):
        raw = "The answer is foo."
        assert _strip_think_block(raw) == "The answer is foo."

    def test_empty_content_unchanged(self):
        assert _strip_think_block("") == ""

    def test_strips_at_first_close_tag_when_multiple(self):
        # Defensive: model emits two `</think>` tags; we cut at the first.
        # Anything between the first and second close tag is treated as answer.
        raw = "outer reasoning</think>middle</think>final answer"
        assert _strip_think_block(raw) == "middle</think>final answer"

    def test_strips_when_close_tag_at_start(self):
        raw = "</think>just the answer"
        assert _strip_think_block(raw) == "just the answer"

    def test_preserves_internal_whitespace_after_strip(self):
        raw = "thinking...</think>\n\nLine 1\n\nLine 2"
        assert _strip_think_block(raw) == "Line 1\n\nLine 2"


class TestStripGenerations:
    def test_strips_each_generation_message(self):
        result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="reason A</think>answer A")),
                ChatGeneration(message=AIMessage(content="reason B</think>answer B")),
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


class TestOpenAIProviderIgnoresEnableThinking:
    def test_enable_thinking_does_not_raise(self):
        # OpenAI's API doesn't honor chat_template_kwargs; the parameter is
        # accepted for signature parity and silently ignored.
        provider = OpenAIProvider(api_key="sk-test")
        model = provider.get_chat_model(enable_thinking=False)
        assert getattr(model, "extra_body", None) is None
