"""Tests for the synthesize node, including token budget and condensation."""

import pytest

from src.agent.nodes.synthesize import (
    CHARS_PER_TOKEN,
    _estimate_tokens,
    _format_tool_results,
)
from src.agent.state import ToolResult
from src.config import settings


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_empty(self):
        """Empty string should return 0 tokens."""
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        """Short string should estimate correctly."""
        text = "Hello world"  # 11 chars
        expected = 11 // CHARS_PER_TOKEN
        assert _estimate_tokens(text) == expected

    def test_estimate_tokens_long(self):
        """Long string should estimate proportionally."""
        text = "a" * 1000
        expected = 1000 // CHARS_PER_TOKEN
        assert _estimate_tokens(text) == expected


class TestTokenBudget:
    """Tests for token budget configuration."""

    def test_default_budget(self):
        """Default budget should be set."""
        assert settings.SYNTHESIS_TOKEN_BUDGET > 0

    def test_budget_is_reasonable(self):
        """Budget should be in a reasonable range for LLMs."""
        # Should be at least 10K tokens for any useful synthesis
        assert settings.SYNTHESIS_TOKEN_BUDGET >= 10000
        # Should not exceed typical LLM context windows
        assert settings.SYNTHESIS_TOKEN_BUDGET <= 200000


class TestToolResultFormatting:
    """Tests for tool result formatting."""

    def test_format_empty_results(self):
        """Empty results should return empty string."""
        assert _format_tool_results([]) == ""

    def test_format_successful_result(self):
        """Successful result should include data."""
        results = [
            ToolResult(
                step_id="step_1",
                tool_name="test_tool",
                server="test-server",
                success=True,
                data={"key": "value"},
                duration_ms=100,
            )
        ]
        formatted = _format_tool_results(results)
        assert "test_tool" in formatted
        assert "SUCCESS" in formatted
        assert "key" in formatted

    def test_format_failed_result(self):
        """Failed result should include error."""
        results = [
            ToolResult(
                step_id="step_1",
                tool_name="test_tool",
                server="test-server",
                success=False,
                error="Connection timeout",
                duration_ms=100,
            )
        ]
        formatted = _format_tool_results(results)
        assert "test_tool" in formatted
        assert "FAILED" in formatted
        assert "Connection timeout" in formatted


class TestCondensationTrigger:
    """Tests for when condensation should be triggered."""

    def test_small_results_no_condensation(self):
        """Results under budget should not trigger condensation."""
        # Create results that are clearly under budget
        small_data = {"result": "small"}
        results = [
            ToolResult(
                step_id="step_1",
                tool_name="test_tool",
                server="test-server",
                success=True,
                data=small_data,
                duration_ms=100,
            )
        ]
        formatted = _format_tool_results(results)
        tokens = _estimate_tokens(formatted)
        assert tokens < settings.SYNTHESIS_TOKEN_BUDGET

    def test_large_results_trigger_condensation(self):
        """Results over budget should trigger condensation."""
        # Create results that exceed budget
        # Budget is in tokens, so we need chars = tokens * CHARS_PER_TOKEN
        chars_needed = (settings.SYNTHESIS_TOKEN_BUDGET + 1000) * CHARS_PER_TOKEN
        large_data = {"result": "x" * chars_needed}
        results = [
            ToolResult(
                step_id="step_1",
                tool_name="test_tool",
                server="test-server",
                success=True,
                data=large_data,
                duration_ms=100,
            )
        ]
        formatted = _format_tool_results(results)
        tokens = _estimate_tokens(formatted)
        assert tokens > settings.SYNTHESIS_TOKEN_BUDGET


@pytest.mark.asyncio
class TestCondensation:
    """Integration tests for condensation (requires LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key - run manually")
    async def test_condense_large_results(self):
        """Test that large results get condensed."""
        from src.agent.nodes.synthesize import _condense_tool_results

        query = "What AI software is available?"
        large_results = "Software list:\n" + "\n".join(
            [f"- software_{i}: A software package for testing" for i in range(500)]
        )

        condensed = await _condense_tool_results(query, large_results)

        # Condensed should be shorter than original
        assert len(condensed) < len(large_results)
        # Condensed should still contain relevant info
        assert "software" in condensed.lower()
