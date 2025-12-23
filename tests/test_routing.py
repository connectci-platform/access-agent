"""Tests for routing edge functions."""

from src.agent.edges.routing import (
    should_execute_tools,
    should_recover_or_evaluate,
    should_retry_or_synthesize,
    should_retry_quality,
)
from src.agent.state import QualityEvaluation, QueryAnalysis, ToolResult


class TestShouldExecuteTools:
    """Tests for should_execute_tools routing."""

    def test_no_query_analysis_goes_to_compress(self):
        state = {"query_analysis": None, "planned_tools": []}
        assert should_execute_tools(state) == "compress"

    def test_no_tools_required_goes_to_compress(self):
        state = {
            "query_analysis": QueryAnalysis(
                user_intent="General question about ACCESS",
                requires_tools=False,
            ),
            "planned_tools": [],
        }
        assert should_execute_tools(state) == "compress"

    def test_tools_required_goes_to_execute(self):
        state = {
            "query_analysis": QueryAnalysis(
                user_intent="Search for compute resources",
                requires_tools=True,
            ),
            "planned_tools": [{"step_id": "step_1", "tool_name": "search"}],
        }
        assert should_execute_tools(state) == "execute"

    def test_tools_required_but_none_planned_goes_to_compress(self):
        state = {
            "query_analysis": QueryAnalysis(
                user_intent="Search for resources",
                requires_tools=True,
            ),
            "planned_tools": [],
        }
        assert should_execute_tools(state) == "compress"


class TestShouldRecoverOrEvaluate:
    """Tests for should_recover_or_evaluate routing."""

    def test_all_tools_succeeded_goes_to_evaluate(self):
        state = {
            "tool_results": [
                ToolResult(
                    step_id="step_1",
                    tool_name="search",
                    server="compute-resources",
                    success=True,
                    data={"results": []},
                )
            ]
        }
        assert should_recover_or_evaluate(state) == "evaluate"

    def test_some_tools_failed_goes_to_recover(self):
        state = {
            "tool_results": [
                ToolResult(
                    step_id="step_1",
                    tool_name="search",
                    server="compute-resources",
                    success=True,
                    data={"results": []},
                ),
                ToolResult(
                    step_id="step_2",
                    tool_name="get_details",
                    server="compute-resources",
                    success=False,
                    error="Connection timeout",
                ),
            ]
        }
        assert should_recover_or_evaluate(state) == "recover"

    def test_no_results_goes_to_evaluate(self):
        state = {"tool_results": []}
        assert should_recover_or_evaluate(state) == "evaluate"


class TestShouldRetryOrSynthesize:
    """Tests for should_retry_or_synthesize routing."""

    def test_has_planned_tools_goes_to_execute(self):
        state = {"planned_tools": [{"step_id": "step_1"}]}
        assert should_retry_or_synthesize(state) == "execute"

    def test_no_planned_tools_goes_to_compress(self):
        state = {"planned_tools": []}
        assert should_retry_or_synthesize(state) == "compress"


class TestShouldRetryQuality:
    """Tests for should_retry_quality routing."""

    def test_no_evaluation_goes_to_compress(self):
        state = {"quality_evaluation": None, "attempt_number": 0}
        assert should_retry_quality(state) == "compress"

    def test_helpful_goes_to_compress(self):
        state = {
            "quality_evaluation": QualityEvaluation(
                is_helpful=True,
                confidence="high",
                reason="Complete answer",
            ),
            "attempt_number": 1,
        }
        assert should_retry_quality(state) == "compress"

    def test_unhelpful_with_attempts_left_goes_to_plan(self):
        state = {
            "quality_evaluation": QualityEvaluation(
                is_helpful=False,
                confidence="low",
                reason="Missing information",
            ),
            "attempt_number": 1,
        }
        assert should_retry_quality(state) == "plan"

    def test_unhelpful_max_attempts_reached_goes_to_compress(self):
        state = {
            "quality_evaluation": QualityEvaluation(
                is_helpful=False,
                confidence="low",
                reason="Missing information",
            ),
            "attempt_number": 3,  # Equals MAX_QUALITY_ATTEMPTS
        }
        assert should_retry_quality(state) == "compress"
