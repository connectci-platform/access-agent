"""Tests for execute node functions."""

from src.agent.nodes.execute import (
    _build_dependency_levels,
    _parse_path,
    _resolve_parameters,
    _resolve_reference,
)
from src.agent.state import ToolCall, ToolResult


class TestParsePath:
    """Tests for _parse_path function."""

    def test_simple_path(self):
        result = _parse_path("data")
        assert result == ["data"]

    def test_dotted_path(self):
        result = _parse_path("data.resources")
        assert result == ["data", "resources"]

    def test_path_with_array_index(self):
        result = _parse_path("data.resources[0]")
        assert result == ["data", "resources", 0]

    def test_path_with_multiple_indices(self):
        result = _parse_path("data[0].items[1].name")
        assert result == ["data", 0, "items", 1, "name"]

    def test_path_with_field_after_index(self):
        result = _parse_path("resources[0].id")
        assert result == ["resources", 0, "id"]


class TestResolveReference:
    """Tests for _resolve_reference function."""

    def test_simple_reference(self):
        # Path after $step_1. navigates within result.data
        # So for data={"resources": [{"name": "ACES"}]}, path is "resources[0].name"
        results = {
            "step_1": ToolResult(
                step_id="step_1",
                tool_name="search",
                server="compute",
                success=True,
                data={"resources": [{"name": "ACES"}]},
            )
        }
        result = _resolve_reference("$step_1.resources[0].name", results)
        assert result == "ACES"

    def test_reference_to_unknown_step(self):
        results = {}
        result = _resolve_reference("$step_1.value", results)
        assert result == "$step_1.value"  # Returns original

    def test_reference_to_failed_step(self):
        results = {
            "step_1": ToolResult(
                step_id="step_1",
                tool_name="search",
                server="compute",
                success=False,
                error="Failed",
            )
        }
        result = _resolve_reference("$step_1.value", results)
        assert result == "$step_1.value"  # Returns original

    def test_invalid_reference_format(self):
        results = {}
        result = _resolve_reference("not_a_reference", results)
        assert result == "not_a_reference"


class TestResolveParameters:
    """Tests for _resolve_parameters function."""

    def test_no_references(self):
        args = {"query": "test", "limit": 10}
        results = {}
        resolved = _resolve_parameters(args, results)
        assert resolved == {"query": "test", "limit": 10}

    def test_with_reference(self):
        args = {"resource_id": "$step_1.id"}
        results = {
            "step_1": ToolResult(
                step_id="step_1",
                tool_name="search",
                server="compute",
                success=True,
                data={"id": "aces-123"},
            )
        }
        resolved = _resolve_parameters(args, results)
        assert resolved == {"resource_id": "aces-123"}

    def test_nested_dict_with_reference(self):
        args = {"filters": {"resource_id": "$step_1.id"}}
        results = {
            "step_1": ToolResult(
                step_id="step_1",
                tool_name="search",
                server="compute",
                success=True,
                data={"id": "aces-123"},
            )
        }
        resolved = _resolve_parameters(args, results)
        assert resolved == {"filters": {"resource_id": "aces-123"}}

    def test_list_with_reference(self):
        args = {"ids": ["$step_1.id", "static-id"]}
        results = {
            "step_1": ToolResult(
                step_id="step_1",
                tool_name="search",
                server="compute",
                success=True,
                data={"id": "aces-123"},
            )
        }
        resolved = _resolve_parameters(args, results)
        assert resolved == {"ids": ["aces-123", "static-id"]}


class TestBuildDependencyLevels:
    """Tests for _build_dependency_levels function."""

    def test_no_dependencies(self):
        tools = [
            ToolCall(step_id="step_1", tool_name="t1", server="s", arguments={}, depends_on=[]),
            ToolCall(step_id="step_2", tool_name="t2", server="s", arguments={}, depends_on=[]),
        ]
        levels = _build_dependency_levels(tools)
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_sequential_dependencies(self):
        tools = [
            ToolCall(step_id="step_1", tool_name="t1", server="s", arguments={}, depends_on=[]),
            ToolCall(
                step_id="step_2", tool_name="t2", server="s", arguments={}, depends_on=["step_1"]
            ),
            ToolCall(
                step_id="step_3", tool_name="t3", server="s", arguments={}, depends_on=["step_2"]
            ),
        ]
        levels = _build_dependency_levels(tools)
        assert len(levels) == 3
        assert levels[0][0].step_id == "step_1"
        assert levels[1][0].step_id == "step_2"
        assert levels[2][0].step_id == "step_3"

    def test_parallel_with_shared_dependency(self):
        tools = [
            ToolCall(step_id="step_1", tool_name="t1", server="s", arguments={}, depends_on=[]),
            ToolCall(
                step_id="step_2", tool_name="t2", server="s", arguments={}, depends_on=["step_1"]
            ),
            ToolCall(
                step_id="step_3", tool_name="t3", server="s", arguments={}, depends_on=["step_1"]
            ),
        ]
        levels = _build_dependency_levels(tools)
        assert len(levels) == 2
        assert len(levels[0]) == 1  # step_1
        assert len(levels[1]) == 2  # step_2 and step_3 in parallel
