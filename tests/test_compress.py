"""Tests for the compress node."""

import pytest

from src.agent.nodes.compress import (
    _compress_data,
    _compress_list,
    _compress_single_result,
    _estimate_size,
    compress_node,
)
from src.agent.state import ToolResult


class TestCompressData:
    """Tests for _compress_data function."""

    def test_compress_none_returns_none(self):
        assert _compress_data(None, None) is None

    def test_compress_dict_with_items(self):
        data = {
            "total": 2,
            "items": [
                {"id": "1", "name": "Resource 1", "extra": "ignored"},
                {"id": "2", "name": "Resource 2", "extra": "ignored"},
            ],
        }
        result = _compress_data(data, ["id", "name"])
        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert result["items"][0] == {"id": "1", "name": "Resource 1"}
        assert "extra" not in result["items"][0]

    def test_compress_dict_with_results(self):
        data = {
            "results": [
                {"name": "Test", "description": "Desc", "other": "field"},
            ],
        }
        result = _compress_data(data, ["name", "description"])
        assert "items" in result
        assert result["items"][0] == {"name": "Test", "description": "Desc"}

    def test_compress_list_directly(self):
        data = [
            {"id": "1", "name": "A", "extra": "x"},
            {"id": "2", "name": "B", "extra": "y"},
        ]
        result = _compress_data(data, ["id", "name"])
        assert result == [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]

    def test_compress_without_relevant_fields_keeps_all(self):
        data = {"id": "1", "name": "Test", "extra": "kept"}
        result = _compress_data(data, None)
        assert result == data

    def test_compress_primitive_unchanged(self):
        assert _compress_data("string", None) == "string"
        assert _compress_data(123, None) == 123
        assert _compress_data(True, None) is True


class TestCompressList:
    """Tests for _compress_list function."""

    def test_empty_list(self):
        assert _compress_list([], ["id"]) == []

    def test_compress_with_fields(self):
        items = [
            {"id": "1", "name": "A", "extra": "x"},
            {"id": "2", "name": "B", "extra": "y"},
        ]
        result = _compress_list(items, ["id", "name"])
        assert result == [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]

    def test_compress_without_fields_keeps_all(self):
        items = [{"id": "1", "extra": "x"}]
        result = _compress_list(items, None)
        assert result == items

    def test_non_dict_items_unchanged(self):
        items = ["a", "b", "c"]
        result = _compress_list(items, ["id"])
        assert result == items


class TestCompressSingleResult:
    """Tests for _compress_single_result function."""

    def test_failed_result(self):
        result = ToolResult(
            step_id="step_1",
            tool_name="search_resources",
            server="compute-resources",
            success=False,
            error="Connection timeout",
        )
        compressed = _compress_single_result(result, None)
        assert compressed["tool_name"] == "search_resources"
        assert compressed["success"] is False
        assert compressed["error"] == "Connection timeout"

    def test_successful_result_with_known_tool(self):
        result = ToolResult(
            step_id="step_1",
            tool_name="search_resources",
            server="compute-resources",
            success=True,
            data={
                "total": 1,
                "items": [
                    {
                        "id": "delta.ncsa.access-ci.org",
                        "name": "Delta",
                        "description": "GPU cluster",
                        "organization_names": ["NCSA"],
                        "resourceType": "gpu",
                        "hasGpu": True,
                        "features": [100, 142],  # Should be filtered out
                        "extra_field": "ignored",
                    }
                ],
            },
        )
        compressed = _compress_single_result(result, None)
        assert compressed["success"] is True
        items = compressed["data"]["items"]
        assert len(items) == 1
        assert items[0]["name"] == "Delta"
        assert items[0]["hasGpu"] is True
        assert "features" not in items[0]
        assert "extra_field" not in items[0]

    def test_successful_result_with_unknown_tool(self):
        result = ToolResult(
            step_id="step_1",
            tool_name="unknown_tool",
            server="some-server",
            success=True,
            data={"key": "value", "other": "data"},
        )
        compressed = _compress_single_result(result, None)
        # Unknown tool keeps all data
        assert compressed["data"] == {"key": "value", "other": "data"}


class TestEstimateSize:
    """Tests for _estimate_size function."""

    def test_none_returns_zero(self):
        assert _estimate_size(None) == 0

    def test_dict_size(self):
        data = {"key": "value"}
        size = _estimate_size(data)
        assert size > 0

    def test_list_size(self):
        data = [1, 2, 3]
        size = _estimate_size(data)
        assert size > 0


class TestCompressNode:
    """Tests for the compress_node function."""

    @pytest.mark.asyncio
    async def test_empty_results(self):
        state = {"tool_results": [], "query_analysis": None}
        result = await compress_node(state)
        assert result["compressed_results"] == []

    @pytest.mark.asyncio
    async def test_compresses_resource_results(self):
        state = {
            "tool_results": [
                ToolResult(
                    step_id="step_1",
                    tool_name="search_resources",
                    server="compute-resources",
                    success=True,
                    data={
                        "total": 2,
                        "items": [
                            {
                                "id": "delta",
                                "name": "Delta",
                                "description": "GPU cluster",
                                "organization_names": ["NCSA"],
                                "hasGpu": True,
                                "resourceType": "gpu",
                                "features": [100, 142],
                                "feature_names": ["GPU Computing"],
                            },
                            {
                                "id": "anvil",
                                "name": "Anvil",
                                "description": "HPC system",
                                "organization_names": ["Purdue"],
                                "hasGpu": True,
                                "resourceType": "gpu",
                                "features": [100, 142],
                            },
                        ],
                    },
                )
            ],
            "query_analysis": None,
        }
        result = await compress_node(state)
        compressed = result["compressed_results"]
        assert len(compressed) == 1
        assert compressed[0]["success"] is True

        items = compressed[0]["data"]["items"]
        assert len(items) == 2
        # Check that only relevant fields are kept
        for item in items:
            assert "name" in item
            assert "description" in item
            assert "hasGpu" in item
            assert "features" not in item
            assert "feature_names" not in item
