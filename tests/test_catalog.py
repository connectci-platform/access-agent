"""Tests for catalog aggregator."""

import pytest
from pytest_httpx import HTTPXMock

from src.tools.registry import CatalogAggregator, get_catalog_aggregator

# Test server URLs (simpler than mocking all 10 real servers)
TEST_SERVER_URLS = {
    "test-server-1": "http://test-server-1:3000",
    "test-server-2": "http://test-server-2:3000",
}


class TestCatalogAggregator:
    """Tests for CatalogAggregator."""

    @pytest.mark.asyncio
    async def test_fetch_catalog_success(self, httpx_mock: HTTPXMock):
        """Test successful catalog fetch from multiple servers."""
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-1:3000/tools",
            json={
                "tools": [
                    {
                        "name": "get_status",
                        "description": "Get system status",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "resource": {"type": "string", "description": "Resource name"}
                            },
                            "required": ["resource"],
                        },
                    }
                ]
            },
        )
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-2:3000/tools",
            json={"tools": []},
        )

        aggregator = CatalogAggregator(server_urls=TEST_SERVER_URLS)
        catalog = await aggregator.fetch_catalog()

        assert catalog["total_tools"] == 1
        assert catalog["total_servers"] == 2
        assert "get_status" in catalog["quick_lookup"]
        assert catalog["quick_lookup"]["get_status"]["server"] == "test-server-1"

    @pytest.mark.asyncio
    async def test_fetch_catalog_partial_failure(self, httpx_mock: HTTPXMock):
        """Test catalog fetch with some servers failing."""
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-1:3000/tools",
            json={"tools": [{"name": "working_tool", "description": "Works"}]},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-2:3000/tools",
            status_code=500,
        )

        aggregator = CatalogAggregator(server_urls=TEST_SERVER_URLS)
        catalog = await aggregator.fetch_catalog()

        assert catalog["total_tools"] == 1
        assert catalog["servers_available"] == 1
        assert catalog["total_servers"] == 2
        assert "working_tool" in catalog["quick_lookup"]

    @pytest.mark.asyncio
    async def test_fetch_catalog_caching(self, httpx_mock: HTTPXMock):
        """Test that catalog is cached and not refetched."""
        # First fetch responses
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-1:3000/tools",
            json={"tools": [{"name": "cached_tool", "description": "Cached"}]},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-2:3000/tools",
            json={"tools": []},
        )

        aggregator = CatalogAggregator(server_urls=TEST_SERVER_URLS)

        # First fetch
        catalog1 = await aggregator.fetch_catalog()
        assert catalog1["total_tools"] == 1

        # Second fetch should use cache (no additional HTTP call)
        catalog2 = await aggregator.fetch_catalog()
        assert catalog2 is catalog1  # Same object, from cache

        # Force refresh - add new responses
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-1:3000/tools",
            json={"tools": [{"name": "refreshed_tool", "description": "Refreshed"}]},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://test-server-2:3000/tools",
            json={"tools": []},
        )
        catalog3 = await aggregator.fetch_catalog(force_refresh=True)
        assert "refreshed_tool" in catalog3["quick_lookup"]

    def test_process_tool(self):
        """Test tool processing extracts parameters correctly."""
        aggregator = CatalogAggregator()

        raw_tool = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        }

        processed = aggregator._process_tool(raw_tool)

        assert processed["name"] == "test_tool"
        assert processed["description"] == "A test tool"
        assert len(processed["parameters"]) == 2

        query_param = next(p for p in processed["parameters"] if p["name"] == "query")
        assert query_param["required"] is True
        assert query_param["type"] == "string"

        limit_param = next(p for p in processed["parameters"] if p["name"] == "limit")
        assert limit_param["required"] is False
        assert limit_param["default"] == 10


class TestGetCatalogAggregator:
    """Tests for get_catalog_aggregator singleton."""

    def test_returns_same_instance(self):
        """Test that get_catalog_aggregator returns singleton."""
        agg1 = get_catalog_aggregator()
        agg2 = get_catalog_aggregator()
        assert agg1 is agg2
