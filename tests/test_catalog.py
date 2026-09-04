"""Tests for catalog aggregator."""

from unittest.mock import patch

import pytest
from pytest_httpx import HTTPXMock

from src.tools.registry import CatalogAggregator, get_catalog_aggregator

# Test server URLs. The aggregator capability-filters every build, dropping
# servers no enabled capability owns — so these must be real capability-owned
# server names (both enabled by default), not made-up ones.
TEST_SERVER_URLS = {
    "allocations": "http://allocations:3000",
    "events": "http://events:3000",
}


@pytest.fixture(autouse=True)
def _fresh_capability_registry():
    """Rebuild the capability registry singleton around each test.

    The aggregator's capability filter reads the singleton; tests that patch
    ENABLED/DISABLED_CAPABILITIES need a fresh build, and tests after them
    need the default build back.
    """
    from src.agent.domains import capabilities as cap_module

    cap_module._registry = None
    yield
    cap_module._registry = None


class TestCatalogAggregator:
    """Tests for CatalogAggregator."""

    @pytest.mark.asyncio
    async def test_fetch_catalog_success(self, httpx_mock: HTTPXMock):
        """Test successful catalog fetch from multiple servers."""
        httpx_mock.add_response(
            method="GET",
            url="http://allocations:3000/tools",
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
            url="http://events:3000/tools",
            json={"tools": []},
        )

        aggregator = CatalogAggregator(server_urls=TEST_SERVER_URLS)
        catalog = await aggregator.fetch_catalog()

        assert catalog["total_tools"] == 1
        assert catalog["total_servers"] == 2
        assert "get_status" in catalog["quick_lookup"]
        assert catalog["quick_lookup"]["get_status"]["server"] == "allocations"

    @pytest.mark.asyncio
    async def test_fetch_catalog_partial_failure(self, httpx_mock: HTTPXMock):
        """Test catalog fetch with some servers failing."""
        httpx_mock.add_response(
            method="GET",
            url="http://allocations:3000/tools",
            json={"tools": [{"name": "working_tool", "description": "Works"}]},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://events:3000/tools",
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
            url="http://allocations:3000/tools",
            json={"tools": [{"name": "cached_tool", "description": "Cached"}]},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://events:3000/tools",
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
            url="http://allocations:3000/tools",
            json={"tools": [{"name": "refreshed_tool", "description": "Refreshed"}]},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://events:3000/tools",
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


class TestAggregatorCapabilityFilter:
    """The aggregator's cached catalog must be capability-filtered on EVERY
    build — including refreshes (#240)."""

    def _mock_both(self, httpx_mock: HTTPXMock, urls: dict[str, str]) -> None:
        for name, url in urls.items():
            httpx_mock.add_response(
                method="GET",
                url=f"{url}/tools",
                json={"tools": [{"name": f"{name}__tool", "description": f"{name} tool"}]},
            )

    @pytest.mark.asyncio
    async def test_fetch_drops_disabled_capability_server(self, httpx_mock: HTTPXMock):
        from src.config import settings

        urls = {"allocations": "http://allocations:3000", "xdmod-data": "http://xdmod-data:3000"}
        self._mock_both(httpx_mock, urls)

        with patch.object(settings, "DISABLED_CAPABILITIES", "extract_xdmod_data"):
            aggregator = CatalogAggregator(server_urls=urls)
            catalog = await aggregator.fetch_catalog()

        server_names = [s["server"] for s in catalog["servers"]]
        assert "xdmod-data" not in server_names
        assert "allocations" in server_names
        assert "xdmod-data__tool" not in catalog["quick_lookup"]
        assert catalog["total_tools"] == 1
        assert catalog["servers_available"] == 1
        # total_servers must describe the same filtered universe — /health
        # reports "degraded" (failing the deploy gate) when available < total,
        # so a filtered-out server must not count toward the total.
        assert catalog["total_servers"] == 1

    @pytest.mark.asyncio
    async def test_refresh_keeps_capability_filter(self, httpx_mock: HTTPXMock):
        """Regression for #240: a force_refresh used to rebuild the catalog
        unfiltered, un-hiding disabled servers on /catalog until restart."""
        from src.config import settings

        urls = {"allocations": "http://allocations:3000", "xdmod-data": "http://xdmod-data:3000"}
        self._mock_both(httpx_mock, urls)
        self._mock_both(httpx_mock, urls)  # second round for the refresh

        with patch.object(settings, "DISABLED_CAPABILITIES", "extract_xdmod_data"):
            aggregator = CatalogAggregator(server_urls=urls)
            await aggregator.fetch_catalog()
            refreshed = await aggregator.fetch_catalog(force_refresh=True)

        assert "xdmod-data" not in [s["server"] for s in refreshed["servers"]]
        assert "xdmod-data__tool" not in refreshed["quick_lookup"]

    @pytest.mark.asyncio
    async def test_unknown_server_is_dropped(self, httpx_mock: HTTPXMock):
        """A server no capability owns is filtered (fail closed), not passed."""
        urls = {"allocations": "http://allocations:3000", "mystery": "http://mystery:3000"}
        self._mock_both(httpx_mock, urls)

        aggregator = CatalogAggregator(server_urls=urls)
        catalog = await aggregator.fetch_catalog()

        assert "mystery" not in [s["server"] for s in catalog["servers"]]
        assert "mystery__tool" not in catalog["quick_lookup"]

    @pytest.mark.asyncio
    async def test_filter_skipped_when_registry_unavailable(self, httpx_mock: HTTPXMock):
        """Belt-and-suspenders: a broken capability registry must not empty
        the catalog."""
        self._mock_both(httpx_mock, TEST_SERVER_URLS)

        aggregator = CatalogAggregator(server_urls=TEST_SERVER_URLS)
        with patch(
            "src.agent.domains.capabilities.get_capability_registry",
            side_effect=RuntimeError("registry broken"),
        ):
            catalog = await aggregator.fetch_catalog()

        assert len(catalog["servers"]) == 2


class TestHealthWithFilteredCatalog:
    """/health must not report degraded merely because a capability is
    disabled — the deploy gate hard-fails on anything but "healthy"."""

    @pytest.mark.asyncio
    async def test_disabled_capability_does_not_degrade_health(self, httpx_mock: HTTPXMock):
        from src.api import routes
        from src.config import settings

        urls = {"allocations": "http://allocations:3000", "xdmod-data": "http://xdmod-data:3000"}
        for name, url in urls.items():
            httpx_mock.add_response(
                method="GET",
                url=f"{url}/tools",
                json={"tools": [{"name": f"{name}__tool", "description": f"{name} tool"}]},
            )

        with patch.object(settings, "DISABLED_CAPABILITIES", "extract_xdmod_data"):
            aggregator = CatalogAggregator(server_urls=urls)
            await aggregator.fetch_catalog()
            with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
                result = await routes.health_check()

        assert result["status"] == "healthy"
        assert result["tools"]["servers_total"] == result["tools"]["servers_available"] == 1

    @pytest.mark.asyncio
    async def test_down_enabled_server_still_degrades_health(self, httpx_mock: HTTPXMock):
        from src.api import routes

        httpx_mock.add_response(
            method="GET",
            url="http://allocations:3000/tools",
            json={"tools": []},
        )
        httpx_mock.add_response(
            method="GET",
            url="http://events:3000/tools",
            status_code=500,
        )

        aggregator = CatalogAggregator(server_urls=TEST_SERVER_URLS)
        await aggregator.fetch_catalog()
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            result = await routes.health_check()

        assert result["status"] == "degraded"
        assert "events" in result["tools"]["unavailable_servers"]


class TestRefreshInvalidatesRegistry:
    """Catalog refreshes must drop the cached ToolRegistry so the agent's
    next query rebuilds from the refreshed catalog (#240 — previously a
    refresh only updated the public /catalog payload)."""

    @pytest.mark.asyncio
    async def test_post_refresh_invalidates(self, httpx_mock: HTTPXMock):
        from src.api import routes

        httpx_mock.add_response(
            method="GET",
            url="http://allocations:3000/tools",
            json={"tools": []},
        )
        aggregator = CatalogAggregator(server_urls={"allocations": "http://allocations:3000"})
        sentinel = object()
        with (
            patch.object(routes, "get_catalog_aggregator", return_value=aggregator),
            patch.object(routes, "_registry", sentinel),
        ):
            await routes.refresh_catalog()
            assert routes._registry is None

    @pytest.mark.asyncio
    async def test_get_with_refresh_invalidates(self, httpx_mock: HTTPXMock):
        from src.api import routes

        httpx_mock.add_response(
            method="GET",
            url="http://allocations:3000/tools",
            json={"tools": []},
        )
        aggregator = CatalogAggregator(server_urls={"allocations": "http://allocations:3000"})
        sentinel = object()
        with (
            patch.object(routes, "get_catalog_aggregator", return_value=aggregator),
            patch.object(routes, "_registry", sentinel),
        ):
            await routes.get_catalog(refresh=True)
            assert routes._registry is None

    @pytest.mark.asyncio
    async def test_get_without_refresh_keeps_registry(self, httpx_mock: HTTPXMock):
        from src.api import routes

        httpx_mock.add_response(
            method="GET",
            url="http://allocations:3000/tools",
            json={"tools": []},
        )
        aggregator = CatalogAggregator(server_urls={"allocations": "http://allocations:3000"})
        sentinel = object()
        with (
            patch.object(routes, "get_catalog_aggregator", return_value=aggregator),
            patch.object(routes, "_registry", sentinel),
        ):
            await routes.get_catalog(refresh=False)
            assert routes._registry is sentinel


class TestGetCatalogAggregator:
    """Tests for get_catalog_aggregator singleton."""

    def test_returns_same_instance(self):
        """Test that get_catalog_aggregator returns singleton."""
        agg1 = get_catalog_aggregator()
        agg2 = get_catalog_aggregator()
        assert agg1 is agg2
