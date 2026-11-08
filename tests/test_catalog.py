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
    async def test_the_hint_survives_from_fetch_to_the_wrappers(self, httpx_mock: HTTPXMock):
        """Fetched JSON → aggregator → capability filter → the loop's wrappers."""
        from src.agent.domains.tools import create_mcp_tools_from_catalog

        def tool(name, hint):
            return {"name": name, "inputSchema": {}, "annotations": {"readOnlyHint": hint}}

        httpx_mock.add_response(
            method="GET",
            url="http://events:3000/tools",
            json={"tools": [tool("toggle_alerts", False), tool("search_events", True)]},
        )
        httpx_mock.add_response(
            method="GET", url="http://allocations:3000/tools", json={"tools": []}
        )

        catalog = await CatalogAggregator(server_urls=TEST_SERVER_URLS).fetch_catalog()
        wrappers = {w.name: w for w in create_mcp_tools_from_catalog(catalog, "u@x")}

        assert wrappers["toggle_alerts"].read_only is False
        assert wrappers["search_events"].read_only is True

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
    async def test_compute_resources_is_owned_and_retained(self, httpx_mock: HTTPXMock):
        """compute-resources had no owning capability until 2026-09-03 —
        the agent silently lost search_resources/get_resource_hardware. Pin
        the ownership so the fail-closed filter can't orphan it again."""
        from src.agent.domains.capabilities import get_capability_registry

        assert "compute-resources" in get_capability_registry().enabled_mcp_servers()

        urls = {"compute-resources": "http://compute-resources:3000"}
        self._mock_both(httpx_mock, urls)
        aggregator = CatalogAggregator(server_urls=urls)
        catalog = await aggregator.fetch_catalog()
        assert "compute-resources" in [s["server"] for s in catalog["servers"]]

    @pytest.mark.asyncio
    async def test_all_servers_filtered_reports_healthy(self, httpx_mock: HTTPXMock):
        """Degenerate pin: with every fetched server capability-filtered,
        counts are 0/0 and /health stays "healthy" — disabled-by-config is
        an operator choice, not an outage. (A genuinely down enabled server
        still degrades; see test_down_enabled_server_still_degrades_health.)"""
        from src.api import routes
        from src.config import settings

        urls = {"xdmod-data": "http://xdmod-data:3000"}
        self._mock_both(httpx_mock, urls)

        with patch.object(settings, "DISABLED_CAPABILITIES", "extract_xdmod_data"):
            aggregator = CatalogAggregator(server_urls=urls)
            catalog = await aggregator.fetch_catalog()
            with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
                result = await routes.health_check()

        assert catalog["total_servers"] == catalog["servers_available"] == 0
        assert result["status"] == "healthy"

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


class TestRefreshReachesRegistry:
    """Catalog refreshes must reach the agent's cached ToolRegistry: the next
    get_registry() rebuilds from the refreshed catalog (#240 — previously a
    refresh only updated the public /catalog payload)."""

    URL = "http://allocations:3000"

    def _respond(self, httpx_mock: HTTPXMock, tool: str | None) -> None:
        if tool is None:
            httpx_mock.add_response(method="GET", url=f"{self.URL}/tools", status_code=500)
        else:
            httpx_mock.add_response(
                method="GET",
                url=f"{self.URL}/tools",
                json={"tools": [{"name": tool, "description": tool}]},
            )

    def _aggregator(self) -> CatalogAggregator:
        # Long retry delay: these tests are about refreshes, not recovery.
        return CatalogAggregator(server_urls={"allocations": self.URL}, recovery_initial_delay=3600)

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        from src.api import routes

        with (
            patch.object(routes, "_registry", None),
            patch.object(routes, "_registry_catalog", None),
        ):
            yield

    @pytest.mark.asyncio
    async def test_post_refresh_rebuilds(self, httpx_mock: HTTPXMock):
        from src.api import routes

        self._respond(httpx_mock, "old_tool")
        self._respond(httpx_mock, "new_tool")
        aggregator = self._aggregator()
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            before = await routes.get_registry()
            await routes.refresh_catalog()
            after = await routes.get_registry()
        assert set(before.tools) == {"old_tool"}
        assert set(after.tools) == {"new_tool"}

    @pytest.mark.asyncio
    async def test_get_with_refresh_rebuilds(self, httpx_mock: HTTPXMock):
        from src.api import routes

        self._respond(httpx_mock, "old_tool")
        self._respond(httpx_mock, "new_tool")
        aggregator = self._aggregator()
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            await routes.get_registry()
            await routes.get_catalog(refresh=True)
            after = await routes.get_registry()
        assert set(after.tools) == {"new_tool"}

    @pytest.mark.asyncio
    async def test_failed_refresh_keeps_registry(self):
        """A refresh whose fetch raises must 500 and leave the cached
        registry intact — never invalidate on failure."""
        from fastapi import HTTPException

        from src.api import routes

        class _BoomAggregator:
            async def fetch_catalog(self, force_refresh: bool = False):
                raise RuntimeError("all servers on fire")

        sentinel = object()
        with (
            patch.object(routes, "get_catalog_aggregator", return_value=_BoomAggregator()),
            patch.object(routes, "_registry", sentinel),
        ):
            with pytest.raises(HTTPException):
                await routes.refresh_catalog()
            assert routes._registry is sentinel

    @pytest.mark.asyncio
    async def test_blip_refresh_keeps_registry(self, httpx_mock: HTTPXMock):
        """A refresh that catches every MCP server down (per-server failures
        don't fail the fetch) must not be adopted into the agent — an
        unauthenticated refresh timed during a blip would otherwise strip
        the agent's tools until the next refresh or restart."""
        from src.api import routes

        self._respond(httpx_mock, "good_tool")
        self._respond(httpx_mock, None)
        aggregator = self._aggregator()
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            before = await routes.get_registry()
            await routes.refresh_catalog()
            after = await routes.get_registry()
            await aggregator.stop_recovery()
        assert after is before
        assert set(after.tools) == {"good_tool"}

    @pytest.mark.asyncio
    async def test_all_down_at_startup_still_builds_a_registry(self, httpx_mock: HTTPXMock):
        """The blip guard only protects a registry we already have."""
        from src.api import routes

        self._respond(httpx_mock, None)
        aggregator = self._aggregator()
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            registry = await routes.get_registry()
            await aggregator.stop_recovery()
        assert registry.tools == {}

    @pytest.mark.asyncio
    async def test_rebuilt_registry_reflects_refreshed_catalog(self, httpx_mock: HTTPXMock):
        """End to end: after a successful refresh, the rebuilt registry
        serves the refreshed, capability-filtered catalog — the disabled
        server's tools are gone from what the agent sees."""
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
            with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
                await routes.refresh_catalog()
                rebuilt = await routes.get_registry()
                assert "allocations__tool" in rebuilt.tools
                assert "xdmod-data__tool" not in rebuilt.tools

    @pytest.mark.asyncio
    async def test_get_without_refresh_keeps_registry(self, httpx_mock: HTTPXMock):
        from src.api import routes

        self._respond(httpx_mock, "tool")
        aggregator = self._aggregator()
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            before = await routes.get_registry()
            await routes.get_catalog(refresh=False)
            assert await routes.get_registry() is before


class TestGetCatalogAggregator:
    """Tests for get_catalog_aggregator singleton."""

    def test_returns_same_instance(self):
        """Test that get_catalog_aggregator returns singleton."""
        agg1 = get_catalog_aggregator()
        agg2 = get_catalog_aggregator()
        assert agg1 is agg2
