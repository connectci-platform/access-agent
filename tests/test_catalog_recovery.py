"""Background recovery of MCP servers missing from the startup catalog (D8-3020).

On 2026-11-08 the agent started while MCP servers were restarting, cached a
catalog missing three of them, and nothing retried: those tools stayed gone
until a manual refresh. These tests pin the recovery loop and the atomic
catalog swap it relies on.
"""

import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from src.tools.registry import (
    RECOVERY_INITIAL_DELAY,
    RECOVERY_MAX_ATTEMPTS,
    RECOVERY_MAX_DELAY,
    CatalogAggregator,
    ToolRegistry,
)

URLS = {
    "allocations": "http://allocations:3000",
    "events": "http://events:3000",
    "compute-resources": "http://compute-resources:3000",
}


@pytest.fixture(autouse=True)
def _fresh_capability_registry():
    from src.agent.domains import capabilities as cap_module

    cap_module._registry = None
    yield
    cap_module._registry = None


class FakeServers:
    """Stands in for the MCP servers' /tools endpoints.

    ``up`` names the servers that answer; any other server raises a connect
    error, the way a restarting container does. ``calls`` records every
    server URL fetched, in order.
    """

    def __init__(self, up: set[str]):
        self.up = set(up)
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None

    async def __call__(self, client: httpx.AsyncClient, server_url: str) -> dict[str, Any]:
        self.calls.append(server_url)
        if self.gate is not None:
            await self.gate.wait()
        name = next(n for n, u in URLS.items() if u == server_url)
        if name not in self.up:
            raise httpx.ConnectError(f"{name} is restarting")
        return {"tools": [{"name": f"{name}__tool", "description": f"{name} tool"}]}

    def fetched(self, name: str) -> int:
        return self.calls.count(URLS[name])


def _aggregator(fake: FakeServers, urls: dict[str, str] | None = None, **kwargs: Any):
    kwargs.setdefault("recovery_initial_delay", 0)
    kwargs.setdefault("recovery_max_delay", 0)
    aggregator = CatalogAggregator(server_urls=urls or URLS, **kwargs)
    aggregator._fetch_server_tools = fake  # type: ignore[method-assign]
    return aggregator


async def _wait_for_recovery(aggregator: CatalogAggregator) -> None:
    task = aggregator.recovery_task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)


def _status(catalog: dict[str, Any]) -> dict[str, str]:
    return {s["server"]: s["status"] for s in catalog["servers"]}


class TestBackoff:
    def test_schedule_starts_short_and_caps_at_a_minute(self):
        aggregator = CatalogAggregator(server_urls=URLS)
        delays = [aggregator._recovery_delay(n) for n in range(RECOVERY_MAX_ATTEMPTS)]
        assert delays[:6] == [2, 4, 8, 16, 32, 60]
        assert max(delays) == RECOVERY_MAX_DELAY == 60
        assert delays[0] == RECOVERY_INITIAL_DELAY
        # Bounded: the loop gives up after roughly a quarter hour.
        assert 10 * 60 < sum(delays) < 20 * 60


class TestRecovery:
    async def test_full_catalog_starts_no_recovery(self):
        fake = FakeServers(up=set(URLS))
        aggregator = _aggregator(fake)
        await aggregator.fetch_catalog()
        assert aggregator.recovery_task is None

    async def test_server_recovers_and_its_tools_appear(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        startup = await aggregator.fetch_catalog()
        assert startup["servers_available"] == 1
        assert startup["total_servers"] == 3

        fake.up = set(URLS)
        await _wait_for_recovery(aggregator)

        catalog = aggregator.catalog
        assert set(_status(catalog).values()) == {"available"}
        assert catalog["servers_available"] == catalog["total_servers"] == 3
        assert catalog["total_tools"] == 3
        assert catalog["quick_lookup"]["events__tool"] == {
            "server": "events",
            "url": "http://events:3000/tools/events__tool",
        }
        assert aggregator.last_refresh is not None

    async def test_only_the_missing_servers_are_retried(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        await aggregator.fetch_catalog()
        fake.up = set(URLS)
        await _wait_for_recovery(aggregator)

        # One startup fetch each; the retry never re-fetches a healthy server.
        assert fake.fetched("allocations") == 1
        assert fake.fetched("events") == 2

    async def test_servers_recovering_at_different_times(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        await aggregator.fetch_catalog()

        seen: list[dict[str, str]] = []
        real_fetch = aggregator._fetch_servers

        async def fetch_then_bring_up(urls: dict[str, str]) -> list[Any]:
            results = await real_fetch(urls)
            seen.append(_status(aggregator.catalog))
            fake.up.add("events" if "events" not in fake.up else "compute-resources")
            return results

        aggregator._fetch_servers = fetch_then_bring_up  # type: ignore[method-assign]
        await _wait_for_recovery(aggregator)

        assert set(_status(aggregator.catalog).values()) == {"available"}
        # Recovered one server per attempt, each merged as it came back.
        assert len(seen) == 3

    async def test_server_that_never_recovers_gives_up_after_bounded_attempts(self, caplog):
        fake = FakeServers(up={"allocations", "events"})
        aggregator = _aggregator(fake, recovery_max_attempts=4)
        await aggregator.fetch_catalog()
        await _wait_for_recovery(aggregator)

        assert fake.fetched("compute-resources") == 1 + 4
        assert aggregator.recovery_task is not None
        assert aggregator.recovery_task.done()
        assert _status(aggregator.catalog)["compute-resources"] == "unavailable"
        assert "compute-resources" in caplog.text
        assert "giving up" in caplog.text

    async def test_no_task_is_leaked_after_giving_up(self):
        fake = FakeServers(up=set())
        aggregator = _aggregator(fake, recovery_max_attempts=2)
        await aggregator.fetch_catalog()
        await _wait_for_recovery(aggregator)

        pending = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
        ]
        assert pending == []

    async def test_a_second_fetch_does_not_start_a_second_loop(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake, recovery_initial_delay=60, recovery_max_delay=60)
        await aggregator.fetch_catalog()
        first = aggregator.recovery_task
        assert first is not None

        await aggregator.fetch_catalog(force_refresh=True)
        assert aggregator.recovery_task is first
        await aggregator.stop_recovery()

    async def test_a_refresh_after_giving_up_starts_a_new_loop(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake, recovery_max_attempts=1)
        await aggregator.fetch_catalog()
        first = aggregator.recovery_task
        await _wait_for_recovery(aggregator)

        await aggregator.fetch_catalog(force_refresh=True)
        assert aggregator.recovery_task is not first
        fake.up = set(URLS)
        await _wait_for_recovery(aggregator)
        assert aggregator.catalog["servers_available"] == 3

    async def test_refresh_during_recovery_is_not_overwritten(self):
        """A full refresh that lands while a retry is in flight wins for the
        servers it found available; the retry only fills what is still
        missing when it finishes."""
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        await aggregator.fetch_catalog()

        fake.gate = asyncio.Event()
        await asyncio.sleep(0.01)  # the retry is now parked on the gate
        assert not aggregator.recovery_task.done()  # type: ignore[union-attr]

        fake.up = set(URLS)
        refresh = asyncio.create_task(aggregator.fetch_catalog(force_refresh=True))
        await asyncio.sleep(0.01)
        fake.gate.set()
        refreshed = await refresh
        await _wait_for_recovery(aggregator)

        assert aggregator.catalog is refreshed
        assert refreshed["servers_available"] == 3

    async def test_an_unexpected_error_does_not_kill_the_loop(self, caplog):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        await aggregator.fetch_catalog()

        real_fetch = aggregator._fetch_servers
        calls = 0

        async def flaky(urls: dict[str, str]) -> list[Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("client blew up")
            fake.up = set(URLS)
            return await real_fetch(urls)

        aggregator._fetch_servers = flaky  # type: ignore[method-assign]
        await _wait_for_recovery(aggregator)

        assert aggregator.catalog["servers_available"] == 3
        assert "client blew up" in caplog.text


class TestShutdown:
    async def test_stop_recovery_cancels_the_task(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake, recovery_initial_delay=60, recovery_max_delay=60)
        await aggregator.fetch_catalog()
        task = aggregator.recovery_task
        assert task is not None and not task.done()

        await aggregator.stop_recovery()

        assert task.cancelled()
        assert aggregator.recovery_task is None

    async def test_stop_recovery_is_a_no_op_without_a_task(self):
        aggregator = CatalogAggregator(server_urls=URLS)
        await aggregator.stop_recovery()
        assert aggregator.recovery_task is None

    async def test_lifespan_shutdown_cancels_recovery(self):
        from src.main import app, lifespan
        from src.tools import registry as registry_module

        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake, recovery_initial_delay=60, recovery_max_delay=60)
        with patch.object(registry_module, "_aggregator", aggregator):
            async with lifespan(app):
                task = aggregator.recovery_task
                assert task is not None and not task.done()
            assert task.cancelled()
            assert aggregator.recovery_task is None


class TestAtomicSwap:
    async def test_recovery_replaces_the_catalog_object_and_never_mutates_the_old_one(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        before = await aggregator.fetch_catalog()
        snapshot = {
            "servers": [dict(s) for s in before["servers"]],
            "quick_lookup": dict(before["quick_lookup"]),
            "total_tools": before["total_tools"],
        }

        fake.up = set(URLS)
        await _wait_for_recovery(aggregator)

        assert aggregator.catalog is not before
        assert before["servers"] == snapshot["servers"]
        assert before["quick_lookup"] == snapshot["quick_lookup"]
        assert before["total_tools"] == snapshot["total_tools"]

    async def test_concurrent_readers_see_old_or_new_catalog_never_partial(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        await aggregator.fetch_catalog()
        fake.up = set(URLS)

        observed: list[tuple[int, int, int]] = []
        done = asyncio.Event()

        async def reader() -> None:
            while not done.is_set():
                c = aggregator.catalog
                available = [s for s in c["servers"] if s["status"] == "available"]
                lookup_servers = {i["server"] for i in c["quick_lookup"].values()}
                # Every snapshot is internally consistent.
                assert c["servers_available"] == len(available)
                assert c["total_tools"] == sum(s["tool_count"] for s in c["servers"])
                assert lookup_servers == {s["server"] for s in available}
                observed.append((c["servers_available"], c["total_servers"], c["total_tools"]))
                await asyncio.sleep(0)

        readers = [asyncio.create_task(reader()) for _ in range(5)]
        await _wait_for_recovery(aggregator)
        await asyncio.sleep(0)
        done.set()
        await asyncio.gather(*readers)

        assert set(observed) <= {(1, 3, 1), (3, 3, 3)}
        assert (3, 3, 3) in observed

    async def test_tool_registry_does_not_mutate_the_aggregator_catalog(self):
        from src.config import settings

        fake = FakeServers(up=set(URLS))
        aggregator = _aggregator(fake)
        catalog = await aggregator.fetch_catalog()
        servers_before = catalog["servers"]
        lookup_before = catalog["quick_lookup"]

        with patch.object(settings, "DISABLED_CAPABILITIES", "browse_resources"):
            from src.agent.domains import capabilities as cap_module

            cap_module._registry = None
            registry = ToolRegistry(catalog=catalog)

        assert "compute-resources__tool" not in registry.tools
        assert catalog["servers"] is servers_before
        assert catalog["quick_lookup"] is lookup_before
        assert "compute-resources" in {s["server"] for s in catalog["servers"]}


class TestCapabilityFilterAfterRecovery:
    async def test_disabled_server_is_never_retried_or_advertised(self):
        from src.config import settings

        urls = {**URLS, "xdmod-data": "http://xdmod-data:3000"}
        fake = FakeServers(up={"allocations"})
        with patch.object(settings, "DISABLED_CAPABILITIES", "extract_xdmod_data"):
            aggregator = CatalogAggregator(
                server_urls=urls, recovery_initial_delay=0, recovery_max_delay=0
            )

            async def fetch(client: httpx.AsyncClient, server_url: str) -> dict[str, Any]:
                if server_url == urls["xdmod-data"]:
                    fake.calls.append(server_url)
                    return {"tools": [{"name": "xdmod-data__tool", "description": ""}]}
                return await fake(client, server_url)

            aggregator._fetch_server_tools = fetch  # type: ignore[method-assign]
            await aggregator.fetch_catalog()
            fake.up = set(URLS)
            await _wait_for_recovery(aggregator)

        catalog = aggregator.catalog
        assert "xdmod-data" not in _status(catalog)
        assert "xdmod-data__tool" not in catalog["quick_lookup"]
        assert catalog["total_servers"] == catalog["servers_available"] == 3
        # Fetched once at startup (then filtered out), never retried.
        assert fake.calls.count(urls["xdmod-data"]) == 1


class TestRegistryPicksUpRecovery:
    """The agent's ToolRegistry is cached in routes; a swapped catalog must
    reach the next request without a manual refresh."""

    async def test_next_request_after_recovery_sees_recovered_tools(self):
        from src.api import routes

        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        with (
            patch.object(routes, "get_catalog_aggregator", return_value=aggregator),
            patch.object(routes, "_registry", None),
            patch.object(routes, "_registry_catalog", None),
        ):
            await aggregator.fetch_catalog()
            first = await routes.get_registry()
            assert set(first.tools) == {"allocations__tool"}
            assert await routes.get_registry() is first

            fake.up = set(URLS)
            await _wait_for_recovery(aggregator)

            second = await routes.get_registry()
            assert second is not first
            assert set(second.tools) == {
                "allocations__tool",
                "events__tool",
                "compute-resources__tool",
            }
            assert second.catalog["servers_available"] == 3

    async def test_lifespan_recovery_reaches_health(self):
        from src.api import routes

        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake)
        with patch.object(routes, "get_catalog_aggregator", return_value=aggregator):
            await aggregator.fetch_catalog()
            degraded = await routes.health_check()
            assert degraded["status"] == "degraded"
            assert set(degraded["tools"]["unavailable_servers"]) == {
                "events",
                "compute-resources",
            }

            fake.up = set(URLS)
            await _wait_for_recovery(aggregator)
            healthy = await routes.health_check()
            assert healthy["status"] == "healthy"


class TestRecoveryEdges:
    async def test_loop_exits_without_fetching_when_a_refresh_restored_everything(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake, recovery_initial_delay=0.05, recovery_max_delay=0.05)
        await aggregator.fetch_catalog()
        task = aggregator.recovery_task

        fake.up = set(URLS)
        await aggregator.fetch_catalog(force_refresh=True)
        calls_after_refresh = len(fake.calls)
        await _wait_for_recovery(aggregator)

        assert aggregator.recovery_task is task
        assert len(fake.calls) == calls_after_refresh

    async def test_stop_recovery_after_the_loop_finished(self):
        fake = FakeServers(up={"allocations"})
        aggregator = _aggregator(fake, recovery_max_attempts=1)
        await aggregator.fetch_catalog()
        await _wait_for_recovery(aggregator)

        await aggregator.stop_recovery()
        assert aggregator.recovery_task is None
