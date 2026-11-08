"""Tests for the server-side allocated-resources profile (D8-3028).

The profile is built from the allocations MCP tool ``get_my_rp_accounts``,
called as the signed-in user, and never from anything the browser sends.
"""

import asyncio
from typing import Any

import pytest

from src.agent import profile_source
from src.agent.profile import AllocatedResource, UserProfile, render_profile_section
from src.agent.profile_source import allocated_profile_for, resources_from_accounts, with_slugs
from src.tools.mcp_client import MCPToolResult

GROUPS = {
    "delta-gpu.ncsa.access-ci.org": ["delta"],
    "delta-cpu.ncsa.access-ci.org": ["delta"],
    "bridges2-gpu.psc.access-ci.org": ["bridges-2"],
    "shared.example.org": ["anvil", "expanse"],
    "weird.example.org": ["--"],
    "long.example.org": ["a" * 40],
}


def _slugs_for(resource_id: str) -> list[str]:
    return GROUPS.get(resource_id, [])


def _map(data: Any) -> list[AllocatedResource] | None:
    resources = resources_from_accounts(data)
    return None if resources is None else with_slugs(resources, _slugs_for)


def _account(name: Any = "NCSA Delta GPU", resource_id: Any = "delta-gpu.ncsa.access-ci.org"):
    return {
        "resource_id": resource_id,
        "rp_display_name": name,
        "rp_username": "alice",
        "account_state": "active",
        "grants": [{"project_balance": 5000, "billable_unit": "GPU hours"}],
    }


def _payload(accounts: list[Any], state: str = "rows_fresh") -> dict[str, Any]:
    return {"accounts": accounts, "state": state, "synced_at": "2026-11-01T00:00:00+00:00"}


# ── Mapping ──────────────────────────────────────────────────────────────


def test_maps_an_account_to_a_resource_with_its_group_slug():
    assert _map(_payload([_account()])) == [
        AllocatedResource(
            name="NCSA Delta GPU", rp_slug="delta", resource_id="delta-gpu.ncsa.access-ci.org"
        )
    ]


def test_a_hyphenated_group_slug_takes_the_form_the_doc_search_sends():
    """Drupal's slug for Bridges-2 is 'bridges-2'; search_access_documents
    sends 'bridges2' for it, and that is what the prompt must name."""
    (resource,) = _map(_payload([_account("Bridges-2 GPU", "bridges2-gpu.psc.access-ci.org")]))
    assert resource.rp_slug == "bridges2"


@pytest.mark.parametrize(
    "resource_id",
    [
        "ungrouped.example.org",  # in no group
        "shared.example.org",  # in two groups: ambiguous
        "weird.example.org",  # group slug that normalizes to nothing
        "long.example.org",  # group slug longer than the field allows
    ],
)
def test_an_unmappable_resource_keeps_its_name_without_a_slug(resource_id):
    (resource,) = _map(_payload([_account("Some Storage", resource_id)]))
    assert resource == AllocatedResource(name="Some Storage", rp_slug=None, resource_id=resource_id)


def test_the_slug_never_comes_from_the_display_name():
    """A name that would normalize to a real slug ('delta') gets none without
    a resource id that maps to the group."""
    (resource,) = _map(_payload([_account("Delta", None)]))
    assert resource == AllocatedResource(name="Delta", rp_slug=None, resource_id=None)


@pytest.mark.parametrize("resource_id", ["bad id/with spaces", "x" * 129, 42, ""])
def test_an_invalid_resource_id_is_dropped_and_so_is_the_slug(resource_id):
    (resource,) = _map(_payload([_account("Delta GPU", resource_id)]))
    assert resource == AllocatedResource(name="Delta GPU", rp_slug=None, resource_id=None)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Jetstream2 & Storage", "Jetstream2 Storage"),
        ("  -Delta GPU  ", "Delta GPU"),
        ("Délta\nGPU", "D lta GPU"),
        ("A" * 80, "A" * 64),
        ("Expanse " + "B" * 60, "Expanse " + "B" * 56),
    ],
)
def test_a_name_with_characters_the_field_rejects_is_sanitized(raw, clean):
    (resource,) = _map(_payload([_account(raw)]))
    assert resource.name == clean


def test_a_name_truncated_onto_a_space_loses_the_trailing_space():
    raw = "A" * 63 + " tail"
    (resource,) = _map(_payload([_account(raw)]))
    assert resource.name == "A" * 63


@pytest.mark.parametrize("name", [None, 7, "", "   ", "&&&", "<<>>"])
def test_a_row_without_a_usable_name_is_dropped(name):
    payload = _payload([_account(name), _account("NCSA Delta CPU", "delta-cpu.ncsa.access-ci.org")])
    assert [r.name for r in _map(payload)] == ["NCSA Delta CPU"]


@pytest.mark.parametrize("row", ["a string", 3, None, ["list"]])
def test_a_row_that_is_not_an_object_is_dropped(row):
    payload = _payload([row, _account()])
    assert [r.name for r in _map(payload)] == ["NCSA Delta GPU"]


def test_at_most_32_resources_are_kept():
    accounts = [_account(f"Resource {i}", f"r{i}.example.org") for i in range(40)]
    resources = _map(_payload(accounts))
    assert [r.name for r in resources] == [f"Resource {i}" for i in range(32)]
    UserProfile(allocated_resources=resources)  # still a valid profile


def test_a_stale_list_still_counts():
    assert len(_map(_payload([_account()], "rows_stale"))) == 1


def test_a_fresh_empty_list_means_no_allocations():
    assert _map(_payload([], "no_rows_fresh")) == []


@pytest.mark.parametrize(
    "data",
    [
        _payload([], "syncing"),
        _payload([_account()], "syncing"),
        _payload([], "rows_fresh"),  # rows exist but none could be resolved
        _payload([], "error"),
        _payload([_account()], "something_new"),
        {"accounts": [_account()]},  # no state
        {"state": "rows_fresh", "accounts": "nope"},
        {"data": {"state": "syncing", "accounts": []}},
        "The request succeeded but returned no data. It may still be syncing.",
        None,
        ["not", "an", "object"],
        {"error": {"code": "unauthenticated", "message": "Session expired"}},
    ],
)
def test_a_response_that_does_not_say_what_the_user_holds_is_unknown(data):
    assert _map(data) is None


def test_accounts_that_all_fail_validation_are_unknown_not_none():
    payload = _payload([_account(None), "junk"])
    assert _map(payload) is None


# ── Lookup, cache, and failure modes ─────────────────────────────────────


class _FakeRPCache:
    def __init__(self, slugs: list[str] | None = None) -> None:
        self.slugs = ["delta", "bridges-2"] if slugs is None else slugs
        self.loads = 0

    async def ensure_loaded(self) -> None:  # must never be awaited on the request path
        raise AssertionError("profile lookup waited on the group vocabulary")

    def refresh_in_background(self) -> None:
        self.loads += 1

    def list_slugs(self) -> list[str]:
        return self.slugs

    def slugs_for_resource_id(self, resource_id: str) -> list[str]:
        return _slugs_for(resource_id)


class _FakeMCP:
    """Stands in for MCPClient; records every call."""

    calls: list[tuple[str, str, dict[str, Any], str | None]]
    result: MCPToolResult | Exception
    delay: float = 0.0

    def __init__(self) -> None:
        pass

    async def call_tool(self, server, tool_name, arguments, acting_user=None):
        type(self).calls.append((server, tool_name, arguments, acting_user))
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(profile_source, "_clock", lambda: now[0])
    return now


@pytest.fixture
def rp_cache(monkeypatch):
    cache = _FakeRPCache()
    monkeypatch.setattr(profile_source, "get_rp_cache", lambda: cache)
    return cache


@pytest.fixture
def mcp(monkeypatch, rp_cache, clock):
    fake = type("FakeMCP", (_FakeMCP,), {})
    fake.calls = []
    fake.result = MCPToolResult(success=True, data=_payload([_account()]))
    fake.delay = 0.0
    monkeypatch.setattr(profile_source, "MCPClient", fake)
    profile_source._reset_cache_for_test()
    yield fake
    profile_source._reset_cache_for_test()


DELTA_PROFILE = UserProfile(
    allocated_resources=[
        AllocatedResource(
            name="NCSA Delta GPU", rp_slug="delta", resource_id="delta-gpu.ncsa.access-ci.org"
        )
    ]
)


async def test_an_anonymous_run_makes_no_lookup(mcp):
    assert await allocated_profile_for(None) is None
    assert mcp.calls == []


async def test_a_signed_in_run_calls_the_tool_as_that_user(mcp, rp_cache):
    assert await allocated_profile_for("alice@access-ci.org") == DELTA_PROFILE
    assert mcp.calls == [("allocations", "get_my_rp_accounts", {}, "alice@access-ci.org")]
    assert rp_cache.loads == 1


async def test_a_user_with_no_accounts_gets_an_empty_list(mcp):
    mcp.result = MCPToolResult(success=True, data=_payload([], "no_rows_fresh"))
    assert await allocated_profile_for("alice@x") == UserProfile(allocated_resources=[])


async def test_a_second_message_within_the_lifetime_hits_the_cache(mcp, clock):
    await allocated_profile_for("alice@x")
    clock[0] += profile_source.SUCCESS_TTL_S - 1
    assert await allocated_profile_for("alice@x") == DELTA_PROFILE
    assert len(mcp.calls) == 1


async def test_the_cached_profile_expires(mcp, clock):
    await allocated_profile_for("alice@x")
    clock[0] += profile_source.SUCCESS_TTL_S + 1
    mcp.result = MCPToolResult(success=True, data=_payload([], "no_rows_fresh"))
    assert await allocated_profile_for("alice@x") == UserProfile(allocated_resources=[])
    assert len(mcp.calls) == 2


async def test_the_cache_is_per_user(mcp):
    await allocated_profile_for("alice@x")
    mcp.result = MCPToolResult(success=True, data=_payload([], "no_rows_fresh"))

    assert await allocated_profile_for("bob@x") == UserProfile(allocated_resources=[])
    assert await allocated_profile_for("alice@x") == DELTA_PROFILE
    assert [c[3] for c in mcp.calls] == ["alice@x", "bob@x"]


async def test_the_cache_holds_a_bounded_number_of_users(mcp, monkeypatch):
    monkeypatch.setattr(profile_source, "MAX_CACHED_USERS", 2)
    for user in ("a@x", "b@x", "c@x"):
        await allocated_profile_for(user)
    await allocated_profile_for("a@x")  # evicted as the oldest, so looked up again
    await allocated_profile_for("c@x")  # still cached

    assert [c[3] for c in mcp.calls] == ["a@x", "b@x", "c@x", "a@x"]


async def test_the_lookup_is_bounded_by_a_short_timeout(mcp, monkeypatch):
    monkeypatch.setattr(profile_source, "LOOKUP_TIMEOUT_S", 0.01)
    mcp.delay = 5.0
    assert await allocated_profile_for("alice@x") is None


def test_the_timeout_is_a_few_seconds_at_most():
    assert 0 < profile_source.LOOKUP_TIMEOUT_S <= 5


@pytest.mark.parametrize(
    "result",
    [
        MCPToolResult(success=False, error="HTTP 403: forbidden"),
        MCPToolResult(success=False, error="Timeout", timed_out=True),
        MCPToolResult(success=True, data=_payload([], "syncing")),
        MCPToolResult(success=True, data="not json"),
        RuntimeError("boom"),
    ],
    ids=["error", "timed-out", "syncing", "garbage", "raises"],
)
async def test_every_failure_falls_back_to_no_profile(mcp, result):
    mcp.result = result
    assert await allocated_profile_for("alice@x") is None


async def test_without_the_group_vocabulary_the_profile_has_no_slugs(mcp, rp_cache):
    rp_cache.slugs = []
    rp_cache.slugs_for_resource_id = lambda _rid: []
    assert await allocated_profile_for("alice@x") == UNSLUGGED_DELTA


async def test_a_failure_is_retried_after_a_short_pause_not_every_message(mcp, clock):
    mcp.result = MCPToolResult(success=False, error="down")
    await allocated_profile_for("alice@x")
    clock[0] += profile_source.FAILURE_TTL_S - 1
    assert await allocated_profile_for("alice@x") is None
    assert len(mcp.calls) == 1

    clock[0] += 2
    mcp.result = MCPToolResult(success=True, data=_payload([_account()]))
    assert await allocated_profile_for("alice@x") == DELTA_PROFILE
    assert len(mcp.calls) == 2


def test_failures_are_retried_sooner_than_successes_expire():
    assert profile_source.FAILURE_TTL_S < profile_source.SUCCESS_TTL_S <= 600


async def test_a_cancelled_run_cancels_the_lookup(mcp):
    mcp.delay = 5.0
    task = asyncio.create_task(allocated_profile_for("alice@x"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_failures_do_not_log_account_details(mcp, caplog):
    mcp.result = MCPToolResult(success=False, error="HTTP 500: alice holds NCSA Delta GPU")
    with caplog.at_level("DEBUG", logger="src.agent.profile_source"):
        await allocated_profile_for("alice@x")

    assert caplog.records
    for record in caplog.records:
        text = record.getMessage()
        assert "Delta" not in text
        assert "alice" not in text


# ── Vocabulary readiness, single-flight, LRU (review fixes) ──────────────


@pytest.fixture
def real_rp_cache(monkeypatch):
    """A real RPSectionCache whose Drupal fetch the test controls."""
    from src.services.rp_cache import RPInfo, RPSectionCache

    cache = RPSectionCache()
    release = asyncio.Event()
    fetches: list[int] = []

    async def _fetch():
        fetches.append(1)
        await release.wait()
        return {
            "delta": RPInfo(
                slug="delta", title="Delta", resource_ids=["delta-gpu.ncsa.access-ci.org"]
            )
        }

    cache._fetch_resources = _fetch
    monkeypatch.setattr(profile_source, "get_rp_cache", lambda: cache)
    cache.release = release
    cache.fetches = fetches
    yield cache
    release.set()


UNSLUGGED_DELTA = UserProfile(
    allocated_resources=[
        AllocatedResource(name="NCSA Delta GPU", resource_id="delta-gpu.ncsa.access-ci.org")
    ],
    groups_known=False,
)


async def test_a_slow_vocabulary_does_not_hold_up_or_spoil_the_lookup(mcp, real_rp_cache):
    """The groups fetch is still running: the profile comes back at once,
    without slugs, and gains them once the vocabulary loads, with no second
    allocations call."""
    profile = await asyncio.wait_for(allocated_profile_for("alice@x"), timeout=1)
    assert profile == UNSLUGGED_DELTA
    assert real_rp_cache.fetches == [1]

    real_rp_cache.release.set()
    await real_rp_cache._task

    assert await allocated_profile_for("alice@x") == DELTA_PROFILE
    assert len(mcp.calls) == 1
    assert real_rp_cache.fetches == [1]


async def test_concurrent_first_lookups_during_a_refresh_still_get_a_profile(mcp, real_rp_cache):
    real_rp_cache.refresh_in_background()
    await asyncio.sleep(0)
    assert real_rp_cache._refreshing

    a, b = await asyncio.gather(allocated_profile_for("a@x"), allocated_profile_for("b@x"))

    assert a == b == UNSLUGGED_DELTA
    assert real_rp_cache.fetches == [1]


async def test_a_failed_vocabulary_refresh_still_gives_a_profile(mcp, real_rp_cache):
    async def _fail():
        raise RuntimeError("bad url")

    real_rp_cache._fetch_resources = _fail
    real_rp_cache.refresh_in_background()
    await real_rp_cache._task

    assert await allocated_profile_for("alice@x") == UNSLUGGED_DELTA


async def test_concurrent_misses_for_one_user_share_one_call(mcp):
    mcp.delay = 0.05
    results = await asyncio.gather(*(allocated_profile_for("alice@x") for _ in range(5)))

    assert results == [DELTA_PROFILE] * 5
    assert len(mcp.calls) == 1
    assert profile_source._inflight == {}


async def test_one_cancelled_waiter_does_not_cancel_the_shared_lookup(mcp):
    mcp.delay = 0.05
    first = asyncio.create_task(allocated_profile_for("alice@x"))
    second = asyncio.create_task(allocated_profile_for("alice@x"))
    await asyncio.sleep(0.01)
    first.cancel()

    assert await second == DELTA_PROFILE
    with pytest.raises(asyncio.CancelledError):
        await first
    assert len(mcp.calls) == 1


async def test_eviction_drops_the_least_recently_used_user(mcp, monkeypatch):
    monkeypatch.setattr(profile_source, "MAX_CACHED_USERS", 2)
    await allocated_profile_for("a@x")
    await allocated_profile_for("b@x")
    await allocated_profile_for("a@x")  # a is now the most recently used
    await allocated_profile_for("c@x")  # evicts b, not a
    await allocated_profile_for("a@x")
    await allocated_profile_for("b@x")

    assert [c[3] for c in mcp.calls] == ["a@x", "b@x", "c@x", "b@x"]


async def test_the_profile_says_whether_the_group_list_was_loaded(mcp, real_rp_cache):
    before = await allocated_profile_for("alice@x")
    assert before.groups_known is False
    assert "No scoped documentation" not in render_profile_section(before)

    real_rp_cache.release.set()
    await real_rp_cache._task
    mcp.result = MCPToolResult(
        success=True, data=_payload([_account("Delta Storage", "ungrouped.example.org")])
    )
    after = await allocated_profile_for("bob@x")
    assert after.groups_known is True
    assert "No scoped documentation exists for Delta Storage" in render_profile_section(after)
