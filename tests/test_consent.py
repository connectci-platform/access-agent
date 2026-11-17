"""The signed-in user's ACCESS Assistant personalization consent (D8-3124).

Read from Drupal through the allocations server's unlisted tools. Anything
that can't tell yields ``None``, which the routes treat as "no profile, no
offer".
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.agent import consent
from src.agent.consent import Consent, consent_for, record_consent, should_offer
from src.tools.mcp_client import MCPToolResult

USER = "alice@access-ci.org"
NOW = datetime(2026, 11, 16, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset():
    consent._reset_cache_for_test()
    yield
    consent._reset_cache_for_test()


class _FakeClient:
    def __init__(self, *results: MCPToolResult, delay: float = 0.0) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.delay = delay

    async def call_tool(self, server, tool, args, acting_user=None, **_):
        self.calls.append((server, tool, args, acting_user))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


def _ok(data: Any) -> MCPToolResult:
    return MCPToolResult(success=True, data=data)


@pytest.fixture
def client(monkeypatch):
    def install(*results: MCPToolResult, delay: float = 0.0) -> _FakeClient:
        fake = _FakeClient(*results, delay=delay)
        monkeypatch.setattr(consent, "MCPClient", lambda: fake)
        return fake

    return install


async def test_reads_state_and_ask_date_as_the_user(client):
    fake = client(_ok({"consent": "unset", "asked_at": "2026-10-01T00:00:00Z"}))
    got = await consent_for(USER)
    assert got == Consent(state="unset", asked_at=datetime(2026, 10, 1, tzinfo=UTC))
    assert fake.calls == [("allocations", "get_assistant_consent", {}, USER)]


async def test_anonymous_makes_no_call(client):
    fake = client(_ok({"consent": "allow", "asked_at": None}))
    assert await consent_for(None) is None
    assert fake.calls == []


@pytest.mark.parametrize(
    "result",
    [
        MCPToolResult(success=False, error="boom"),
        _ok({"consent": "maybe"}),
        _ok({"consent": "allow", "asked_at": "not a date"}),
        _ok("text"),
    ],
)
async def test_anything_unreadable_is_none(client, result):
    client(result)
    assert await consent_for(USER) is None


async def test_a_slow_lookup_is_none(client, monkeypatch):
    monkeypatch.setattr(consent, "LOOKUP_TIMEOUT_S", 0.01)
    client(_ok({"consent": "allow", "asked_at": None}), delay=0.5)
    assert await consent_for(USER) is None


async def test_cached_until_recorded(client):
    fake = client(
        _ok({"consent": "unset", "asked_at": None}),
        _ok({"consent": "allow", "asked_at": "2026-11-16T12:00:00Z"}),
    )
    assert (await consent_for(USER)).state == "unset"
    assert (await consent_for(USER)).state == "unset"
    assert len(fake.calls) == 1

    recorded = await record_consent(USER, "allow")
    assert recorded == Consent(state="allow", asked_at=NOW)
    assert fake.calls[-1] == ("allocations", "set_assistant_consent", {"consent": "allow"}, USER)
    # The write's answer is what the next read sees, without another call.
    assert await consent_for(USER) == recorded
    assert len(fake.calls) == 2


async def test_a_failed_write_forgets_the_cached_state(client):
    fake = client(
        _ok({"consent": "allow", "asked_at": None}),
        MCPToolResult(success=False, error="409"),
        _ok({"consent": "deny", "asked_at": None}),
    )
    await consent_for(USER)
    assert await record_consent(USER, "not_now") is None
    assert (await consent_for(USER)).state == "deny"
    assert len(fake.calls) == 3


async def test_record_refuses_other_values(client):
    fake = client(_ok({}))
    with pytest.raises(ValueError):
        await record_consent(USER, "allow_once")
    assert fake.calls == []


@pytest.mark.parametrize(
    ("state", "asked", "expected"),
    [
        ("unset", None, True),
        ("unset", NOW - timedelta(days=30), True),
        ("unset", NOW - timedelta(days=29, hours=23), False),
        ("allow", None, False),
        ("deny", None, False),
    ],
)
def test_should_offer_only_unset_and_not_asked_in_30_days(state, asked, expected):
    assert should_offer(Consent(state=state, asked_at=asked), now=NOW) is expected


def test_should_offer_never_without_a_state():
    assert should_offer(None, now=NOW) is False


@pytest.mark.parametrize("asked", [12345, ["2026-01-01"]])
async def test_a_non_string_ask_date_is_unreadable(client, asked):
    client(_ok({"consent": "unset", "asked_at": asked}))
    assert await consent_for(USER) is None


async def test_an_ask_date_without_a_zone_is_utc(client):
    client(_ok({"consent": "unset", "asked_at": "2026-10-01T08:00:00"}))
    assert (await consent_for(USER)).asked_at == datetime(2026, 10, 1, 8, tzinfo=UTC)


async def test_a_client_error_is_none(monkeypatch):
    class _Broken:
        async def call_tool(self, *a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(consent, "MCPClient", _Broken)
    assert await consent_for(USER) is None


async def test_the_cache_keeps_only_the_most_recent_users(client, monkeypatch):
    monkeypatch.setattr(consent, "MAX_CACHED_USERS", 2)
    fake = client(_ok({"consent": "allow", "asked_at": None}))
    for user in ("a@x", "b@x", "c@x"):
        await consent_for(user)
    await consent_for("c@x")
    await consent_for("a@x")
    # c was still cached; a had been dropped to make room for c.
    assert [call[3] for call in fake.calls] == ["a@x", "b@x", "c@x", "a@x"]
