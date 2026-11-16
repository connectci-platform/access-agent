"""When a signed-in run is personalized, and when the user is offered it (D8-3124)."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.agent import personalization
from src.agent.consent import Consent
from src.agent.personalization import offer_after_turn, personalized_profile
from src.agent.profile import AllocatedResource, UserProfile

USER = "alice@access-ci.org"
NOW = datetime(2026, 11, 16, 12, tzinfo=UTC)
PROFILE = UserProfile(allocated_resources=[AllocatedResource(name="Delta GPU")])


@pytest.fixture
def lookups(monkeypatch):
    calls: dict[str, list[Any]] = {"consent": [], "profile": [], "record": []}

    def install(consent: Consent | None, recorded: Consent | None = None):
        async def consent_for(user):
            calls["consent"].append(user)
            return consent

        async def allocated_profile_for(user):
            calls["profile"].append(user)
            return PROFILE

        async def record_consent(user, choice):
            calls["record"].append((user, choice))
            return recorded

        monkeypatch.setattr(personalization, "consent_for", consent_for)
        monkeypatch.setattr(personalization, "allocated_profile_for", allocated_profile_for)
        monkeypatch.setattr(personalization, "record_consent", record_consent)
        return calls

    return install


def _c(state, asked=None):
    return Consent(state=state, asked_at=asked)


async def test_allow_gets_the_profile(lookups):
    calls = lookups(_c("allow"))
    assert await personalized_profile(USER, once=False) == (_c("allow"), PROFILE)
    assert calls["profile"] == [USER]


@pytest.mark.parametrize("once", [False, True])
async def test_deny_never_looks_up_allocations(lookups, once):
    calls = lookups(_c("deny"))
    assert await personalized_profile(USER, once=once) == (_c("deny"), None)
    assert calls["profile"] == []


async def test_unset_gets_it_only_for_allow_once(lookups):
    calls = lookups(_c("unset"))
    assert await personalized_profile(USER, once=False) == (_c("unset"), None)
    assert calls["profile"] == []
    assert await personalized_profile(USER, once=True) == (_c("unset"), PROFILE)


@pytest.mark.parametrize("once", [False, True])
async def test_unknown_consent_is_no_profile(lookups, once):
    calls = lookups(None)
    assert await personalized_profile(USER, once=once) == (None, None)
    assert calls["profile"] == []


async def test_anonymous_makes_no_lookups(lookups):
    calls = lookups(_c("allow"))
    assert await personalized_profile(None, once=True) == (None, None)
    assert calls == {"consent": [], "profile": [], "record": []}


def _state(*servers: str) -> dict[str, Any]:
    return {
        "tool_results": [
            {"step_id": str(i), "tool_name": "t", "server": s, "success": True}
            for i, s in enumerate(servers)
        ]
    }


async def test_offers_after_a_resource_question_and_records_the_ask(lookups):
    calls = lookups(_c("unset"), recorded=_c("unset", NOW))
    offered = await offer_after_turn(
        USER, _c("unset"), _state("compute-resources"), once=False, now=NOW
    )
    assert offered is True
    assert calls["record"] == [(USER, "not_now")]


@pytest.mark.parametrize("server", ["compute-resources", "software-discovery"])
async def test_resource_and_software_tools_count(lookups, server):
    lookups(_c("unset"), recorded=_c("unset", NOW))
    assert await offer_after_turn(USER, _c("unset"), _state(server), once=False, now=NOW)


@pytest.mark.parametrize(
    ("consent", "state", "once"),
    [
        (_c("unset"), _state("events"), False),
        (_c("unset"), {}, False),
        (_c("unset"), _state("compute-resources"), True),
        (_c("unset", NOW - timedelta(days=3)), _state("compute-resources"), False),
        (_c("deny"), _state("compute-resources"), False),
        (_c("allow"), _state("compute-resources"), False),
        (None, _state("compute-resources"), False),
    ],
)
async def test_no_offer(lookups, consent, state, once):
    calls = lookups(_c("unset"), recorded=_c("unset", NOW))
    assert await offer_after_turn(USER, consent, state, once=once, now=NOW) is False
    assert calls["record"] == []


async def test_no_offer_when_the_ask_cant_be_recorded(lookups):
    lookups(_c("unset"), recorded=None)
    assert (
        await offer_after_turn(USER, _c("unset"), _state("compute-resources"), once=False, now=NOW)
        is False
    )


async def test_a_refused_call_doesnt_count(lookups):
    lookups(_c("unset"), recorded=_c("unset", NOW))
    state = {
        "tool_results": [
            {
                "step_id": "1",
                "tool_name": "t",
                "server": "compute-resources",
                "success": True,
                "refused": True,
            }
        ]
    }
    assert await offer_after_turn(USER, _c("unset"), state, once=False, now=NOW) is False


async def test_two_turns_ending_together_offer_once(monkeypatch):
    """Two tabs: the second re-reads under the lock and sees the first's ask."""
    import asyncio

    stored = {"consent": _c("unset")}
    records: list[str] = []

    async def consent_for(user):
        await asyncio.sleep(0)
        return stored["consent"]

    async def record_consent(user, choice):
        await asyncio.sleep(0.01)
        records.append(choice)
        stored["consent"] = _c("unset", NOW)
        return stored["consent"]

    monkeypatch.setattr(personalization, "consent_for", consent_for)
    monkeypatch.setattr(personalization, "record_consent", record_consent)
    monkeypatch.setattr(personalization, "_offer_locks", {})

    results = await asyncio.gather(
        *(
            offer_after_turn(USER, _c("unset"), _state("compute-resources"), once=False, now=NOW)
            for _ in range(2)
        )
    )
    assert sorted(results) == [False, True]
    assert records == ["not_now"]
