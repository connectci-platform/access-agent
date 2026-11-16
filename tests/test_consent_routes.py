"""/api/v1/me/assistant-consent: the client's read and write of the setting (D8-3124)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.consent import Consent
from src.api import consent_routes
from src.main import app

URL = "/api/v1/me/assistant-consent"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def bearer(agent_tokens):
    return agent_tokens.headers("runner@x")


async def test_get_reads_the_callers_setting(client, bearer, monkeypatch):
    read = AsyncMock(return_value=Consent("unset", datetime(2026, 11, 1, tzinfo=UTC)))
    monkeypatch.setattr(consent_routes, "consent_for", read)

    r = await client.get(URL, headers=bearer)

    assert r.status_code == 200
    assert r.json() == {"consent": "unset", "asked_at": "2026-11-01T00:00:00+00:00"}
    assert r.headers["cache-control"] == "no-store"
    read.assert_awaited_once_with("runner@x")


@pytest.mark.parametrize("choice", ["allow", "deny", "not_now"])
async def test_post_records_the_callers_choice(client, bearer, monkeypatch, choice):
    state = "unset" if choice == "not_now" else choice
    write = AsyncMock(return_value=Consent(state, None))
    monkeypatch.setattr(consent_routes, "record_consent", write)

    r = await client.post(URL, headers=bearer, json={"consent": choice})

    assert r.status_code == 200
    assert r.json() == {"consent": state, "asked_at": None}
    write.assert_awaited_once_with("runner@x", choice)


@pytest.mark.parametrize("body", [{"consent": "allow_once"}, {"consent": "yes"}, {}, {"x": 1}])
async def test_post_rejects_anything_else(client, bearer, monkeypatch, body):
    write = AsyncMock()
    monkeypatch.setattr(consent_routes, "record_consent", write)

    r = await client.post(URL, headers=bearer, json=body)

    assert r.status_code == 422
    write.assert_not_awaited()


@pytest.mark.parametrize("method", ["get", "post"])
async def test_anonymous_is_401_with_no_lookup(client, monkeypatch, method):
    read, write = AsyncMock(), AsyncMock()
    monkeypatch.setattr(consent_routes, "consent_for", read)
    monkeypatch.setattr(consent_routes, "record_consent", write)

    r = await getattr(client, method)(
        URL, **({"json": {"consent": "allow"}} if method == "post" else {})
    )

    assert r.status_code == 401
    read.assert_not_awaited()
    write.assert_not_awaited()


async def test_unreachable_setting_is_503(client, bearer, monkeypatch):
    monkeypatch.setattr(consent_routes, "consent_for", AsyncMock(return_value=None))
    monkeypatch.setattr(consent_routes, "record_consent", AsyncMock(return_value=None))

    assert (await client.get(URL, headers=bearer)).status_code == 503
    r = await client.post(URL, headers=bearer, json={"consent": "deny"})
    assert r.status_code == 503
    assert "couldn't be reached" in r.json()["error"]
