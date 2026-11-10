"""GET /threads/{id}/attachments/{attachment_id}: the same access as reading the thread (D8-3054).

Every refusal is the unknown-thread 404, so a reference copied from another
thread, or another caller's thread, tells the caller nothing.
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.thread_owners import get_thread_owner_store
from src.tool_attachments import externalize

KEY = "k" * 43
TABLE = {
    "version": 1,
    "columns": [{"name": "User", "kind": "label"}],
    "rows": [["a"], ["b"]],
    "row_count": 2,
}


@pytest.fixture(autouse=True)
def _sqlite_stores(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as owners
    import src.tool_attachments as attachments

    owners._store = None
    attachments._store = None


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _stored(thread_id: str) -> str:
    return externalize({"table": TABLE}, thread_id)["table_ref"]["id"]


def _url(thread_id: str, attachment_id: str) -> str:
    return f"/api/v1/threads/{thread_id}/attachments/{attachment_id}"


async def test_the_owner_gets_the_stored_payload(client, agent_tokens):
    get_thread_owner_store().claim_thread("t-a", "alice@x")
    attachment = _stored("t-a")

    r = await client.get(_url("t-a", attachment), headers=agent_tokens.headers("alice@x"))

    assert r.status_code == 200
    assert r.json() == TABLE


async def test_another_signed_in_caller_gets_404(client, agent_tokens):
    get_thread_owner_store().claim_thread("t-a", "alice@x")
    attachment = _stored("t-a")

    r = await client.get(_url("t-a", attachment), headers=agent_tokens.headers("bob@x"))

    assert r.status_code == 404


async def test_a_signed_out_caller_on_a_signed_in_thread_gets_404(client):
    get_thread_owner_store().claim_thread("t-a", "alice@x")

    r = await client.get(_url("t-a", _stored("t-a")))

    assert r.status_code == 404


async def test_an_anonymous_thread_needs_its_claim_key(client):
    get_thread_owner_store().claim_thread("t-anon", None, KEY)
    attachment = _stored("t-anon")

    ok = await client.get(_url("t-anon", attachment), headers={"X-Thread-Claim-Key": KEY})
    wrong = await client.get(_url("t-anon", attachment), headers={"X-Thread-Claim-Key": "x" * 43})
    none = await client.get(_url("t-anon", attachment))

    assert ok.status_code == 200
    assert json.loads(ok.text) == TABLE
    assert wrong.status_code == 404
    assert none.status_code == 404


async def test_a_reference_from_another_thread_gets_404(client, agent_tokens):
    """Alice owns both threads; an id from one isn't served under the other."""
    store = get_thread_owner_store()
    store.claim_thread("t-a", "alice@x")
    store.claim_thread("t-b", "alice@x")
    attachment = _stored("t-b")

    r = await client.get(_url("t-a", attachment), headers=agent_tokens.headers("alice@x"))

    assert r.status_code == 404
    assert r.json() == {"detail": "Thread not found"}


async def test_an_unknown_id_or_thread_gets_404(client, agent_tokens):
    get_thread_owner_store().claim_thread("t-a", "alice@x")
    headers = agent_tokens.headers("alice@x")

    unknown_id = await client.get(_url("t-a", "0" * 32), headers=headers)
    unknown_thread = await client.get(_url("t-none", "0" * 32), headers=headers)

    assert unknown_id.status_code == 404
    assert unknown_thread.status_code == 404
