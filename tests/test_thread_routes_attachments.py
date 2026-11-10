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
    return externalize({"table": TABLE}, thread_id)[0]["table_ref"]["id"]


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


async def test_a_caller_that_takes_gzip_gets_it_compressed(client, agent_tokens):
    get_thread_owner_store().claim_thread("t-a", "alice@x")
    attachment = _stored("t-a")
    headers = agent_tokens.headers("alice@x")

    zipped = await client.get(
        _url("t-a", attachment), headers={**headers, "Accept-Encoding": "gzip"}
    )
    plain = await client.get(
        _url("t-a", attachment), headers={**headers, "Accept-Encoding": "identity"}
    )

    assert zipped.headers["content-encoding"] == "gzip"
    assert zipped.json() == TABLE  # httpx decodes it
    assert "content-encoding" not in plain.headers
    assert plain.json() == TABLE
    assert plain.headers["cache-control"] == "private, no-store"


class _Graph:
    def __init__(self, messages):
        self.messages = messages

    async def aget_state(self, config):
        from types import SimpleNamespace

        return SimpleNamespace(values={"messages": self.messages})


def _request():
    from types import SimpleNamespace

    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=object())))


def _ref_message(attachment_id: str):
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content="summary",
        tool_call_id="c",
        name="get_raw_data",
        artifact={"structured_content": {"table_ref": {"id": attachment_id, "kind": "table"}}},
    )


async def test_after_a_run_attachments_no_message_refers_to_are_deleted(
    client, agent_tokens, monkeypatch
):
    """A retried, edited or compacted turn's table goes; the kept turn's stays."""
    from src.tool_attachments import get_tool_attachment_store

    get_thread_owner_store().claim_thread("t-a", "alice@x")
    kept, replaced = _stored("t-a"), _stored("t-a")

    async def _fake_stream(**_kwargs):
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake_stream)
    monkeypatch.setattr(
        "src.api.thread_routes.create_checkpointed_graph",
        lambda cp: _Graph([_ref_message(kept)]),
    )
    from langgraph.checkpoint.memory import InMemorySaver

    app.state.checkpointer = InMemorySaver()
    try:
        r = await client.post(
            "/api/v1/threads/t-a/runs/stream",
            headers=agent_tokens.headers("alice@x"),
            json={"input": {"messages": [{"role": "user", "content": "again"}]}},
        )
    finally:
        app.state.checkpointer = None

    assert r.status_code == 200
    assert get_tool_attachment_store().ids_for("t-a") == {kept}
    assert replaced != kept


async def test_the_sweep_waits_out_a_stopped_turn_still_saving(monkeypatch):
    from src.api import thread_routes
    from src.tool_attachments import get_tool_attachment_store

    get_thread_owner_store().claim_thread("t-a", "alice@x")
    attachment = _stored("t-a")
    monkeypatch.setattr(thread_routes, "stopped_turn_saving", lambda t: True)
    monkeypatch.setattr(thread_routes, "create_checkpointed_graph", lambda cp: _Graph([]))

    await thread_routes._sweep_attachments("t-a", object())  # type: ignore[arg-type]

    assert get_tool_attachment_store().ids_for("t-a") == {attachment}


async def test_the_sweep_never_raises(monkeypatch, caplog):
    from src.api import thread_routes

    get_thread_owner_store().claim_thread("t-a", "alice@x")
    _stored("t-a")

    def broken(cp):
        raise RuntimeError("checkpointer gone")

    monkeypatch.setattr(thread_routes, "create_checkpointed_graph", broken)

    await thread_routes._sweep_attachments("t-a", object())  # type: ignore[arg-type]
    assert "Couldn't sweep attachments" in caplog.text


async def test_a_thread_with_nothing_stored_reads_no_state(monkeypatch):
    from src.api import thread_routes

    def unexpected(cp):
        raise AssertionError("read the thread for nothing")

    monkeypatch.setattr(thread_routes, "create_checkpointed_graph", unexpected)

    await thread_routes._sweep_attachments("t-empty", object())  # type: ignore[arg-type]


async def test_a_table_this_run_stores_survives_the_cleanup(client, agent_tokens, monkeypatch):
    """Cleanup runs at the start of a run, under its lock, before anything new
    is stored: a table stored mid-run, before its message is saved, stays."""
    from src.tool_attachments import get_tool_attachment_store

    get_thread_owner_store().claim_thread("t-a", "alice@x")
    stale = _stored("t-a")
    fresh: list[str] = []

    async def _fake_stream(**_kwargs):
        fresh.append(_stored("t-a"))
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake_stream)
    # The saved thread refers to neither: the stale one goes, the fresh one stays.
    monkeypatch.setattr("src.api.thread_routes.create_checkpointed_graph", lambda cp: _Graph([]))
    from langgraph.checkpoint.memory import InMemorySaver

    app.state.checkpointer = InMemorySaver()
    try:
        r = await client.post(
            "/api/v1/threads/t-a/runs/stream",
            headers=agent_tokens.headers("alice@x"),
            json={"input": {"messages": [{"role": "user", "content": "q"}]}},
        )
    finally:
        app.state.checkpointer = None

    assert r.status_code == 200
    assert get_tool_attachment_store().ids_for("t-a") == {fresh[0]}
    assert stale not in fresh


async def test_a_database_error_reads_as_not_found(client, agent_tokens, monkeypatch):
    from src.tool_attachments import ToolAttachmentStore

    get_thread_owner_store().claim_thread("t-a", "alice@x")
    attachment = _stored("t-a")

    def down(self):
        raise RuntimeError("database is down")

    monkeypatch.setattr(ToolAttachmentStore, "_sessions", down)
    owner = await client.get(_url("t-a", attachment), headers=agent_tokens.headers("alice@x"))
    stranger = await client.get(_url("t-a", attachment), headers=agent_tokens.headers("bob@x"))

    assert owner.status_code == 404
    assert stranger.status_code == 404
    assert owner.json() == stranger.json() == {"detail": "Thread not found"}
