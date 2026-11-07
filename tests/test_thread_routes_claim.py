"""Tests for POST /threads/{id}/claim and the key rule on the upgrade paths.

A fullscreen thread id travels in ``?threadId=``, so knowing it proves nothing.
Only the claim key the browser sent on the thread's first anonymous run does.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.identity import CLAIM_KEY_HEADER
from src.main import app
from src.thread_owners import _hash_user, get_thread_owner_store
from src.turn_reporter import TurnReportBase, TurnReporter

KEY = "k" * 43
OTHER_KEY = "o" * 43
RUN_BODY = {"input": {"messages": [{"role": "user", "content": "what is ACCESS?"}]}}


@pytest.fixture(autouse=True)
def _sqlite_owner_store(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.thread_owners as m

    m._store = None


@pytest.fixture
def reporter(monkeypatch, tmp_path):
    r = TurnReporter()
    r._engine = create_engine(f"sqlite:///{tmp_path / 'reports.db'}")
    TurnReportBase.metadata.create_all(r._engine)
    r._session_factory = sessionmaker(bind=r._engine)
    r._initialized = True
    monkeypatch.setattr("src.api.thread_routes.get_turn_reporter", lambda: r)
    monkeypatch.setattr("src.turn_reporter.get_turn_reporter", lambda: r)
    return r


@pytest.fixture(autouse=True)
def _mock_stream_agent(monkeypatch, reporter):
    async def _fake(**_kwargs):
        yield "messages", (AIMessageChunk(content="hi", id="m1"), {"langgraph_node": "model"})
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


@pytest.fixture
def bearer_for(agent_tokens):
    return agent_tokens.headers


async def _anon_run(client, thread_id: str, key: str | None = KEY) -> None:
    headers = {CLAIM_KEY_HEADER: key} if key else {}
    r = await client.post(
        f"/api/v1/threads/{thread_id}/runs/stream", headers=headers, json=RUN_BODY
    )
    assert r.status_code == 200


async def _claim(client, thread_id: str, bearer: dict[str, str], key: str | None = KEY):
    headers = {**bearer, **({CLAIM_KEY_HEADER: key} if key else {})}
    return await client.post(f"/api/v1/threads/{thread_id}/claim", headers=headers)


async def _sidebar(client, bearer: dict[str, str]) -> list[str]:
    r = await client.post("/api/v1/threads/search", headers=bearer, json={"limit": 10})
    assert r.status_code == 200
    return [t["thread_id"] for t in r.json()]


async def test_claim_with_key_lists_the_conversation_without_another_message(client, bearer_for):
    await _anon_run(client, "t-demo")
    bearer = bearer_for("me@x")
    assert await _sidebar(client, bearer) == []

    r = await _claim(client, "t-demo", bearer)

    assert r.status_code == 200
    assert r.json() == {"thread_id": "t-demo"}
    assert await _sidebar(client, bearer) == ["t-demo"]
    owner = get_thread_owner_store().resolve_owner("t-demo")
    assert owner is not None
    assert owner.was_authenticated is True
    assert owner.user_hash == _hash_user("me@x")


async def test_claim_requires_a_signed_in_caller(client):
    await _anon_run(client, "t-anon")

    r = await client.post("/api/v1/threads/t-anon/claim", headers={CLAIM_KEY_HEADER: KEY})

    assert r.status_code == 401


@pytest.mark.parametrize("key", [None, OTHER_KEY, "short"])
async def test_knowing_the_thread_id_is_not_enough(client, bearer_for, key):
    """Someone with the shared link gets the same 404 as for an unknown thread,
    the thread stays anonymous, and the rightful owner can still claim it."""
    await _anon_run(client, "t-shared")
    thief = bearer_for("thief@x")

    r = await _claim(client, "t-shared", thief, key)
    unknown = await _claim(client, "t-unknown", thief, key)

    assert r.status_code == 404
    assert (r.status_code, r.json()) == (unknown.status_code, unknown.json())
    assert await _sidebar(client, thief) == []
    owner = get_thread_owner_store().resolve_owner("t-shared")
    assert owner is not None
    assert owner.was_authenticated is False
    assert (await _claim(client, "t-shared", bearer_for("me@x"))).status_code == 200


async def test_a_later_anonymous_visitor_cannot_replace_the_key(client, bearer_for):
    await _anon_run(client, "t-keyed")
    await _anon_run(client, "t-keyed", OTHER_KEY)

    assert (await _claim(client, "t-keyed", bearer_for("thief@x"), OTHER_KEY)).status_code == 404
    assert (await _claim(client, "t-keyed", bearer_for("me@x"))).status_code == 200


async def test_a_thread_without_a_stored_key_cannot_be_claimed(client, bearer_for):
    """Rows from before this route, and widget rows, have no proof to check."""
    await _anon_run(client, "t-legacy", key=None)

    assert (await _claim(client, "t-legacy", bearer_for("me@x"), KEY)).status_code == 404


async def test_a_claimed_thread_cannot_be_claimed_by_anyone_else(client, bearer_for):
    await _anon_run(client, "t-taken")
    assert (await _claim(client, "t-taken", bearer_for("me@x"))).status_code == 200

    r = await _claim(client, "t-taken", bearer_for("thief@x"))

    assert r.status_code == 404
    assert await _sidebar(client, bearer_for("thief@x")) == []
    history = await client.post(
        "/api/v1/threads/t-taken/history", headers=bearer_for("thief@x"), json={}
    )
    assert history.status_code == 404


async def test_claiming_again_as_the_owner_succeeds(client, bearer_for):
    await _anon_run(client, "t-again")
    bearer = bearer_for("me@x")
    assert (await _claim(client, "t-again", bearer)).status_code == 200

    r = await _claim(client, "t-again", bearer, key=None)

    assert r.status_code == 200
    assert await _sidebar(client, bearer) == ["t-again"]


async def test_an_authed_owned_thread_is_not_claimable_with_any_key(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "alice@x", KEY)

    assert (await _claim(client, "t-owned", bearer_for("thief@x"))).status_code == 404


async def test_run_with_the_key_upgrades_the_thread(client, bearer_for):
    await _anon_run(client, "t-run")
    bearer = bearer_for("me@x")

    r = await client.post(
        "/api/v1/threads/t-run/runs/stream",
        headers={**bearer, CLAIM_KEY_HEADER: KEY},
        json=RUN_BODY,
    )

    assert r.status_code == 200
    owner = get_thread_owner_store().resolve_owner("t-run")
    assert owner is not None
    assert owner.user_hash == _hash_user("me@x")


@pytest.mark.parametrize("key", [None, OTHER_KEY])
async def test_run_without_the_key_cannot_take_a_keyed_thread(client, bearer_for, key):
    await _anon_run(client, "t-run-thief")
    thief = bearer_for("thief@x")
    headers = {**thief, **({CLAIM_KEY_HEADER: key} if key else {})}

    r = await client.post("/api/v1/threads/t-run-thief/runs/stream", headers=headers, json=RUN_BODY)

    assert r.status_code == 404
    owner = get_thread_owner_store().resolve_owner("t-run-thief")
    assert owner is not None
    assert owner.was_authenticated is False
    assert await _sidebar(client, thief) == []


async def test_query_cannot_take_a_keyed_thread(client, bearer_for):
    """/query shares the id space, so it must not be a way around the key."""
    await _anon_run(client, "t-query-thief")

    r = await client.post(
        "/api/v1/query",
        headers=bearer_for("thief@x"),
        json={"query": "hello", "session_id": "t-query-thief"},
    )

    assert r.status_code == 404
    owner = get_thread_owner_store().resolve_owner("t-query-thief")
    assert owner is not None
    assert owner.was_authenticated is False
