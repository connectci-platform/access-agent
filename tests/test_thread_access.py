"""The access rule, row by row: a thread id alone grants nothing.

Reading, writing or claiming a conversation needs its owner, or the key from
the browser that started it. Every refusal is the same 404 as an unknown thread.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk

from src.api.identity import CLAIM_KEY_HEADER
from src.main import app
from src.thread_owners import ThreadAccess, _hash_user, get_thread_owner_store

KEY = "k" * 43
WRONG_KEY = "o" * 43
OWNER = "owner@x"
OTHER = "other@x"
NOT_FOUND = {"detail": "Thread not found"}
RUN_BODY = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
WIDGET_ID = f"qa_bot_session_{uuid.uuid4()}"
FULLSCREEN_ID = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def _sqlite_owner_store(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.thread_owners as m

    m._store = None


@pytest.fixture(autouse=True)
def _mock_agents(monkeypatch):
    async def _run(**_kwargs):
        yield "messages", (AIMessageChunk(content="hi", id="m1"), {"langgraph_node": "model"})
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    async def _query(**_kwargs):
        yield "updates", {"__end__": {"final_answer": "hi", "tools_used": []}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _run)
    registry = AsyncMock()
    registry.catalog = {"tools": [], "quick_lookup": {}}
    with (
        patch("src.api.routes.stream_agent", side_effect=_query),
        patch("src.api.routes.get_registry", return_value=registry),
    ):
        yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


@pytest.fixture
def headers(agent_tokens):
    def _headers(user: str | None, key: str | None) -> dict[str, str]:
        bearer = agent_tokens.headers(user) if user else {}
        return {**bearer, **({CLAIM_KEY_HEADER: key} if key else {})}

    return _headers


def _seed(kind: str) -> str:
    store = get_thread_owner_store()
    thread_id = FULLSCREEN_ID
    if kind == "keyed":
        store.claim_thread(thread_id, None, KEY)
    elif kind == "owned":
        store.claim_thread(thread_id, OWNER)
    elif kind == "keyless":
        store.claim_thread(thread_id, None)
    return thread_id


async def _read(client, thread_id, headers):
    return await client.post(f"/api/v1/threads/{thread_id}/history", headers=headers, json={})


async def _cancel(client, thread_id, headers):
    return await client.post(f"/api/v1/threads/{thread_id}/runs/r1/cancel", headers=headers)


async def _run(client, thread_id, headers):
    return await client.post(
        f"/api/v1/threads/{thread_id}/runs/stream", headers=headers, json=RUN_BODY
    )


async def _query(client, thread_id, headers):
    return await client.post(
        "/api/v1/query", headers=headers, json={"query": "hi", "session_id": thread_id}
    )


async def _claim(client, thread_id, headers):
    return await client.post(f"/api/v1/threads/{thread_id}/claim", headers=headers)


# (thread kind, caller, key) for each refused row. Cancel's "allowed" answer is
# a 404 too (no run in flight), so these rows all pin the response body.
REFUSED = [
    pytest.param("keyed", None, None, id="keyed-anon-no-key"),
    pytest.param("keyed", None, WRONG_KEY, id="keyed-anon-wrong-key"),
    pytest.param("keyed", OTHER, None, id="keyed-signed-in-no-key"),
    pytest.param("keyed", OTHER, WRONG_KEY, id="keyed-signed-in-wrong-key"),
    pytest.param("owned", None, None, id="owned-anon"),
    pytest.param("owned", None, KEY, id="owned-anon-any-key"),
    pytest.param("owned", OTHER, None, id="owned-other-user"),
    pytest.param("owned", OTHER, KEY, id="owned-other-user-any-key"),
    pytest.param("keyless", None, None, id="keyless-anon"),
    pytest.param("keyless", None, KEY, id="keyless-anon-any-key"),
    pytest.param("keyless", OTHER, None, id="keyless-signed-in"),
    pytest.param("keyless", OTHER, KEY, id="keyless-signed-in-any-key"),
]
ROUTES = [
    pytest.param(_read, id="read"),
    pytest.param(_cancel, id="cancel"),
    pytest.param(_run, id="run"),
    pytest.param(_query, id="query"),
]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(("kind", "user", "key"), REFUSED)
async def test_refused_callers_get_the_unknown_thread_404(client, headers, route, kind, user, key):
    thread_id = _seed(kind)
    before = get_thread_owner_store().resolve_owner(thread_id)

    r = await route(client, thread_id, headers(user, key))

    assert r.status_code == 404
    assert r.json() == NOT_FOUND
    assert get_thread_owner_store().resolve_owner(thread_id) == before


@pytest.mark.parametrize(("kind", "user", "key"), [p for p in REFUSED if p.values[1]])
async def test_refused_claims_get_the_unknown_thread_404(client, headers, kind, user, key):
    thread_id = _seed(kind)
    before = get_thread_owner_store().resolve_owner(thread_id)

    r = await _claim(client, thread_id, headers(user, key))

    assert r.status_code == 404
    assert r.json() == NOT_FOUND
    assert get_thread_owner_store().resolve_owner(thread_id) == before


@pytest.mark.parametrize(("kind", "user", "key"), [p for p in REFUSED if p.values[1]])
async def test_refused_handoff_mints_get_the_unknown_thread_404(client, headers, kind, user, key):
    thread_id = _seed(kind)

    r = await client.post(f"/api/v1/threads/{thread_id}/handoff", headers=headers(user, key))

    assert r.status_code == 404
    assert r.json() == NOT_FOUND


async def test_a_key_holder_cannot_mint_a_handoff_before_claiming(client, headers):
    """A handoff binds to an owner, and an unclaimed thread has none."""
    thread_id = _seed("keyed")

    r = await client.post(f"/api/v1/threads/{thread_id}/handoff", headers=headers(OWNER, KEY))

    assert r.status_code == 404
    assert r.json() == NOT_FOUND


@pytest.mark.parametrize("route", [_read, _cancel, _claim], ids=["read", "cancel", "claim"])
async def test_an_unknown_thread_is_a_404(client, headers, route):
    r = await route(client, "t-unknown", headers(OWNER, KEY))

    assert r.status_code == 404


@pytest.mark.parametrize(
    ("route", "status"),
    [(_read, 200), (_run, 200), (_query, 200), (_claim, 200)],
    ids=["read", "run", "query", "claim"],
)
async def test_the_owner_keeps_access(client, headers, route, status):
    thread_id = _seed("owned")

    assert (await route(client, thread_id, headers(OWNER, None))).status_code == status


async def test_the_owner_can_cancel(client, headers):
    thread_id = _seed("owned")

    r = await _cancel(client, thread_id, headers(OWNER, None))

    assert r.json() == {"detail": "Run not found"}  # past the gate, no run in flight


@pytest.mark.parametrize("route", [_read, _run, _query], ids=["read", "run", "query"])
async def test_the_anonymous_key_holder_keeps_access(client, headers, route):
    thread_id = _seed("keyed")

    assert (await route(client, thread_id, headers(None, KEY))).status_code == 200
    owner = get_thread_owner_store().resolve_owner(thread_id)
    assert owner is not None
    assert owner.was_authenticated is False


async def test_the_anonymous_key_holder_can_cancel(client, headers):
    thread_id = _seed("keyed")

    r = await _cancel(client, thread_id, headers(None, KEY))

    assert r.json() == {"detail": "Run not found"}


async def test_an_anonymous_claim_is_a_401(client, headers):
    thread_id = _seed("keyed")

    assert (await _claim(client, thread_id, headers(None, KEY))).status_code == 401


async def test_a_signed_in_reader_with_the_key_does_not_take_ownership(client, headers):
    thread_id = _seed("keyed")

    assert (await _read(client, thread_id, headers(OWNER, KEY))).status_code == 200
    owner = get_thread_owner_store().resolve_owner(thread_id)
    assert owner is not None
    assert owner.was_authenticated is False


@pytest.mark.parametrize("route", [_run, _query, _claim], ids=["run", "query", "claim"])
async def test_a_signed_in_writer_with_the_key_takes_ownership(client, headers, route):
    thread_id = _seed("keyed")

    assert (await route(client, thread_id, headers(OWNER, KEY))).status_code == 200
    owner = get_thread_owner_store().resolve_owner(thread_id)
    assert owner is not None
    assert owner.user_hash == _hash_user(OWNER)
    assert (await _read(client, thread_id, headers(None, KEY))).status_code == 404


async def test_a_handoff_thread_keeps_working_for_its_owner(client, headers):
    """A widget session flips on /query, mints a handoff, and resumes fullscreen."""
    assert (await _query(client, WIDGET_ID, headers(None, None))).status_code == 200
    assert (await _query(client, WIDGET_ID, headers(OWNER, None))).status_code == 200

    minted = await client.post(f"/api/v1/threads/{WIDGET_ID}/handoff", headers=headers(OWNER, None))
    exchanged = await client.post(
        "/api/v1/handoff/exchange",
        headers=headers(OWNER, None),
        json={"token": minted.json()["token"]},
    )

    assert exchanged.json() == {"thread_id": WIDGET_ID}
    assert (await _read(client, WIDGET_ID, headers(OWNER, None))).status_code == 200
    assert (await _run(client, WIDGET_ID, headers(OWNER, None))).status_code == 200


# ---- /query's exception: callers that aren't fullscreen threads -------------


async def test_the_widget_continues_its_own_session_anonymously(client, headers):
    assert (await _query(client, WIDGET_ID, headers(None, None))).status_code == 200
    assert (await _query(client, WIDGET_ID, headers(None, None))).status_code == 200


async def test_signing_in_on_the_widget_flips_its_session(client, headers):
    await _query(client, WIDGET_ID, headers(None, None))

    assert (await _query(client, WIDGET_ID, headers(OWNER, None))).status_code == 200

    owner = get_thread_owner_store().resolve_owner(WIDGET_ID)
    assert owner is not None
    assert owner.user_hash == _hash_user(OWNER)


@pytest.mark.parametrize("route", [_read, _run, _claim], ids=["read", "run", "claim"])
async def test_a_widget_session_is_only_open_on_query(client, headers, route):
    await _query(client, WIDGET_ID, headers(None, None))

    assert (await route(client, WIDGET_ID, headers(OWNER, None))).status_code == 404


@pytest.mark.parametrize(
    "session_id",
    ["wave3-persona-01-1a2b3c4d", "redteam-suite-v1__p1__1a2b3c4d", "sess_abc", "redteam-default"],
)
async def test_non_browser_callers_keep_multi_turn_sessions(client, headers, session_id):
    assert (await _query(client, session_id, headers(None, None))).status_code == 200
    assert (await _query(client, session_id, headers(None, None))).status_code == 200


@pytest.mark.parametrize(
    "session_id",
    [
        "wave3-persona-01-1a2b3c4d",
        f"qa_bot_session_{uuid.uuid4()}-extra",
        "qa_bot_session_not-a-uuid",
        f"x{WIDGET_ID}",
    ],
)
async def test_only_an_exact_widget_id_flips(client, headers, session_id):
    await _query(client, session_id, headers(None, None))

    assert (await _query(client, session_id, headers(OTHER, None))).status_code == 404
    owner = get_thread_owner_store().resolve_owner(session_id)
    assert owner is not None
    assert owner.was_authenticated is False


async def test_an_older_fullscreen_thread_stays_locked_with_uppercase_id(client, headers):
    thread_id = str(uuid.uuid4()).upper()
    get_thread_owner_store().claim_thread(thread_id, None)

    assert (await _query(client, thread_id, headers(None, None))).status_code == 404


# ---- losing the first-writer race ------------------------------------------


def _stale_then_real():
    """The first read saw no thread; another writer's row landed right after."""
    store = get_thread_owner_store()
    real = store.access
    answers = iter([ThreadAccess.NEW])
    return patch.object(
        store,
        "access",
        side_effect=lambda *args: next(answers, None) or real(*args),
    )


@pytest.mark.parametrize("route", [_run, _query], ids=["run", "query"])
async def test_losing_the_first_write_to_someone_else_is_a_404(client, headers, route):
    thread_id = _seed("owned")

    with _stale_then_real():
        r = await route(client, thread_id, headers(OTHER, None))

    assert r.status_code == 404
    assert r.json() == NOT_FOUND


async def test_losing_the_first_write_to_an_anonymous_browser_is_a_404(client, headers):
    thread_id = _seed("keyed")

    with _stale_then_real():
        r = await _run(client, thread_id, headers(None, WRONG_KEY))

    assert r.status_code == 404


async def test_losing_the_first_write_to_your_own_other_request_is_fine(client, headers):
    thread_id = _seed("keyed")

    with _stale_then_real():
        r = await _run(client, thread_id, headers(None, KEY))

    assert r.status_code == 200


async def test_the_widget_racing_itself_on_a_new_session_is_fine(client, headers):
    get_thread_owner_store().claim_thread(WIDGET_ID, None)

    with _stale_then_real():
        r = await _query(client, WIDGET_ID, headers(None, None))

    assert r.status_code == 200


@pytest.mark.parametrize("route", [_run, _query], ids=["run", "query"])
async def test_a_new_thread_works_without_a_database(client, headers, monkeypatch, route):
    """No owner store means nothing to own, so the failed claim isn't a lost race."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    import src.thread_owners as m

    m._store = None

    assert (await route(client, "t-no-db", headers(None, KEY))).status_code == 200


async def test_a_claim_never_creates_a_thread(client, headers):
    assert (await _claim(client, "t-new", headers(OWNER, KEY))).status_code == 404
    assert get_thread_owner_store().resolve_owner("t-new") is None
