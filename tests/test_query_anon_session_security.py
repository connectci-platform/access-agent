"""Security tests for the /api/v1/query widget path.

Covers:
  - Finding I2: a server-minted anon session_id must be unguessable (was
    `sess_{millis}`, reconstructable from the request time). The session_id is
    the credential for the anon thread, so a low-entropy fallback let an
    attacker who knew the approximate request time reconstruct another user's
    thread id and read/continue their conversation.
  - Finding I4: the per-thread run lock is acquired synchronously in the handler
    and released in the streaming generator's finally. If the generator is never
    started, the lock must still be released so the thread is not 409-wedged.

Harness mirrors tests/test_pooled_checkpointer.py: stream_agent + registry are
mocked so no LLM/MCP infra is needed.
"""

import base64

import pytest
from httpx import ASGITransport, AsyncClient

FAKE_AGENT_RESULT = {
    "final_answer": "Test response",
    "tools_used": [],
    "query_analysis": None,
    "query_classification": None,
}


async def fake_stream_agent(**kwargs: object):
    # Echo the session_id the handler resolved so the test can inspect it.
    yield "custom", {"__resolved_session_id__": kwargs.get("session_id")}
    yield "updates", {"__end__": FAKE_AGENT_RESULT}


@pytest.fixture(autouse=True)
def _no_turnstile(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    # Pin the owner store at a temp sqlite file so this suite never touches a real
    # Postgres in a dev env that has one up. The widget claim_thread path reads
    # settings.DATABASE_URL; without this the tests pass only when DATABASE_URL is
    # unset (CI) and fail against the repo's default localhost DSN. Mirrors the
    # autouse pin in tests/test_thread_owners.py.
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as owners

    owners._store = None
    yield
    owners._store = None


@pytest.fixture(autouse=True)
def _reset_run_registry():
    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    tr._active_by_thread.clear()
    tr._task_by_run.clear()


@pytest.fixture
async def client():
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


async def _post_query(client, body):
    return await client.post("/api/v1/query", json=body)


async def test_anon_session_id_is_high_entropy(client, monkeypatch):
    """Omitting session_id yields a high-entropy `sess_<token_urlsafe>` id, not a
    guessable `sess_<timestamp>`."""
    from unittest.mock import AsyncMock, patch

    captured: dict[str, object] = {}

    async def _capture_stream(**kwargs: object):
        captured["session_id"] = kwargs.get("session_id")
        async for ev in fake_stream_agent(**kwargs):
            yield ev

    with (
        patch("src.api.routes.stream_agent", side_effect=_capture_stream),
        patch("src.api.routes.get_registry", new_callable=AsyncMock) as mock_reg,
    ):
        reg = AsyncMock()
        reg.catalog = {"tools": [], "quick_lookup": {}}
        mock_reg.return_value = reg

        r = await _post_query(client, {"query": "hi"})  # no session_id
        assert r.status_code == 200

    sid = captured["session_id"]
    assert isinstance(sid, str)
    assert sid.startswith("sess_")
    token = sid[len("sess_") :]
    # NOT a parseable timestamp (the old guessable form was all digits).
    assert not token.isdigit()
    # token_urlsafe(24) → 32 base64url chars → decodes to 24 bytes of entropy.
    assert len(token) >= 32
    assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    # Sanity: it is real base64url of >= 24 bytes.
    padded = token + "=" * (-len(token) % 4)
    assert len(base64.urlsafe_b64decode(padded)) >= 24


async def test_real_client_session_id_passes_through_untouched(client):
    """A real client's qa_bot_session_<uuid> id is used verbatim, not replaced."""
    from unittest.mock import AsyncMock, patch

    captured: dict[str, object] = {}

    async def _capture_stream(**kwargs: object):
        captured["session_id"] = kwargs.get("session_id")
        async for ev in fake_stream_agent(**kwargs):
            yield ev

    real_id = "qa_bot_session_123e4567-e89b-12d3-a456-426614174000"
    with (
        patch("src.api.routes.stream_agent", side_effect=_capture_stream),
        patch("src.api.routes.get_registry", new_callable=AsyncMock) as mock_reg,
    ):
        reg = AsyncMock()
        reg.catalog = {"tools": [], "quick_lookup": {}}
        mock_reg.return_value = reg

        r = await _post_query(client, {"query": "hi", "session_id": real_id})
        assert r.status_code == 200

    assert captured["session_id"] == real_id


async def test_lock_released_when_stream_generator_never_started(client):
    """If the StreamingResponse generator is never iterated, the per-thread lock
    must not stay held (Finding I4).

    Simulates the never-started path: the handler acquires the lock synchronously,
    but the generator's finally only runs once the generator is started. Here we
    force the widget handler's response construction path to raise AFTER acquire,
    which must release the lock. A subsequent acquire on the same thread must
    succeed (not 409-wedged).
    """
    from unittest.mock import AsyncMock, patch

    import src.api.thread_runs as tr

    sid = "wedge-test-thread"

    with (
        patch("src.api.routes.get_registry", new_callable=AsyncMock) as mock_reg,
        # Force the synchronous post-acquire path to blow up before hand-off, i.e.
        # before the streaming generator is ever started (its finally never runs).
        patch("src.api.routes.StreamingResponse", side_effect=RuntimeError("boom")),
    ):
        reg = AsyncMock()
        reg.catalog = {"tools": [], "quick_lookup": {}}
        mock_reg.return_value = reg

        # ASGITransport re-raises app exceptions rather than converting to a 500;
        # what matters is that the lock is not leaked on the way out.
        with pytest.raises(RuntimeError, match="boom"):
            await _post_query(client, {"query": "hi", "session_id": sid})

    # The lock the handler acquired must have been released on the failure path.
    assert sid not in tr._active_by_thread
    # And a fresh acquire on the same thread succeeds (not permanently 409-wedged).
    tr.acquire_thread_run(sid, "next-run")
    tr.release_thread_run(sid, "next-run")
