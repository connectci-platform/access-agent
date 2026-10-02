"""Tests for POST /threads/{id}/runs/stream (the gated, bounded streaming run)
and the widget-path ownership/lock sharing.

Harness copied from tests/test_thread_routes_history.py (JWKS server + ES256 JWT
cookie minting). stream_agent is mocked so no LLM/MCP is needed. The thread-owner
store is pointed at a temp sqlite file.
"""

import asyncio
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from httpx import ASGITransport, AsyncClient
from jwt import algorithms as jwt_algorithms
from langchain_core.messages import AIMessageChunk

from src.auth import configure_trusted_issuers
from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Test key pair and JWKS server (copied from tests/test_thread_routes_history.py)
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "thread-run-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "thread-run-test-kid"


class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(JWKS_RESPONSE)

    def log_message(self, format, *args):
        pass


def _start_jwks_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def _make_jwt(sub: str, issuer: str = ISSUER) -> str:
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": sub,
        "iat": now - 3600,
        "exp": now + 3600,
    }
    return jwt.encode(payload, PRIVATE_PEM, algorithm="ES256", headers={"kid": KID})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _jwks_server():
    server, jwks_url = _start_jwks_server()
    configure_trusted_issuers({ISSUER: jwks_url})
    yield
    server.shutdown()
    configure_trusted_issuers({})


@pytest.fixture(autouse=True)
def _sqlite_owner_store(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as m

    m._store = None  # reset singleton


@pytest.fixture(autouse=True)
def _sqlite_turn_reporter(monkeypatch, tmp_path):
    """Point get_turn_reporter()'s module-level singleton at the SAME temp
    sqlite file _sqlite_owner_store uses (load-bearing: get_turn_reporter()
    caches its engine on first touch, and this is the first run-test suite to
    touch it — without a per-test reset, a stale engine from another test, or
    "" from CI env, leaks in and the "exactly one row" assertions go
    order-dependent/flaky)."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.turn_reporter as reporter_mod

    reporter_mod._turn_reporter = None
    yield
    reporter_mod._turn_reporter = None


@pytest.fixture(autouse=True)
def _reset_run_registry():
    """Clear the in-process per-thread run registry between tests."""
    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    tr._active_by_thread.clear()
    tr._task_by_run.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


@pytest.fixture
def valid_cookie_for():
    def _make(sub: str) -> dict[str, str]:
        return {"SESSaccess_auth": _make_jwt(sub)}

    return _make


@pytest.fixture
def valid_cookie(valid_cookie_for):
    return valid_cookie_for("runner@x")


@pytest.fixture
def seed_owner():
    def _seed(thread_id: str, user: str) -> None:
        get_thread_owner_store().claim_thread(thread_id, user)

    return _seed


def _mock_stream_agent(monkeypatch, *, content: str = "hi", msg_id: str = "m1"):
    """Patch thread_routes.stream_agent to yield a canned messages chunk + updates."""

    async def _fake(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=content, id=msg_id),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        yield "updates", {"tool_calling_loop": {"final_answer": content}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)


# ---------------------------------------------------------------------------
# POST /threads/{id}/runs/stream
# ---------------------------------------------------------------------------


async def test_run_stream_emits_protocol_events(client, valid_cookie, monkeypatch):
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-new/runs/stream",
        cookies=valid_cookie,
        json={
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 200
    body = r.text
    assert "event: metadata" in body
    assert "event: messages/partial" in body
    assert "event: messages/complete" in body
    assert "event: values" in body
    assert "event: end" not in body  # protocol has no end terminator


async def test_run_anon_caller_on_new_thread_200(client, monkeypatch):
    """Anon caller (no cookie), new thread → 200, not 401. claim_thread lazily
    creates an anon-owned thread for a None caller."""
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-anon-new/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200


async def test_run_anon_caller_on_authed_owned_thread_404(client, valid_cookie_for, seed_owner):
    """Anon (no cookie) caller hitting an existing AUTHED-owned thread is a
    non-owner: 404, not 401/403."""
    seed_owner("t-owned-authed", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned-authed/runs/stream",
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 404


async def test_run_authed_caller_still_requires_nothing_new(client, valid_cookie, monkeypatch):
    """Regression: an authed caller on a brand-new thread still gets 200."""
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-authed-new/runs/stream",
        cookies=valid_cookie,
        json={
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 200


async def test_run_stream_releases_lock_if_generator_never_starts(
    client, valid_cookie, monkeypatch
):
    """If the streaming generator is never started, the per-thread lock must not
    leak (Finding I4).

    The lock is acquired synchronously in the handler and released in the
    generator's finally, but an async generator that is never iterated never runs
    its finally. Force the handler's synchronous post-acquire path to raise
    (before StreamingResponse is handed back) and assert the thread is not left
    409-wedged: a subsequent run on the same thread must succeed, not 409.
    """
    import src.api.thread_runs as tr

    with (
        patch("src.api.thread_routes.StreamingResponse", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await client.post(
            "/api/v1/threads/t-wedge/runs/stream",
            cookies=valid_cookie,
            json={"input": {"messages": [{"role": "user", "content": "x"}]}},
        )

    assert "t-wedge" not in tr._active_by_thread  # lock released on the failure path

    # And a real run on the same thread now succeeds (not permanently wedged).
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-wedge/runs/stream",
        cookies=valid_cookie,
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert r.status_code == 200


async def test_run_on_another_users_thread_404(client, valid_cookie_for, seed_owner, monkeypatch):
    _mock_stream_agent(monkeypatch)
    seed_owner("t-owned", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned/runs/stream",
        cookies=valid_cookie_for("bob@x"),
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 404


async def test_resource_context_from_config_forwarded(client, valid_cookie, monkeypatch):
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        if False:
            yield  # make it an async generator

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)
    await client.post(
        "/api/v1/threads/t/runs/stream",
        cookies=valid_cookie,
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
            "config": {"configurable": {"resource_context": "delta"}},
        },
    )
    assert captured.get("resource_context") == "delta"


async def test_concurrent_run_rejected(client, valid_cookie, monkeypatch):
    release = asyncio.Event()

    async def _blocking_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        await release.wait()
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _blocking_stream)

    body = {
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "if_not_exists": "create",
    }

    async def _fire():
        return await client.post(
            "/api/v1/threads/t-busy/runs/stream", cookies=valid_cookie, json=body
        )

    first_task = asyncio.create_task(_fire())
    # Give the first run time to acquire the lock before firing the second.
    for _ in range(50):
        await asyncio.sleep(0.01)
        import src.api.thread_runs as tr

        if "t-busy" in tr._active_by_thread:
            break

    second = await _fire()
    assert second.status_code == 409

    release.set()
    await first_task


# ---------------------------------------------------------------------------
# Step 6: widget path — ownership + shared lock
# ---------------------------------------------------------------------------


async def test_widget_and_fullscreen_share_lock(client, valid_cookie, monkeypatch):
    """A widget /query and a /runs/stream on the same session_id can't run at once."""
    release = asyncio.Event()

    async def _blocking_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        await release.wait()
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    async def _fake_registry():
        from types import SimpleNamespace

        return SimpleNamespace(catalog={"servers": []})

    monkeypatch.setattr("src.api.thread_runs._active_by_thread", {}, raising=False)
    monkeypatch.setattr("src.api.routes.stream_agent", _blocking_stream)
    monkeypatch.setattr("src.api.routes.get_registry", _fake_registry)
    monkeypatch.setattr("src.api.thread_routes.stream_agent", _blocking_stream)

    async def _fire_widget():
        return await client.post(
            "/api/v1/query",
            cookies=valid_cookie,
            json={"query": "hi", "session_id": "shared-sess"},
        )

    widget_task = asyncio.create_task(_fire_widget())
    for _ in range(50):
        await asyncio.sleep(0.01)
        import src.api.thread_runs as tr

        if "shared-sess" in tr._active_by_thread:
            break

    fullscreen = await client.post(
        "/api/v1/threads/shared-sess/runs/stream",
        cookies=valid_cookie,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert fullscreen.status_code == 409

    release.set()
    await widget_task


async def test_widget_thread_owned_by_creator(client, valid_cookie_for, monkeypatch):
    """A widget turn claims ownership; a stranger's runs/stream on it → 404."""

    async def _fake_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    async def _fake_registry():
        from types import SimpleNamespace

        return SimpleNamespace(catalog={"servers": []})

    monkeypatch.setattr("src.api.routes.stream_agent", _fake_stream)
    monkeypatch.setattr("src.api.routes.get_registry", _fake_registry)
    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake_stream)

    # Alice runs a widget turn on a new session → she owns it.
    widget = await client.post(
        "/api/v1/query",
        cookies=valid_cookie_for("alice@x"),
        json={"query": "hi", "session_id": "widget-sess"},
    )
    assert widget.status_code == 200
    _ = widget.text  # drain the stream so the generator's finally releases the lock

    assert get_thread_owner_store().check_access("widget-sess", "alice@x") is True

    # Bob can't run Alice's widget-born thread.
    bob = await client.post(
        "/api/v1/threads/widget-sess/runs/stream",
        cookies=valid_cookie_for("bob@x"),
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert bob.status_code == 404


async def test_widget_registry_failure_does_not_leak_lock(client, valid_cookie, monkeypatch):
    """get_registry() failing inside the widget stream must release the per-thread lock.

    acquire_thread_run runs in the handler; if get_registry() (MCP catalog fetch,
    can fail on a cold/refreshing cache) raised OUTSIDE held_run, the lock would
    never be released and the session_id would be permanently 409-locked.
    """

    async def _boom_registry():
        raise RuntimeError("catalog cold")

    monkeypatch.setattr("src.api.routes.get_registry", _boom_registry)

    r = await client.post(
        "/api/v1/query",
        cookies=valid_cookie,
        json={"query": "hi", "session_id": "leak-sess"},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream; the error path runs and held_run's finally fires

    import src.api.thread_runs as tr

    assert "leak-sess" not in tr._active_by_thread  # lock released, not leaked
    # A subsequent acquire on the same session must NOT 409.
    tr.acquire_thread_run("leak-sess", "next-run")  # raises HTTPException(409) if leaked


# ---------------------------------------------------------------------------
# thread_run_stream error branches: cancel / timeout / generic exception,
# and final_messages built from a real "updates" chunk.
# ---------------------------------------------------------------------------


async def test_run_stream_builds_final_messages_from_updates_chunk(
    client, valid_cookie, monkeypatch
):
    """The 'updates' branch pulls node_output['messages'] into final_messages,
    which the 'values' event carries even when no messages/partial ever fired."""

    async def _fake(**_kwargs):
        yield "updates", {"some_node": {"messages": [{"type": "human", "content": "hi"}]}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)

    r = await client.post(
        "/api/v1/threads/t-updates/runs/stream",
        cookies=valid_cookie,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert 'event: values\ndata: {"messages": [{"type": "human", "content": "hi"}]}' in r.text


async def test_run_stream_cancelled_mid_turn_emits_error_and_propagates(
    valid_cookie_for, seed_owner
):
    """A real cancel (client disconnect / cancel_run) while streaming emits
    event: error error:cancelled, then propagates so the task truly cancels.

    Drives thread_run_stream directly (not via httpx) and cancels the task
    consuming its body_iterator — ASGITransport doesn't cleanly support
    cancelling the serving task out from under a client.post() call, but the
    endpoint itself is an ordinary async function returning a
    StreamingResponse, so this exercises the real code path without that
    transport limitation.
    """
    from types import SimpleNamespace

    from src.api.thread_routes import RunRequest, thread_run_stream

    seed_owner("t-cancel-mid", "canceller@x")

    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()

    started = asyncio.Event()
    saw_cancelled = asyncio.Event()

    async def _long_stream(**_kwargs):
        started.set()
        try:
            await asyncio.sleep(10)
            yield "updates", {}
        except asyncio.CancelledError:
            saw_cancelled.set()
            raise

    cookie_token = valid_cookie_for("canceller@x")["SESSaccess_auth"]
    raw_request = SimpleNamespace(
        cookies={"SESSaccess_auth": cookie_token},
        app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)),
    )
    body = RunRequest.model_validate({"input": {"messages": [{"role": "user", "content": "hi"}]}})

    collected: list[str] = []

    async def _consume() -> None:
        with patch("src.api.thread_routes.stream_agent", _long_stream):
            response = await thread_run_stream("t-cancel-mid", body, raw_request)
            async for chunk in response.body_iterator:
                collected.append(chunk)

    task = asyncio.ensure_future(_consume())
    await started.wait()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "t-cancel-mid" in tr._active_by_thread:
            break

    run_id = tr._active_by_thread["t-cancel-mid"]
    assert tr.cancel_run("t-cancel-mid", run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await task

    assert saw_cancelled.is_set()
    assert any('event: error\ndata: {"error": "cancelled"}' in c for c in collected)
    assert "t-cancel-mid" not in tr._active_by_thread


async def test_run_stream_timeout_emits_error_and_returns(client, valid_cookie, monkeypatch):
    """asyncio.timeout expiry (TimeoutError, not CancelledError in 3.11) emits
    event: error error:timeout and returns without re-raising."""
    from src.config import settings

    monkeypatch.setattr(settings, "AGENT_TURN_TIMEOUT_S", 0, raising=False)

    async def _slow_stream(**_kwargs):
        await asyncio.sleep(0.05)
        yield "updates", {}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _slow_stream)

    r = await client.post(
        "/api/v1/threads/t-timeout/runs/stream",
        cookies=valid_cookie,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert 'event: error\ndata: {"error": "timeout"}' in r.text


async def test_run_stream_generic_exception_emits_agent_error(client, valid_cookie, monkeypatch):
    """An unexpected exception from stream_agent is caught, logged, and turned
    into event: error error:agent_error rather than propagating as a 500."""

    async def _boom(**_kwargs):
        raise RuntimeError("unexpected agent failure")
        yield  # pragma: no cover - unreachable, makes this an async generator

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _boom)

    r = await client.post(
        "/api/v1/threads/t-boom/runs/stream",
        cookies=valid_cookie,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert 'event: error\ndata: {"error": "agent_error"}' in r.text


# ---------------------------------------------------------------------------
# Resolve-first ownership gate: anon->authed upgrade (replaces claim-then-check)
# ---------------------------------------------------------------------------


async def test_run_anon_thread_resumed_by_authed_caller_upgrades_ownership(
    client, valid_cookie_for, seed_owner, monkeypatch
):
    """An anon-owned thread resumed by an authed caller flips ownership: the
    anon owner later signs in and resolve_owner reports was_authenticated=True.
    Mirrors /query's upgrade test in tests/test_query_ownership.py, minus the
    sidebar assertion (turn_reports writes are Task 3a's job)."""
    _mock_stream_agent(monkeypatch)
    seed_owner("t-upgrade", None)  # anon-owned

    r = await client.post(
        "/api/v1/threads/t-upgrade/runs/stream",
        cookies=valid_cookie_for("me@access-ci.org"),
        json={"input": {"messages": [{"role": "user", "content": "hi again"}]}},
    )
    assert r.status_code == 200

    owner = get_thread_owner_store().resolve_owner("t-upgrade")
    assert owner is not None
    assert owner.was_authenticated is True

    import hashlib

    expected_hash = hashlib.sha256(b"me@access-ci.org").hexdigest()[:16]
    assert owner.user_hash == expected_hash


async def test_run_anon_owned_thread_resumed_by_anon_caller_stays_anon(
    client, seed_owner, monkeypatch
):
    """Existing anon-owned thread + the anon owner resuming (no cookie) ->
    allowed, ownership stays anon-owned. Neither gate branch fires (owner is
    anon-owned and caller is None), so this falls through to the final
    unconditional claim_thread, which must be a harmless no-op here."""
    _mock_stream_agent(monkeypatch)
    seed_owner("t-anon-resume", None)  # anon-owned

    r = await client.post(
        "/api/v1/threads/t-anon-resume/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "hi again"}]}},
    )
    assert r.status_code == 200

    owner = get_thread_owner_store().resolve_owner("t-anon-resume")
    assert owner is not None
    assert owner.was_authenticated is False
    assert owner.user_hash is None


async def test_run_authed_owned_thread_wrong_caller_404(
    client, valid_cookie_for, seed_owner, monkeypatch
):
    """Authed-owned thread + wrong authed caller -> 404 (never 403), via the
    single check_access call in the resolve-first gate."""
    _mock_stream_agent(monkeypatch)
    seed_owner("t-owned-wrong-caller", "alice@x")

    r = await client.post(
        "/api/v1/threads/t-owned-wrong-caller/runs/stream",
        cookies=valid_cookie_for("bob@x"),
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert r.status_code == 404


async def test_run_authed_owned_path_calls_check_access_exactly_once(
    client, valid_cookie_for, seed_owner, monkeypatch
):
    """Guard against the double-gate regression: Task 2's claim-then-check
    fallback (`if owned is False: _require_access(...)`) must be gone. The
    resolve-first gate has exactly ONE check_access call on the authed-owned
    path, inside the branch that mirrors routes.py's /query gate."""
    _mock_stream_agent(monkeypatch)
    seed_owner("t-single-check", "alice@x")

    with patch(
        "src.thread_owners.ThreadOwnerStore.check_access",
        wraps=get_thread_owner_store().check_access,
    ) as spy:
        r = await client.post(
            "/api/v1/threads/t-single-check/runs/stream",
            cookies=valid_cookie_for("alice@x"),
            json={"input": {"messages": [{"role": "user", "content": "x"}]}},
        )
        assert r.status_code == 200
        assert spy.call_count == 1


async def test_run_losing_concurrent_upgrade_caller_does_not_backfill(
    client, valid_cookie_for, seed_owner, monkeypatch
):
    """Two authed callers race their first post-login turn on the same
    anon-owned thread. The loser's resolve_owner read is stale (pre-dates the
    winner's flip); the loser's own upgrade_owner call must lose the race
    (returns False) and therefore must NOT call backfill_user_hash — mirrors
    tests/test_query_ownership.py::test_losing_race_caller_does_not_backfill_winners_thread,
    adapted: this path doesn't write turn_reports rows (Task 3a), so the
    property is verified by spying on backfill_user_hash rather than
    inspecting a historical row.
    """
    from src.thread_owners import ThreadOwner

    _mock_stream_agent(monkeypatch)
    seed_owner("t-race", None)  # anon-owned

    owner_store = get_thread_owner_store()
    # Alice's upgrade_owner call has already won the race and committed.
    assert owner_store.upgrade_owner("t-race", "alice@access-ci.org") is True

    # Bob's request resolved the owner as anon-owned (a stale read taken
    # before alice's flip committed) — fake only the read; the real
    # upgrade_owner call below still hits the live (already-flipped) row.
    stale_anon_owner = ThreadOwner(False, None)
    with (
        patch.object(owner_store, "resolve_owner", return_value=stale_anon_owner),
        patch("src.api.thread_routes.get_turn_reporter") as mock_get_reporter,
    ):
        mock_reporter = mock_get_reporter.return_value
        r = await client.post(
            "/api/v1/threads/t-race/runs/stream",
            cookies=valid_cookie_for("bob@access-ci.org"),
            json={"input": {"messages": [{"role": "user", "content": "hi again"}]}},
        )
        assert r.status_code == 200
        mock_reporter.backfill_user_hash.assert_not_called()

    # Ownership must remain alice's — bob's upgrade_owner call lost the race.
    owner = owner_store.resolve_owner("t-race")
    assert owner is not None
    assert owner.was_authenticated is True
    import hashlib

    assert owner.user_hash == hashlib.sha256(b"alice@access-ci.org").hexdigest()[:16]


# ---------------------------------------------------------------------------
# Task 3a: turn_reports row written from the thread/run path
# ---------------------------------------------------------------------------


async def test_run_stream_writes_one_turn_report_row(client, monkeypatch):
    """An anon thread/run turn writes exactly one turn_reports row: session_id
    is the thread_id, user_hash is NULL (anon caller), query_text is the
    user's text. This is the real proof the reporter block ran (not just "no
    exception") — a forgotten get_turn_capture import would NameError, get
    swallowed by the block's `except Exception`, and silently write zero
    rows."""
    from src.turn_reporter import get_turn_reporter

    _mock_stream_agent(monkeypatch, content="the answer")

    r = await client.post(
        "/api/v1/threads/t-report-row/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "what is ACCESS?"}]}},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream so the post-stream reporter block runs

    reporter = get_turn_reporter()
    assert reporter.count_turns_for_session("t-report-row") == 1

    session = reporter._session_factory()
    try:
        from src.turn_reporter import TurnReport

        row = session.query(TurnReport).filter_by(session_id="t-report-row").one()
    finally:
        session.close()

    assert row.session_id == "t-report-row"
    assert row.user_hash is None
    assert row.query_text == "what is ACCESS?"
    assert row.success is True


async def test_run_stream_failed_turn_writes_success_false_row(client, monkeypatch):
    """A thread/run turn whose stream_agent raises mid-stream still writes
    exactly one turn_reports row, with success=False — failed-turn parity
    with /query's except-block write (routes.py:500-527)."""
    from src.turn_reporter import get_turn_reporter

    async def _boom(**_kwargs):
        yield (
            "messages",
            (AIMessageChunk(content="partial", id="m1"), {"langgraph_node": "tool_calling_loop"}),
        )
        raise RuntimeError("mid-stream failure")

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _boom)

    r = await client.post(
        "/api/v1/threads/t-report-fail/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "will this fail?"}]}},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream so the error-path reporter block runs

    reporter = get_turn_reporter()
    assert reporter.count_turns_for_session("t-report-fail") == 1

    session = reporter._session_factory()
    try:
        from src.turn_reporter import TurnReport

        row = session.query(TurnReport).filter_by(session_id="t-report-fail").one()
    finally:
        session.close()

    assert row.success is False
    assert row.query_text == "will this fail?"
