"""End-to-end tests for Bearer agent-token identity through the FastAPI app.

Tests the full HTTP request path: Authorization header → request_identity →
route handler. Mocks only the LLM agent so no OpenAI/MCP infrastructure is
needed. Tokens come from the shared ``agent_tokens`` fixture.

Identity is token-only: QueryRequest has no acting_user body field, and the
SESSaccess_auth cookie is never read, so a caller without a token is always
anonymous. Tests below that send an `acting_user` key in the JSON body are
verifying it is silently ignored (unknown field, no `extra="forbid"`), not
exercising a real fallback.

Run with: uv run pytest tests/test_auth_e2e.py -v
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Force-empty (not setdefault): this suite mocks the agent and must never touch
# a real DB. With setdefault, an exported DATABASE_URL (normal dev shell) leaks
# through and the route's turn reporter writes real rows to local Postgres.
os.environ["DATABASE_URL"] = ""  # Disable checkpointing + turn reporting

from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Fake agent response to avoid LLM/MCP dependencies
FAKE_AGENT_RESULT = {
    "final_answer": "Test response",
    "tools_used": [],
    "query_analysis": None,
    "query_classification": None,
}


@pytest.fixture(autouse=True)
def _sqlite_owner_store(monkeypatch, tmp_path):
    """Point the thread-owner store at a temp sqlite file for this module.

    The module-level DATABASE_URL="" above keeps the *turn reporter* off a
    real DB; thread_owners.py reads settings.DATABASE_URL independently, so it
    needs its own override to exercise claim_thread here.
    """
    from src.config import settings

    monkeypatch.setattr(
        settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'owners.db'}", raising=False
    )
    import src.thread_owners as m

    m._store = None  # reset singleton
    yield
    m._store = None


@pytest.fixture(autouse=True)
def _disable_turnstile(monkeypatch):
    """Isolate these tests from local .env TURNSTILE_SECRET_KEY leakage.

    These tests exercise the anonymous-user path through /api/v1/query.
    If TURNSTILE_SECRET_KEY is set (e.g. from a local .env loaded by
    conftest), the route returns 400 before auth runs, breaking tests
    that are unrelated to Turnstile.
    """
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)


@pytest.fixture
def mock_agent():
    """Mock stream_agent to avoid needing LLM/MCP infrastructure.

    stream_agent is an async generator that yields (stream_type, chunk) tuples.
    We mock it to yield a single 'updates' chunk containing FAKE_AGENT_RESULT,
    which is enough for the SSE endpoint to build a done event.
    """

    async def fake_stream(**kwargs):
        yield "updates", {"__end__": FAKE_AGENT_RESULT}

    with patch("src.api.routes.stream_agent", side_effect=fake_stream) as mock:
        yield mock


@pytest.fixture
def mock_registry():
    """Mock get_registry to avoid MCP catalog fetch."""
    mock_reg = AsyncMock()
    mock_reg.catalog = {"tools": [], "quick_lookup": {}}
    with patch("src.api.routes.get_registry", return_value=mock_reg):
        yield mock_reg


@pytest.fixture
async def client():
    """Create an httpx AsyncClient bound to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Who the caller is
# ---------------------------------------------------------------------------


async def test_valid_token_sets_acting_user(client, mock_agent, mock_registry, agent_tokens):
    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        headers=agent_tokens.headers("jsmith@access-ci.org"),
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") == "jsmith@access-ci.org"


async def test_no_token_anonymous(client, mock_agent, mock_registry):
    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


async def test_cookie_only_request_is_anonymous(client, mock_agent, mock_registry, agent_tokens):
    """The cookie is never read, even when it holds a token that would verify."""
    token = agent_tokens.token("jsmith@access-ci.org")

    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        headers={"cookie": f"SESSaccess_auth={token}"},
    )

    assert response.status_code == 200
    assert mock_agent.call_args.kwargs.get("acting_user") is None


async def test_body_acting_user_is_ignored_without_token(client, mock_agent, mock_registry):
    """A body acting_user is NOT used — it is silently ignored (unknown field
    on QueryRequest) and the caller is anonymous.
    """
    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "session_id": "s-body-ignored",
            "acting_user": "body-user@access-ci.org",
        },
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


async def test_token_overrides_body(client, mock_agent, mock_registry, agent_tokens):
    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "acting_user": "body-user@access-ci.org",
        },
        headers=agent_tokens.headers("token-user@access-ci.org"),
    )

    assert response.status_code == 200
    assert mock_agent.call_args.kwargs.get("acting_user") == "token-user@access-ci.org"


async def test_body_acting_user_does_not_bypass_turnstile(
    client, mock_agent, mock_registry, monkeypatch
):
    """Turnstile enabled, no token, no turnstile token, but a body acting_user
    is present. This caller is anonymous and must be challenged.
    """
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "test-secret", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate", raising=False)

    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "session_id": "s-turnstile-no-bypass",
            "acting_user": "spoofed-user@access-ci.org",
        },
    )

    assert response.status_code == 200
    assert response.json().get("requires_turnstile") is True
    mock_agent.assert_not_called()


async def test_response_format(client, mock_agent, mock_registry, agent_tokens):
    """The SSE stream contains a done event with the expected fields."""
    response = await client.post(
        "/api/v1/query",
        json={"query": "Test question", "session_id": "test-sess", "question_id": "test-q"},
        headers=agent_tokens.headers("jsmith@access-ci.org"),
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    done_event = None
    for block in response.text.split("\n\n"):
        if block.startswith("event: done"):
            for line in block.split("\n"):
                if line.startswith("data: "):
                    done_event = json.loads(line[6:])
                    break

    assert done_event is not None, "No 'done' SSE event found"
    assert done_event["success"] is True
    assert done_event["response"] == "Test response"
    assert done_event["metadata"]["question_id"] == "test-q"
    assert isinstance(done_event["metadata"]["tools_used"], list)


# ---------------------------------------------------------------------------
# Thread ownership on the widget path
# ---------------------------------------------------------------------------


async def test_body_acting_user_claims_anon_owned_thread(client, mock_agent, mock_registry):
    """A widget turn without a token claims an anon-owned thread, whatever the
    body says: a spoofable body field can't write a was_authenticated=True row.
    """
    session_id = "sess-body-only"
    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "session_id": session_id,
            "acting_user": "spoofed-user@access-ci.org",
        },
    )

    assert response.status_code == 200
    assert mock_agent.call_args.kwargs.get("acting_user") is None
    owner = get_thread_owner_store().resolve_owner(session_id)
    assert owner is not None
    assert owner.was_authenticated is False
    assert owner.user_hash is None


async def test_valid_token_claims_authenticated_owned_thread(
    client, mock_agent, mock_registry, agent_tokens
):
    session_id = "sess-token-owned"
    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?", "session_id": session_id},
        headers=agent_tokens.headers("jsmith@access-ci.org"),
    )

    assert response.status_code == 200
    assert mock_agent.call_args.kwargs.get("acting_user") == "jsmith@access-ci.org"
    owner = get_thread_owner_store().resolve_owner(session_id)
    assert owner is not None
    assert owner.was_authenticated is True
    assert owner.user_hash is not None


# ---------------------------------------------------------------------------
# Unauthenticated endpoints and /whoami
# ---------------------------------------------------------------------------


async def test_health_no_auth(client):
    """Health works without auth. Status may be "healthy" or "degraded"
    depending on the shared catalog aggregator's state; both mean reachable.
    """
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("healthy", "degraded")


async def test_whoami_valid_token(client, agent_tokens):
    response = await client.get(
        "/api/v1/whoami", headers=agent_tokens.headers("jsmith@access-ci.org")
    )

    assert response.status_code == 200
    assert response.json() == {"authenticated": True, "user": "jsmith@access-ci.org"}


async def test_whoami_no_token(client):
    response = await client.get("/api/v1/whoami")

    assert response.status_code == 200
    assert response.json() == {"authenticated": False, "user": None}


async def test_whoami_never_cached_by_intermediary(client, agent_tokens):
    """/whoami is an identity response and must never be cached by any
    intermediary — a shared cache serving one caller's cached body to
    another would leak identity across users."""
    response = await client.get(
        "/api/v1/whoami", headers=agent_tokens.headers("jsmith@access-ci.org")
    )

    assert response.status_code == 200
    assert response.headers.get("cache-control") == "private, no-store"
    assert response.headers.get("vary") == "Authorization"


async def test_capabilities_unlock_for_a_token(client, agent_tokens):
    anonymous = await client.get("/api/v1/capabilities")
    signed_in = await client.get(
        "/api/v1/capabilities", headers=agent_tokens.headers("jsmith@access-ci.org")
    )

    assert anonymous.json()["is_authenticated"] is False
    assert signed_in.json()["is_authenticated"] is True


@pytest.mark.parametrize("signed_in", [True, False], ids=["token", "anonymous"])
async def test_rating_is_bound_to_the_token_identity(client, agent_tokens, signed_in):
    usage_logger = MagicMock()
    usage_logger.log_rating.return_value = "ok"
    headers = agent_tokens.headers("jsmith@access-ci.org") if signed_in else {}

    with (
        patch("src.api.routes.get_usage_logger", return_value=usage_logger),
        patch("src.turn_reporter.get_turn_reporter"),
    ):
        response = await client.post(
            "/api/v1/rating",
            json={"query_id": "q-1", "rating": "helpful", "session_id": "s-1"},
            headers=headers,
        )

    assert response.status_code == 200
    assert usage_logger.log_rating.call_args.kwargs["acting_user"] == (
        "jsmith@access-ci.org" if signed_in else None
    )
