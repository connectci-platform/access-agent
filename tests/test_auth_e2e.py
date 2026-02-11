"""End-to-end tests for JWT cookie authentication through the FastAPI app.

Tests the full HTTP request path: cookies → auth → route handler.
Mocks only the LLM agent so no OpenAI/MCP infrastructure is needed.

Body fallback is handled in the route (using the already-parsed
QueryRequest.acting_user), NOT in auth.py — so the ASGI body stream
is only consumed once by FastAPI.

Run with: uv run pytest tests/test_auth_e2e.py -v
"""

import os
import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

# Set a test JWT secret before importing the app (config reads env at import)
TEST_SECRET = "e2e-test-jwt-secret-32-bytes-ok!"
os.environ.setdefault("JWT_SECRET", TEST_SECRET)
os.environ.setdefault("ALLOW_BODY_ACTING_USER", "true")
os.environ.setdefault("VAULT_TOKEN", "")  # Disable Vault for tests
os.environ.setdefault("DATABASE_URL", "")  # Disable checkpointing

from src.main import app  # noqa: E402


def _make_jwt(sub: str, expired: bool = False, secret: str = TEST_SECRET) -> str:
    """Create a signed JWT for testing."""
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "iat": now - 3600,
            "exp": (now - 10) if expired else (now + 3600),
        },
        secret,
        algorithm="HS256",
    )


# Fake agent response to avoid LLM/MCP dependencies
FAKE_AGENT_RESULT = {
    "final_answer": "Test response",
    "tools_used": [],
    "query_analysis": None,
    "query_classification": None,
    "execution_strategy": "test",
}


@pytest.fixture
def mock_agent():
    """Mock run_agent to avoid needing LLM/MCP infrastructure."""
    with patch("src.api.routes.run_agent", new_callable=AsyncMock) as mock:
        mock.return_value = FAKE_AGENT_RESULT
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
# Test: Valid JWT cookie → acting_user extracted
# ---------------------------------------------------------------------------


async def test_valid_cookie_sets_acting_user(client, mock_agent, mock_registry):
    """A valid JWT cookie should result in acting_user being passed to the agent."""
    token = _make_jwt("jsmith@access-ci.org")

    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        cookies={"SESSaccess_auth": token},
    )

    assert response.status_code == 200
    # Verify the agent was called with the correct acting_user
    mock_agent.assert_called_once()
    call_kwargs = mock_agent.call_args
    assert call_kwargs.kwargs.get("acting_user") == "jsmith@access-ci.org"


# ---------------------------------------------------------------------------
# Test: No cookie, no body → anonymous
# ---------------------------------------------------------------------------


async def test_no_cookie_anonymous(client, mock_agent, mock_registry):
    """No cookie and no acting_user body field → anonymous user."""
    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


# ---------------------------------------------------------------------------
# Test: Expired cookie → anonymous (no body fallback)
# ---------------------------------------------------------------------------


async def test_expired_cookie_anonymous(client, mock_agent, mock_registry):
    """Expired JWT cookie → anonymous, even if body acting_user is present."""
    token = _make_jwt("jsmith@access-ci.org", expired=True)

    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "acting_user": "body-user@access-ci.org",
        },
        cookies={"SESSaccess_auth": token},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    # Expired cookie means anonymous — body fallback should NOT be used
    assert mock_agent.call_args.kwargs.get("acting_user") is None


# ---------------------------------------------------------------------------
# Test: Invalid/tampered cookie → anonymous
# ---------------------------------------------------------------------------


async def test_invalid_cookie_anonymous(client, mock_agent, mock_registry):
    """Tampered cookie → anonymous."""
    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        cookies={"SESSaccess_auth": "this-is-not-a-jwt"},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


# ---------------------------------------------------------------------------
# Test: Wrong signing secret → anonymous
# ---------------------------------------------------------------------------


async def test_wrong_secret_anonymous(client, mock_agent, mock_registry):
    """JWT signed with wrong secret → anonymous."""
    token = _make_jwt("jsmith@access-ci.org", secret="wrong-secret-wrong-secret-32b!")

    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        cookies={"SESSaccess_auth": token},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


# ---------------------------------------------------------------------------
# Test: Body fallback when no cookie (transition mode)
# ---------------------------------------------------------------------------


async def test_body_fallback_no_cookie(client, mock_agent, mock_registry):
    """Without a cookie, acting_user from body is used (ALLOW_BODY_ACTING_USER=true)."""
    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "acting_user": "body-user@access-ci.org",
        },
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") == "body-user@access-ci.org"


# ---------------------------------------------------------------------------
# Test: Cookie takes priority over body
# ---------------------------------------------------------------------------


async def test_cookie_overrides_body(client, mock_agent, mock_registry):
    """When both cookie and body acting_user are present, cookie wins."""
    token = _make_jwt("cookie-user@access-ci.org")

    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What GPU resources are available?",
            "acting_user": "body-user@access-ci.org",
        },
        cookies={"SESSaccess_auth": token},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") == "cookie-user@access-ci.org"


# ---------------------------------------------------------------------------
# Test: Response format is correct
# ---------------------------------------------------------------------------


async def test_response_format(client, mock_agent, mock_registry):
    """Verify the response body contains expected fields."""
    token = _make_jwt("jsmith@access-ci.org")

    response = await client.post(
        "/api/v1/query",
        json={
            "query": "Test question",
            "session_id": "test-sess",
            "question_id": "test-q",
        },
        cookies={"SESSaccess_auth": token},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["response"] == "Test response"
    assert body["session_id"] == "test-sess"
    assert body["question_id"] == "test-q"
    assert isinstance(body["tools_used"], list)


# ---------------------------------------------------------------------------
# Test: Health endpoint still works (no auth needed)
# ---------------------------------------------------------------------------


async def test_health_no_auth(client):
    """Health endpoint should work without any auth."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
