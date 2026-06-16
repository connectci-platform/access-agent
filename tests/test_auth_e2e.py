"""End-to-end tests for JWT cookie authentication through the FastAPI app.

Tests the full HTTP request path: cookies → auth → route handler.
Mocks only the LLM agent so no OpenAI/MCP infrastructure is needed.

Uses ES256 (ECDSA P-256) key pairs — no shared secret.

Body fallback is handled in the route (using the already-parsed
QueryRequest.acting_user), NOT in auth.py — so the ASGI body stream
is only consumed once by FastAPI.

Run with: uv run pytest tests/test_auth_e2e.py -v
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import AsyncMock, patch

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

os.environ.setdefault("ALLOW_BODY_ACTING_USER", "true")
# Force-empty (not setdefault): this suite mocks the agent and must never touch
# a real DB. With setdefault, an exported DATABASE_URL (normal dev shell) leaks
# through and the route's turn reporter writes real rows to local Postgres.
os.environ["DATABASE_URL"] = ""  # Disable checkpointing + turn reporting
os.environ.setdefault("TRUSTED_JWKS_URLS", "")  # Configured per-test

from src.auth import configure_trusted_issuers
from src.main import app

# ---------------------------------------------------------------------------
# Test key pair and JWKS server
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "e2e-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "e2e-test-kid"


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


def _make_jwt(sub: str, expired: bool = False, issuer: str = ISSUER) -> str:
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": sub,
        "iat": now - 3600,
        "exp": (now - 10) if expired else (now + 3600),
    }
    return jwt.encode(
        payload,
        PRIVATE_PEM,
        algorithm="ES256",
        headers={"kid": KID},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Fake agent response to avoid LLM/MCP dependencies
FAKE_AGENT_RESULT = {
    "final_answer": "Test response",
    "tools_used": [],
    "query_analysis": None,
    "query_classification": None,
    "execution_strategy": "test",
}


@pytest.fixture(autouse=True)
def _jwks_server():
    """Start a local JWKS server and configure trusted issuers for all tests."""
    server, jwks_url = _start_jwks_server()
    configure_trusted_issuers({ISSUER: jwks_url})
    yield
    server.shutdown()
    configure_trusted_issuers({})


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
# Test: Valid JWT cookie → acting_user extracted
# ---------------------------------------------------------------------------


async def test_valid_cookie_sets_acting_user(client, mock_agent, mock_registry):
    """A valid ES256 JWT cookie should result in acting_user being passed to the agent."""
    token = _make_jwt("jsmith@access-ci.org")

    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        headers={"cookie": f"SESSaccess_auth={token}"},
    )

    assert response.status_code == 200
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
        headers={"cookie": f"SESSaccess_auth={token}"},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


# ---------------------------------------------------------------------------
# Test: Invalid/tampered cookie → anonymous
# ---------------------------------------------------------------------------


async def test_invalid_cookie_anonymous(client, mock_agent, mock_registry):
    """Tampered cookie → anonymous."""
    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        headers={"cookie": "SESSaccess_auth=this-is-not-a-jwt"},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") is None


# ---------------------------------------------------------------------------
# Test: Wrong signing key → anonymous
# ---------------------------------------------------------------------------


async def test_wrong_key_anonymous(client, mock_agent, mock_registry):
    """JWT signed with an unknown key → anonymous."""
    other_key = ec.generate_private_key(ec.SECP256R1())
    other_pem = other_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    now = int(time.time())
    token = jwt.encode(
        {"iss": ISSUER, "sub": "jsmith@access-ci.org", "exp": now + 3600},
        other_pem,
        algorithm="ES256",
        headers={"kid": "unknown-kid"},
    )

    response = await client.post(
        "/api/v1/query",
        json={"query": "What GPU resources are available?"},
        headers={"cookie": f"SESSaccess_auth={token}"},
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
        headers={"cookie": f"SESSaccess_auth={token}"},
    )

    assert response.status_code == 200
    mock_agent.assert_called_once()
    assert mock_agent.call_args.kwargs.get("acting_user") == "cookie-user@access-ci.org"


# ---------------------------------------------------------------------------
# Test: Response format is correct
# ---------------------------------------------------------------------------


async def test_response_format(client, mock_agent, mock_registry):
    """Verify the SSE stream contains a done event with expected fields."""
    token = _make_jwt("jsmith@access-ci.org")

    response = await client.post(
        "/api/v1/query",
        json={
            "query": "Test question",
            "session_id": "test-sess",
            "question_id": "test-q",
        },
        headers={"cookie": f"SESSaccess_auth={token}"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Parse SSE events from response text
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
# Test: Health endpoint still works (no auth needed)
# ---------------------------------------------------------------------------


async def test_health_no_auth(client):
    """Health endpoint should work without any auth."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
