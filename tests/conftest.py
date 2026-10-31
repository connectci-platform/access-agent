"""Pytest configuration and fixtures."""

import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import algorithms as jwt_algorithms

# Load .env file for tests if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with env_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


@pytest.fixture(autouse=True)
def _reset_async_singletons():
    """Drop module-global cached httpx clients between tests.

    ``uky_client._client`` and ``mcp_client._shared_client`` each cache an
    ``httpx.AsyncClient`` bound to whichever event loop was live when it was
    first created. pytest-asyncio gives each test a fresh function-scoped
    loop, so a client created in one test is reused in the next against a
    closed loop -- surfacing as ``RuntimeError: Event loop is closed``.
    ``is_closed`` does not catch this (a dead loop does not close the
    client), so resetting the globals is the reliable fix: each test
    rebuilds its client on its own loop.

    Also resets the turn_capture ContextVar to None so that top-level
    reset_turn_capture() calls in one test don't bleed into the next.
    """
    yield
    from src.services import uky_client
    from src.tools import mcp_client

    uky_client._client = None
    mcp_client._shared_client = None

    # Clear the turn_capture ContextVar between tests to prevent top-level
    # reset_turn_capture() calls in one test from polluting the next.
    try:
        from src.agent.turn_capture import _turn_capture

        _turn_capture.set(None)
    except ImportError:
        pass


@pytest.fixture
def sample_catalog():
    """Sample MCP tool catalog for testing."""
    return {
        "quick_lookup": {
            "search_resources": {"server": "compute-resources"},
            "get_current_outages": {"server": "system-status"},
            "search_software": {"server": "software-discovery"},
        },
        "tools": [
            {
                "name": "search_resources",
                "description": "Search for compute resources",
                "parameters": [
                    {"name": "has_gpu", "type": "boolean", "required": False},
                    {"name": "resource_type", "type": "string", "required": False},
                ],
            },
            {
                "name": "get_current_outages",
                "description": "Get current system outages",
                "parameters": [
                    {"name": "resource_filter", "type": "string", "required": False},
                ],
            },
            {
                "name": "search_software",
                "description": "Search for software packages",
                "parameters": [
                    {"name": "query", "type": "string", "required": True},
                    {"name": "resource", "type": "string", "required": False},
                ],
            },
        ],
    }


@pytest.fixture
def sample_query():
    """Sample user query."""
    return "What GPU resources are available on ACCESS?"


# --- Bearer agent tokens -----------------------------------------------------

AGENT_TOKEN_ISSUER = "https://support.example.test"
AGENT_TOKEN_AUDIENCE = "https://qa.example.test"
_AGENT_TOKEN_KID = "test-agent-kid"
_agent_token_key = ec.generate_private_key(ec.SECP256R1())


class _ExchangeJWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        jwk = json.loads(
            jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(
                _agent_token_key.public_key()
            )
        )
        jwk.update({"kid": _AGENT_TOKEN_KID, "use": "sig", "alg": "ES256"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"keys": [jwk]}).encode())

    def log_message(self, format, *args):
        pass


class AgentTokens:
    """Mints agent tokens that the app's verifier accepts."""

    def token(self, sub: str, **claims: Any) -> str:
        now = int(time.time())
        payload = {
            "iss": AGENT_TOKEN_ISSUER,
            "aud": AGENT_TOKEN_AUDIENCE,
            "azp": "access-qa-bot",
            "sub": sub,
            "jti": secrets.token_hex(8),
            "scope": "agent:read agent:write",
            "iat": now,
            "exp": now + 900,
        }
        payload.update(claims)
        return jwt.encode(
            payload,
            _agent_token_key,
            algorithm="ES256",
            headers={"kid": _AGENT_TOKEN_KID, "typ": "at+jwt"},
        )

    def headers(self, sub: str, **claims: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(sub, **claims)}"}


@pytest.fixture
async def agent_tokens():
    """Install a verifier on the app, backed by a local exchange JWKS.

    ASGITransport doesn't run the lifespan, so the verifier it would create
    is set here instead.
    """
    from src.agent_token import AgentTokenVerifier
    from src.main import app

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ExchangeJWKSHandler)
    Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    verifier = AgentTokenVerifier(
        issuer=AGENT_TOKEN_ISSUER,
        jwks_url=f"http://127.0.0.1:{server.server_address[1]}/.well-known/exchange-jwks.json",
        audience=AGENT_TOKEN_AUDIENCE,
        allowed_azp=frozenset({"access-qa-bot"}),
    )
    app.state.agent_token_verifier = verifier
    yield AgentTokens()
    del app.state.agent_token_verifier
    await verifier.aclose()
    server.shutdown()
    server.server_close()
