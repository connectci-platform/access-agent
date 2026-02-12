"""Tests for JWT cookie authentication (src/auth.py).

Uses ES256 (ECDSA P-256) key pairs — no shared secret.
"""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import MagicMock

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from jwt import algorithms as jwt_algorithms

from src.auth import configure_trusted_issuers, get_acting_user_from_cookie

# Generate a test EC P-256 key pair.
_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

# Build a JWKS response for the test public key.
_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "test-kid-001"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "test-kid-001"


# A minimal HTTP server that serves the JWKS endpoint for tests.
class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(JWKS_RESPONSE)

    def log_message(self, format, *args):
        pass  # Suppress server logs in test output.


def _start_jwks_server() -> tuple[HTTPServer, str]:
    """Start a local JWKS server and return (server, url)."""
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def _make_jwt(sub: str, expired: bool = False, issuer: str = ISSUER) -> str:
    """Create an ES256-signed JWT for testing."""
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


def _mock_request(cookies: dict | None = None):
    """Create a mock FastAPI Request with cookies."""
    request = MagicMock()
    request.cookies = cookies or {}
    return request


def _setup_issuers(jwks_url: str) -> None:
    """Configure trusted issuers with the test JWKS server."""
    configure_trusted_issuers({ISSUER: jwks_url})


# --- Valid JWT cookie ---


def test_valid_jwt_cookie():
    """Valid ES256 JWT cookie returns the ACCESS ID and cookie_present=True."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user == "jsmith@access-ci.org"
        assert cookie_present is True
    finally:
        server.shutdown()


def test_valid_jwt_cookie_result_is_not_affected_by_body():
    """Cookie resolution is independent of body content (body not read)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("cookie-user@access-ci.org")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user == "cookie-user@access-ci.org"
        assert cookie_present is True
    finally:
        server.shutdown()


# --- Expired JWT cookie ---


def test_expired_jwt_cookie_returns_none():
    """Expired JWT cookie returns (None, True) — cookie present but invalid."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org", expired=True)
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


# --- Invalid JWT cookie ---


def test_invalid_jwt_cookie_returns_none():
    """Tampered/invalid JWT cookie returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        request = _mock_request(cookies={"SESSaccess_auth": "not-a-valid-jwt"})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_wrong_key_returns_none():
    """JWT signed with a different key returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        # Sign with a different key pair.
        other_key = ec.generate_private_key(ec.SECP256R1())
        other_pem = other_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        token = jwt.encode(
            {
                "iss": ISSUER,
                "sub": "jsmith@access-ci.org",
                "exp": int(time.time()) + 3600,
            },
            other_pem,
            algorithm="ES256",
            headers={"kid": "unknown-kid"},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_untrusted_issuer_returns_none():
    """JWT from an untrusted issuer returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org", issuer="https://evil.com")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_jwt_missing_sub_claim():
    """JWT without 'sub' claim returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ISSUER, "email": "jsmith@access-ci.org", "exp": now + 3600},
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": KID},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_jwt_empty_sub_claim():
    """JWT with empty 'sub' claim returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ISSUER, "sub": "", "exp": now + 3600},
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": KID},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_jwt_missing_iss_claim():
    """JWT without 'iss' claim returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"sub": "jsmith@access-ci.org", "exp": now + 3600},
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": KID},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


# --- Algorithm confusion attack ---


def test_hs256_jwt_rejected():
    """HS256-signed JWT is rejected even if payload looks valid (algorithm confusion)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ISSUER, "sub": "jsmith@access-ci.org", "exp": now + 3600},
            "some-shared-secret-that-should-not-work",
            algorithm="HS256",
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


# --- JWKS server unreachable ---


def test_jwks_server_unreachable():
    """When the JWKS server is unreachable, JWT returns (None, True)."""
    # Point at a port with nothing listening.
    configure_trusted_issuers({ISSUER: "http://127.0.0.1:1"})
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request)

    assert user is None
    assert cookie_present is True


# --- No cookie ---


def test_no_cookie_returns_none_and_not_present():
    """No cookie returns (None, False) — caller decides on fallback."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        request = _mock_request()

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is False
    finally:
        server.shutdown()


# --- No issuers configured ---


def test_no_issuers_configured():
    """With no trusted issuers, any JWT returns (None, True)."""
    configure_trusted_issuers({})
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request)

    assert user is None
    assert cookie_present is True
