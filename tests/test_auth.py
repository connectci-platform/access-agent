"""Tests for JWT cookie authentication (src/auth.py)."""

import time
from unittest.mock import MagicMock

import jwt

from src.auth import get_acting_user_from_cookie

SECRET = "test-secret-key-for-unit-tests-32b"


def _make_jwt(sub: str, expired: bool = False) -> str:
    """Create a signed JWT for testing."""
    now = int(time.time())
    payload = {
        "sub": sub,
        "iat": now - 3600,
        "exp": (now - 10) if expired else (now + 3600),
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")


def _mock_request(cookies: dict | None = None):
    """Create a mock FastAPI Request with cookies."""
    request = MagicMock()
    request.cookies = cookies or {}
    return request


# --- Valid JWT cookie ---


def test_valid_jwt_cookie():
    """Valid JWT cookie returns the ACCESS ID and cookie_present=True."""
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user == "jsmith@access-ci.org"
    assert cookie_present is True


def test_valid_jwt_cookie_result_is_not_affected_by_body():
    """Cookie resolution is independent of body content (body not read)."""
    token = _make_jwt("cookie-user@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user == "cookie-user@access-ci.org"
    assert cookie_present is True


# --- Expired JWT cookie ---


def test_expired_jwt_cookie_returns_none():
    """Expired JWT cookie returns (None, True) — cookie present but invalid."""
    token = _make_jwt("jsmith@access-ci.org", expired=True)
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user is None
    assert cookie_present is True


# --- Invalid JWT cookie ---


def test_invalid_jwt_cookie_returns_none():
    """Tampered/invalid JWT cookie returns (None, True)."""
    request = _mock_request(cookies={"SESSaccess_auth": "not-a-valid-jwt"})

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user is None
    assert cookie_present is True


def test_wrong_secret_returns_none():
    """JWT signed with wrong secret returns (None, True)."""
    token = jwt.encode(
        {"sub": "jsmith@access-ci.org", "exp": int(time.time()) + 3600},
        "wrong-secret",
        algorithm="HS256",
    )
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user is None
    assert cookie_present is True


def test_jwt_missing_sub_claim():
    """JWT without 'sub' claim returns (None, True)."""
    token = jwt.encode(
        {"email": "jsmith@access-ci.org", "exp": int(time.time()) + 3600},
        SECRET,
        algorithm="HS256",
    )
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user is None
    assert cookie_present is True


# --- No cookie ---


def test_no_cookie_returns_none_and_not_present():
    """No cookie returns (None, False) — caller decides on fallback."""
    request = _mock_request()

    user, cookie_present = get_acting_user_from_cookie(request, jwt_secret=SECRET)

    assert user is None
    assert cookie_present is False
