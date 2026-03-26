"""Tests for Turnstile bot protection."""

import time

import pytest
from pytest_httpx import HTTPXMock

from src.config import settings
from src.turnstile import TurnstileGuard, verify_turnstile_token


@pytest.fixture
def _enable_turnstile(monkeypatch):
    """Enable Turnstile by patching the settings singleton."""
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "test-secret")
    monkeypatch.setattr(settings, "TURNSTILE_SITE_KEY", "test-site-key")


@pytest.fixture
def guard(_enable_turnstile):
    """Fresh TurnstileGuard with Turnstile enabled."""
    return TurnstileGuard()


class TestTurnstileGuard:
    """Tests for TurnstileGuard session tracking."""

    def test_disabled_when_no_secret_key(self):
        """When TURNSTILE_SECRET_KEY is empty, turnstile_enabled is False."""
        assert settings.TURNSTILE_SECRET_KEY == ""
        assert settings.turnstile_enabled is False

    def test_enabled_when_secret_key_set(self, _enable_turnstile):
        assert settings.turnstile_enabled is True

    def test_disabled_guard_never_challenges(self):
        """With Turnstile disabled, requires_challenge is always False."""
        guard = TurnstileGuard()
        assert guard.requires_challenge("any-session") is False

    def test_deferred_mode_allows_free_queries(self, guard, monkeypatch):
        """In deferred mode, first N queries pass without challenge."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "deferred")
        monkeypatch.setattr(settings, "TURNSTILE_FREE_QUERIES", 3)

        session_id = "test-session-1"

        # First 3 queries should not require challenge
        for _ in range(3):
            assert guard.requires_challenge(session_id) is False
            guard.record_query(session_id)

        # 4th query should require challenge
        assert guard.requires_challenge(session_id) is True

    def test_immediate_mode_challenges_first_query(self, guard, monkeypatch):
        """In immediate mode, the first query requires a challenge."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")

        assert guard.requires_challenge("test-session-2") is True

    def test_verified_session_skips_challenge(self, guard, monkeypatch):
        """Once verified, queries pass without challenge."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 3600)

        session_id = "test-session-3"

        assert guard.requires_challenge(session_id) is True
        guard.mark_verified(session_id)
        assert guard.requires_challenge(session_id) is False

    def test_verification_expires_after_ttl(self, guard, monkeypatch):
        """Verification expires after the TTL period."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 1)

        session_id = "test-session-4"

        guard.mark_verified(session_id)
        assert guard.requires_challenge(session_id) is False

        # Manually expire the verification
        guard._sessions[session_id].verified_at = time.time() - 2
        assert guard.requires_challenge(session_id) is True

    def test_separate_sessions_are_independent(self, guard, monkeypatch):
        """Different session IDs have independent state."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")

        guard.mark_verified("session-a")

        assert guard.requires_challenge("session-a") is False
        assert guard.requires_challenge("session-b") is True


class TestVerifyTurnstileToken:
    """Tests for Cloudflare siteverify integration."""

    @pytest.mark.asyncio
    async def test_valid_token(self, httpx_mock: HTTPXMock):
        """Valid token returns True."""
        httpx_mock.add_response(
            url="https://challenges.cloudflare.com/turnstile/v0/siteverify",
            json={"success": True},
        )

        result = await verify_turnstile_token("valid-token")
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_token(self, httpx_mock: HTTPXMock):
        """Invalid token returns False."""
        httpx_mock.add_response(
            url="https://challenges.cloudflare.com/turnstile/v0/siteverify",
            json={"success": False, "error-codes": ["invalid-input-response"]},
        )

        result = await verify_turnstile_token("bad-token")
        assert result is False

    @pytest.mark.asyncio
    async def test_network_error_returns_false(self, httpx_mock: HTTPXMock):
        """Network errors are caught and return False."""
        httpx_mock.add_exception(ConnectionError("DNS resolution failed"))

        result = await verify_turnstile_token("some-token")
        assert result is False
