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

    def test_disabled_when_no_secret_key(self, monkeypatch):
        """When TURNSTILE_SECRET_KEY is empty, turnstile_enabled is False."""
        # Isolate from any local .env TURNSTILE_SECRET_KEY the conftest may load.
        monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "")
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

    def test_expired_verification_resets_query_count(self, guard, monkeypatch):
        """When verification expires, query_count resets so user gets fresh free queries."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "deferred")
        monkeypatch.setattr(settings, "TURNSTILE_FREE_QUERIES", 3)
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 1)

        session_id = "test-session-expire"

        # Use up free queries and verify
        for _ in range(3):
            guard.record_query(session_id)
        guard.mark_verified(session_id)
        assert guard.requires_challenge(session_id) is False

        # Expire the verification
        guard._sessions[session_id].verified_at = time.time() - 2

        # Should NOT immediately challenge — counter should reset
        assert guard.requires_challenge(session_id) is False
        # And we get 3 more free queries
        for _ in range(3):
            guard.record_query(session_id)
        assert guard.requires_challenge(session_id) is True

    def test_eviction_removes_expired_sessions(self, guard, monkeypatch):
        """Expired sessions are cleaned up during eviction."""
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 1)

        # Create a session and verify it
        guard.mark_verified("old-session")
        guard._sessions["old-session"].verified_at = time.time() - 100

        # Force eviction by setting last_eviction in the past
        guard._last_eviction = time.time() - 600

        # Creating a new session triggers eviction
        guard._get_session("new-session")

        assert "old-session" not in guard._sessions
        assert "new-session" in guard._sessions

    def test_eviction_respects_max_sessions(self, guard, monkeypatch):
        """Eviction triggers when MAX_SESSIONS is exceeded."""
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 1)

        # Fill up with expired sessions
        for i in range(50):
            sid = f"bulk-{i}"
            guard._sessions[sid] = guard._get_session(sid)
            guard._sessions[sid].verified = True
            guard._sessions[sid].verified_at = time.time() - 100

        # Set last_eviction recent so only MAX_SESSIONS triggers it
        guard._last_eviction = time.time()

        # Pretend we're over the limit
        monkeypatch.setattr("src.turnstile.MAX_SESSIONS", 30)

        # This should trigger eviction
        guard._get_session("trigger-session")
        assert len(guard._sessions) < 52  # Some should be evicted


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
