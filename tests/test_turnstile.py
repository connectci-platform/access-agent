"""Tests for Turnstile bot protection."""

import time
from unittest.mock import MagicMock

import pytest
from pytest_httpx import HTTPXMock

from src.config import settings
from src.turnstile import (
    NOT_CHARGED,
    TurnstileGuard,
    charge_query,
    check_turnstile,
    get_turnstile_guard,
    verify_turnstile_token,
)


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
        guard._sessions["old-session"].last_seen = time.time() - 100

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
            guard._sessions[sid].last_seen = time.time() - 100

        # Set last_eviction recent so only MAX_SESSIONS triggers it
        guard._last_eviction = time.time()

        # Pretend we're over the limit
        monkeypatch.setattr("src.turnstile.MAX_SESSIONS", 30)

        # This should trigger eviction
        guard._get_session("trigger-session")
        assert len(guard._sessions) < 52  # Some should be evicted

    def _exhaust(self, guard, monkeypatch, session_id):
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "deferred")
        monkeypatch.setattr(settings, "TURNSTILE_FREE_QUERIES", 1)
        guard.record_query(session_id)
        assert guard.requires_challenge(session_id) is True

    def test_eviction_keeps_an_exhausted_session_that_is_still_active(self, guard, monkeypatch):
        """D8-2978: a sweep used to reset the count, because an unverified
        session's verified_at is always 0."""
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 3600)
        self._exhaust(guard, monkeypatch, "exhausted")

        guard._last_eviction = time.time() - 600
        guard._get_session("new-session")

        assert guard.requires_challenge("exhausted") is True

    def test_eviction_expires_an_exhausted_session_after_the_idle_ttl(self, guard, monkeypatch):
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 3600)
        self._exhaust(guard, monkeypatch, "idle")
        guard._sessions["idle"].last_seen = time.time() - 3601

        guard._last_eviction = time.time() - 600
        guard._get_session("new-session")

        assert "idle" not in guard._sessions

    def test_a_challenged_request_keeps_the_session_alive(self, guard, monkeypatch):
        monkeypatch.setattr(settings, "TURNSTILE_SESSION_TTL", 3600)
        self._exhaust(guard, monkeypatch, "retrying")
        guard._sessions["retrying"].last_seen = time.time() - 3601

        assert guard.requires_challenge("retrying") is True

        guard._last_eviction = time.time() - 600
        guard._get_session("new-session")
        assert "retrying" in guard._sessions

    def test_session_count_stays_bounded_when_every_session_is_active(self, guard, monkeypatch):
        """Thread ids are free to mint, so the TTL alone can't bound memory."""
        monkeypatch.setattr("src.turnstile.MAX_SESSIONS", 20)
        guard._last_eviction = time.time()
        for i in range(200):
            guard._get_session(f"flood-{i}")
            guard._sessions[f"flood-{i}"].last_seen = 1_000_000_000_000.0 + i

        assert len(guard._sessions) <= 20
        assert "flood-199" in guard._sessions
        assert "flood-0" not in guard._sessions

    def test_refund_gives_a_question_back(self, guard, monkeypatch):
        self._exhaust(guard, monkeypatch, "refunded")

        guard.refund_query("refunded")

        assert guard.requires_challenge("refunded") is False

    def test_refund_stops_at_zero(self, guard):
        guard._get_session("empty")

        guard.refund_query("empty")

        assert guard._sessions["empty"].query_count == 0

    def test_refund_of_an_evicted_session_does_not_bring_it_back(self, guard):
        guard.refund_query("gone")

        assert "gone" not in guard._sessions


@pytest.fixture(autouse=True)
def _reset_global_guard():
    """check_turnstile goes through the module-level singleton (get_turnstile_guard),
    unlike TestTurnstileGuard above which constructs its own instance — reset it
    so state from one test can't bleed into another."""
    import src.turnstile as ts

    ts._guard = None
    yield
    ts._guard = None


class TestChargeQuery:
    """charge_query counts an anonymous question; its charge refunds it once."""

    @pytest.fixture(autouse=True)
    def _fresh_guard(self, monkeypatch):
        monkeypatch.setattr("src.turnstile._guard", None)

    @staticmethod
    def _count(session_id: str) -> int:
        return get_turnstile_guard()._sessions[session_id].query_count

    def test_anonymous_question_is_counted_immediately(self):
        charge_query(None, "sess")

        assert self._count("sess") == 1

    def test_signed_in_question_is_not_counted(self):
        charge_query("user@x", "sess").refund()

        assert "sess" not in get_turnstile_guard()._sessions

    def test_refund_only_happens_once(self):
        charge_query(None, "sess")
        charge = charge_query(None, "sess")

        charge.refund()
        charge.refund()

        assert self._count("sess") == 1

    def test_failed_count_fails_open_and_has_nothing_to_refund(self, monkeypatch):
        guard = get_turnstile_guard()
        monkeypatch.setattr(guard, "record_query", MagicMock(side_effect=RuntimeError("full")))
        refund = MagicMock()
        monkeypatch.setattr(guard, "refund_query", refund)

        charge_query(None, "sess").refund()

        refund.assert_not_called()

    def test_failed_refund_does_not_raise(self, monkeypatch):
        charge = charge_query(None, "sess")
        monkeypatch.setattr(
            get_turnstile_guard(), "refund_query", MagicMock(side_effect=RuntimeError("gone"))
        )

        charge.refund()

    def test_not_charged_refunds_nothing(self):
        charge_query(None, "sess")

        NOT_CHARGED.refund()

        assert self._count("sess") == 1


class TestCheckTurnstile:
    """Tests for the check_turnstile gate function (moved here from
    src/api/routes.py so both /query and the thread/run endpoint share it)."""

    @pytest.mark.asyncio
    async def test_authed_caller_always_skips(self, _enable_turnstile, monkeypatch):
        """An acting_user short-circuits before the guard is even consulted,
        even in immediate mode."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")
        result = await check_turnstile("someone@access-ci.org", "sess-1", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_anon_caller_not_yet_requiring_challenge_proceeds(
        self, _enable_turnstile, monkeypatch
    ):
        """Deferred mode, free queries remain -> proceeds without a token."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "deferred")
        monkeypatch.setattr(settings, "TURNSTILE_FREE_QUERIES", 3)
        result = await check_turnstile(None, "sess-2", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_anon_caller_requiring_challenge_with_no_token_is_challenged(
        self, _enable_turnstile, monkeypatch
    ):
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")
        result = await check_turnstile(None, "sess-3", None)
        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_anon_caller_with_valid_token_is_verified_and_proceeds(
        self, _enable_turnstile, monkeypatch, httpx_mock: HTTPXMock
    ):
        """A valid turnstile_token marks the session verified and proceeds —
        the branch that was previously only exercised at /query's integration
        level (never directly)."""
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")
        httpx_mock.add_response(
            url="https://challenges.cloudflare.com/turnstile/v0/siteverify",
            json={"success": True},
        )

        result = await check_turnstile(None, "sess-4", "a-valid-token")
        assert result is None
        assert get_turnstile_guard().requires_challenge("sess-4") is False

    @pytest.mark.asyncio
    async def test_anon_caller_with_invalid_token_is_still_challenged(
        self, _enable_turnstile, monkeypatch, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate")
        httpx_mock.add_response(
            url="https://challenges.cloudflare.com/turnstile/v0/siteverify",
            json={"success": False, "error-codes": ["invalid-input-response"]},
        )

        result = await check_turnstile(None, "sess-5", "a-bad-token")
        assert result is not None
        assert result.status_code == 200


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
