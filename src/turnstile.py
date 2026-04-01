"""Cloudflare Turnstile bot protection for anonymous users.

Tracks per-session query counts and verification status. Authenticated
users (JWT cookie) bypass Turnstile entirely. When Turnstile is not
configured (TURNSTILE_SECRET_KEY empty), all checks are skipped.

See: access-qa-planning/turnstile-bot-protection-spec.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from .config import settings

logger = logging.getLogger(__name__)

CLOUDFLARE_SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


@dataclass
class SessionState:
    """Tracks Turnstile state for a single anonymous session."""

    query_count: int = 0
    verified: bool = False
    verified_at: float = 0.0


MAX_SESSIONS = 10_000
EVICTION_INTERVAL = 300  # seconds between cleanup sweeps


class TurnstileGuard:
    """Gate that decides whether an anonymous session needs Turnstile verification.

    Sessions are tracked in-memory by session_id with periodic eviction of
    expired entries. This is sufficient for single-instance deployment; can
    be migrated to Redis if scaling requires shared state.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._last_eviction: float = time.time()

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._evict_expired()
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def _evict_expired(self) -> None:
        """Remove sessions whose verification has expired and that have
        exhausted their free queries. Runs at most once per EVICTION_INTERVAL
        or when the session count exceeds MAX_SESSIONS."""
        now = time.time()
        if now - self._last_eviction < EVICTION_INTERVAL and len(self._sessions) < MAX_SESSIONS:
            return

        ttl = settings.TURNSTILE_SESSION_TTL
        stale = [
            sid
            for sid, s in self._sessions.items()
            if (s.verified and now - s.verified_at > ttl)
            or (
                not s.verified
                and s.query_count >= settings.TURNSTILE_FREE_QUERIES
                and now - s.verified_at > ttl
            )
        ]
        for sid in stale:
            del self._sessions[sid]
        if stale:
            logger.info(
                "Evicted %d expired Turnstile sessions (%d remaining)",
                len(stale),
                len(self._sessions),
            )
        self._last_eviction = now

    def requires_challenge(self, session_id: str) -> bool:
        """Check whether this session must complete a Turnstile challenge.

        Returns False (allow) if:
        - Turnstile is disabled
        - Session is already verified and TTL has not expired
        - Mode is deferred and free queries remain
        """
        if not settings.turnstile_enabled:
            return False

        session = self._get_session(session_id)

        # Already verified — check TTL
        if session.verified:
            elapsed = time.time() - session.verified_at
            if elapsed < settings.TURNSTILE_SESSION_TTL:
                return False
            # Verification expired — reset verified flag AND query count
            # so the user gets a fresh grace period before re-challenge
            session.verified = False
            session.verified_at = 0.0
            session.query_count = 0
            logger.info("Turnstile verification expired for session %s (counter reset)", session_id)

        # Deferred mode: allow free queries; immediate mode: challenge immediately
        return not (
            settings.TURNSTILE_MODE == "deferred"
            and session.query_count < settings.TURNSTILE_FREE_QUERIES
        )

    def record_query(self, session_id: str) -> None:
        """Increment the query count for a session (called after a successful query)."""
        session = self._get_session(session_id)
        session.query_count += 1

    def mark_verified(self, session_id: str) -> None:
        """Mark a session as Turnstile-verified."""
        session = self._get_session(session_id)
        session.verified = True
        session.verified_at = time.time()
        logger.info("Session %s passed Turnstile verification", session_id)


async def verify_turnstile_token(token: str) -> bool:
    """Verify a Turnstile token with Cloudflare's siteverify endpoint.

    Args:
        token: The Turnstile response token from the frontend widget.

    Returns:
        True if the token is valid.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                CLOUDFLARE_SITEVERIFY_URL,
                data={
                    "secret": settings.TURNSTILE_SECRET_KEY,
                    "response": token,
                },
            )
            result = response.json()
            success = bool(result.get("success", False))
            if not success:
                error_codes = result.get("error-codes", [])
                logger.warning("Turnstile verification failed: %s", error_codes)
            return success
    except Exception:
        logger.exception("Turnstile verification request failed")
        return False


# Global instance
_guard: TurnstileGuard | None = None


def get_turnstile_guard() -> TurnstileGuard:
    """Get the global TurnstileGuard instance."""
    global _guard
    if _guard is None:
        _guard = TurnstileGuard()
    return _guard
