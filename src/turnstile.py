"""Cloudflare Turnstile bot protection for anonymous users.

Tracks per-session query counts and verification status. Authenticated
users (Bearer agent token) bypass Turnstile entirely. When Turnstile is not
configured (TURNSTILE_SECRET_KEY empty), all checks are skipped.

See: access-qa-planning/turnstile-bot-protection-spec.md
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field

import httpx
from starlette.responses import JSONResponse

from .config import settings

logger = logging.getLogger(__name__)

CLOUDFLARE_SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


@dataclass
class SessionState:
    """Tracks Turnstile state for a single anonymous session."""

    query_count: int = 0
    verified: bool = False
    verified_at: float = 0.0
    last_seen: float = field(default_factory=time.time)


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
        session = self._sessions.get(session_id)
        if session is None:
            self._evict_expired()
            session = self._sessions[session_id] = SessionState()
        session.last_seen = time.time()
        return session

    def _evict_expired(self) -> None:
        """Remove sessions idle for longer than TURNSTILE_SESSION_TTL. Runs at
        most once per EVICTION_INTERVAL or when the session count reaches
        MAX_SESSIONS.

        Idle time is the test because it can't reset a count someone is still
        using: every request, challenged or not, refreshes last_seen. The TTL is
        the window a verified session gets, so an hour of silence (by default)
        earns the same fresh allowance as an expired verification.

        Thread ids are free to mint, so a full table of active sessions drops
        its oldest as well. A batch, so that the next new session doesn't sweep
        the whole table again.
        """
        now = time.time()
        if now - self._last_eviction < EVICTION_INTERVAL and len(self._sessions) < MAX_SESSIONS:
            return

        ttl = settings.TURNSTILE_SESSION_TTL
        stale = [sid for sid, s in self._sessions.items() if now - s.last_seen > ttl]
        for sid in stale:
            del self._sessions[sid]
        if len(self._sessions) >= MAX_SESSIONS:
            excess = len(self._sessions) - MAX_SESSIONS + 1 + MAX_SESSIONS // 10
            oldest = heapq.nsmallest(
                excess, self._sessions, key=lambda sid: self._sessions[sid].last_seen
            )
            for sid in oldest:
                del self._sessions[sid]
            logger.warning("Turnstile session table full, dropped the %d oldest", len(oldest))
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
        """Increment the query count for a session. Use charge_query, not this."""
        session = self._get_session(session_id)
        session.query_count += 1

    def refund_query(self, session_id: str) -> None:
        """Give one query back, never below zero. A session evicted since the
        charge stays gone."""
        session = self._sessions.get(session_id)
        if session is not None and session.query_count > 0:
            session.query_count -= 1

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


class QueryCharge:
    """One question counted against an anonymous session, refundable once."""

    def __init__(self, session_id: str | None) -> None:
        self._session_id = session_id

    def refund(self) -> None:
        """Give the question back because the agent failed to answer it. Safe to
        call twice, and never raises into the stream."""
        session_id, self._session_id = self._session_id, None
        if session_id is None:
            return
        try:
            get_turnstile_guard().refund_query(session_id)
        except Exception:
            logger.exception("Turnstile query refund failed")


NOT_CHARGED = QueryCharge(None)


def charge_query(acting_user: str | None, session_id: str) -> QueryCharge:
    """Count a question for every route that runs the agent.

    Call it once the run lock is held. Counting up front means leaving early
    (Stop, or a disconnect) still uses the question, so the caller refunds only
    when the agent failed. The lock also makes this exact: a second request on
    the session gets a 409 and is never charged. A failed count lets the
    question through, because the limit must not take the agent down.
    """
    if acting_user:
        return NOT_CHARGED
    try:
        get_turnstile_guard().record_query(session_id)
    except Exception:
        logger.exception("Turnstile query count failed")
        return NOT_CHARGED
    return QueryCharge(session_id)


def turnstile_challenge_response() -> JSONResponse:
    """Build the JSON response that tells the frontend to show the Turnstile widget.

    Single-sourced here so every caller (the /query widget route and the
    thread/run protocol route) returns an identical challenge shape.
    """
    return JSONResponse(
        content={
            "requires_turnstile": True,
            "site_key": settings.TURNSTILE_SITE_KEY,
        }
    )


async def check_turnstile(
    acting_user: str | None,
    session_id: str,
    token: str | None,
) -> JSONResponse | None:
    """Check Turnstile for an anonymous caller. Returns a challenge response, or
    None to proceed.

    ``session_id`` is the key the guard tracks free-query counts and
    verification state under. Authed callers (``acting_user`` truthy) always
    return None immediately — Turnstile exists to gate anonymous traffic only.
    """
    if acting_user:
        return None

    guard = get_turnstile_guard()
    if not guard.requires_challenge(session_id):
        return None

    # Session needs verification — check if a token was provided
    if token and await verify_turnstile_token(token):
        guard.mark_verified(session_id)
        return None

    return turnstile_challenge_response()
