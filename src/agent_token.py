"""Verification of Bearer agent tokens minted by Drupal's ``/api/agent-token``.

The token is an ES256 ``at+jwt`` addressed to the agent (``aud``) and minted
for a registered browser client (``azp``). Its signing keys come from Drupal's
``/.well-known/exchange-jwks.json``, which never lists the cookie key, so a
cookie JWT can't verify here even if it carried the right claims.

The key-set cache is separate from the cookie path's ``PyJWKClient`` in
``src/auth.py`` because the rules differ: PyJWKClient drops its cache on a
failed fetch and raises on an empty set, while here an empty set from a
successful fetch revokes every key, and the last good set survives a failed
fetch for up to ``KEY_SET_MAX_STALE_S``.

Create one verifier per process in the app lifespan and close it on shutdown.
Its HTTP client and in-flight fetch belong to the event loop that first uses
them, so a module-level instance would break under per-test event loops.

Nothing calls this module yet; the request path switches to it in a later PR.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jwt

from .telemetry.http import create_async_client

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from .config import Settings

logger = logging.getLogger(__name__)

KEY_SET_REFRESH_S = 300
KEY_SET_MAX_STALE_S = 2 * 60 * 60
# After a failed fetch, requests keep using the last good set instead of each
# waiting out a connect timeout against an issuer that is down.
FETCH_RETRY_FLOOR_S = 30
# A `kid` is attacker-controlled, so an unknown one forces at most one fetch
# per window. Held per verifier, so bogus kids sent at the cookie path can't
# delay a real rotation here.
UNKNOWN_KID_COOLDOWN_S = 300

MAX_LIFETIME_S = 900
MAX_IAT_SKEW_S = 60
REQUIRED_SCOPE = "agent:read"
MAX_TURN_TIMEOUT_S = 120


class InvalidAgentTokenError(Exception):
    """The token is malformed, wrongly signed, or fails a claim check."""


class AgentTokenUnavailableError(Exception):
    """No usable key set, so no token can be checked (answer 503, not 401)."""


@dataclass(frozen=True)
class AgentIdentity:
    sub: str
    scopes: frozenset[str]
    exp: int


class AgentTokenVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        jwks_url: str,
        audience: str,
        allowed_azp: frozenset[str],
        verify_tls: bool = True,
        clock: Callable[[], float] = time.time,
        cache_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._issuer = issuer
        self._jwks_url = jwks_url
        self._audience = audience
        self._allowed_azp = allowed_azp
        self._clock = clock
        # Claims are wall-clock times; cache ages use a monotonic clock so a
        # clock step can't stall refreshes.
        self._cache_clock = cache_clock
        if not verify_tls:
            # ddev serves a self-signed certificate; the cookie path makes the
            # same exception for local and docker. Signatures are still verified.
            logger.warning("Agent-token JWKS TLS verification disabled")
        self._client = create_async_client(timeout=5.0, verify=verify_tls)

        self._keys: dict[str, jwt.PyJWK] | None = None
        self._fetched_at = 0.0
        self._failed_at: float | None = None
        self._forced_at: float | None = None
        self._inflight: asyncio.Task[None] | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentTokenVerifier:
        return cls(
            issuer=settings.AGENT_TOKEN_ISSUER,
            jwks_url=settings.AGENT_TOKEN_JWKS_URL,
            audience=settings.agent_token_audience,
            allowed_azp=settings.agent_token_allowed_azp,
            verify_tls=settings.ENVIRONMENT not in ("local", "docker"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def verify(self, token: str) -> AgentIdentity:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError as e:
            raise InvalidAgentTokenError("malformed token") from e
        if header.get("alg") != "ES256":
            raise InvalidAgentTokenError("alg is not ES256")
        if header.get("typ") != "at+jwt":
            raise InvalidAgentTokenError("typ is not at+jwt")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise InvalidAgentTokenError("missing kid")

        key = await self._signing_key(kid)
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key.key,
                algorithms=["ES256"],
                issuer=self._issuer,
                audience=self._audience,
                # PyJWT's time checks share one leeway, so allowing 60s of
                # `iat` skew there would also accept expired tokens.
                options={
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                    "strict_aud": True,
                    "require": ["iss", "aud", "sub", "jti", "iat", "exp"],
                },
            )
        except jwt.InvalidTokenError as e:
            raise InvalidAgentTokenError(type(e).__name__) from e

        self._check_times(claims)
        return _identity(claims, self._allowed_azp)

    def _check_times(self, claims: dict[str, Any]) -> None:
        exp, iat = claims["exp"], claims["iat"]
        if not _is_int(exp) or not _is_int(iat):
            raise InvalidAgentTokenError("exp and iat must be integers")
        now = self._clock()
        if exp <= now:
            raise InvalidAgentTokenError("expired")
        if iat > now + MAX_IAT_SKEW_S:
            raise InvalidAgentTokenError("iat is in the future")
        if exp - iat > MAX_LIFETIME_S:
            raise InvalidAgentTokenError("lifetime exceeds 900 seconds")
        nbf = claims.get("nbf")
        if nbf is not None and (not _is_int(nbf) or nbf > now + MAX_IAT_SKEW_S):
            raise InvalidAgentTokenError("not yet valid")

    async def _signing_key(self, kid: str) -> jwt.PyJWK:
        now = self._cache_clock()
        fetched = False
        if (self._keys is None or now - self._fetched_at >= KEY_SET_REFRESH_S) and self._may_fetch(
            now
        ):
            await self._refresh()
            fetched = True

        if (
            self._keys is not None
            and kid not in self._keys
            and not fetched
            and (self._forced_at is None or now - self._forced_at >= UNKNOWN_KID_COOLDOWN_S)
            and self._may_fetch(now)
        ):
            self._forced_at = now
            await self._refresh()

        if self._keys is None or self._cache_clock() - self._fetched_at > KEY_SET_MAX_STALE_S:
            raise AgentTokenUnavailableError("no key set fetched within the staleness limit")
        key = self._keys.get(kid)
        if key is None:
            raise InvalidAgentTokenError("unknown kid")
        return key

    def _may_fetch(self, now: float) -> bool:
        return self._failed_at is None or now - self._failed_at >= FETCH_RETRY_FLOOR_S

    async def _refresh(self) -> None:
        task = self._inflight
        if task is None:
            task = asyncio.get_running_loop().create_task(self._fetch())
            self._inflight = task
            task.add_done_callback(self._clear_inflight)
        # Shielded so one cancelled request doesn't cancel the fetch the
        # others are waiting on.
        await asyncio.shield(task)

    def _clear_inflight(self, task: asyncio.Task[None]) -> None:
        if self._inflight is task:
            self._inflight = None

    async def _fetch(self) -> None:
        try:
            response = await self._client.get(self._jwks_url)
        except Exception as e:
            # Anything that stops the fetch has to land here, or the retry
            # floor never applies and every request fetches again.
            self._fetch_failed(f"{type(e).__name__}: {e}")
            return
        entries = _key_entries(response)
        if entries is None:
            self._fetch_failed(f"status {response.status_code}, or not an object with a keys array")
            return

        keys: dict[str, jwt.PyJWK] = {}
        for entry in entries:
            kid = entry.get("kid") if isinstance(entry, dict) else None
            if not isinstance(kid, str) or not kid:
                continue
            try:
                keys[kid] = jwt.PyJWK(entry, algorithm="ES256")
            except Exception:
                logger.warning("AUTH_INFRA: skipping unusable key %r in agent-token JWKS", kid)
        self._keys = keys
        self._fetched_at = self._cache_clock()
        self._failed_at = None

    def _fetch_failed(self, reason: str) -> None:
        self._failed_at = self._cache_clock()
        logger.error("AUTH_INFRA: agent-token JWKS fetch failed: %s", reason)


def _key_entries(response: httpx.Response) -> list[Any] | None:
    """The ``keys`` array of a usable response, or None for a failed fetch."""
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    keys = body.get("keys") if isinstance(body, dict) else None
    return keys if isinstance(keys, list) else None


def _identity(claims: dict[str, Any], allowed_azp: frozenset[str]) -> AgentIdentity:
    azp = claims.get("azp")
    if not isinstance(azp, str) or azp not in allowed_azp:
        raise InvalidAgentTokenError("azp is not an allowed client")
    sub, jti = claims["sub"], claims["jti"]
    if not isinstance(sub, str) or not sub or not isinstance(jti, str) or not jti:
        raise InvalidAgentTokenError("sub and jti must be non-empty strings")
    scope = claims.get("scope")
    scopes = frozenset(scope.split()) if isinstance(scope, str) else frozenset()
    if REQUIRED_SCOPE not in scopes:
        raise InvalidAgentTokenError("scope lacks agent:read")
    return AgentIdentity(sub=sub, scopes=scopes, exp=int(claims["exp"]))


def check_agent_token_config(settings: Settings) -> None:
    """Refuse to start a server whose agent-token settings can't work.

    For the app lifespan, alongside creating the verifier, rather than a
    Settings validator: the nightly CLIs also load Settings, some with
    ENVIRONMENT=production, and never verify a token.
    """
    # The agent refuses tokens with less than the turn timeout plus 30 seconds
    # left, and that has to stay under the browser's 180-second refresh point.
    if not 1 <= settings.AGENT_TURN_TIMEOUT_S <= MAX_TURN_TIMEOUT_S:
        raise RuntimeError(f"AGENT_TURN_TIMEOUT_S must be between 1 and {MAX_TURN_TIMEOUT_S}")
    if settings.ENVIRONMENT != "production":
        return
    missing = [
        name
        for name in ("AGENT_TOKEN_ISSUER", "AGENT_TOKEN_JWKS_URL")
        if not getattr(settings, name)
    ]
    if missing:
        raise RuntimeError(f"Production requires {' and '.join(missing)} to be set")
    if not settings.AGENT_TOKEN_JWKS_URL.startswith("https://"):
        raise RuntimeError("Production requires AGENT_TOKEN_JWKS_URL to use https")


def _is_int(value: object) -> bool:
    # Python's json accepts NaN and Infinity, which would pass every
    # comparison in _check_times.
    return isinstance(value, int) and not isinstance(value, bool)
