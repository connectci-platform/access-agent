# Cross-surface handoff token store: single-use, short-TTL tokens that carry a
# (thread_id, owner_hash) pair from the mint endpoint (Task 1.2) to the exchange
# endpoint (Task 1.2), so a widget session can hand a conversation off to the
# full-screen client without ever putting the session_id itself in a URL.
#
# Single-use is enforced via an ATOMIC delete-on-fetch, never read-then-delete:
# in-memory, dict.pop(token, None) is atomic under the GIL; Redis uses GETDEL
# (a Redis-server command since 6.2, exposed by redis-py since 4.1 — both well
# below the >=5.0.0 pin). Two-step read-then-delete would allow a race where two
# concurrent exchanges both observe the token before either deletes it.
import secrets
import time
from typing import cast

import redis

from src.config import settings

_SEPARATOR = "\x00"

# Handoff token lifetime. Long enough to open a tab and exchange, short enough
# to bound a leaked-URL replay window. The mint endpoint reports this exact
# value to the client as `expires_in`, so keep the two in sync via this const.
DEFAULT_TTL_SECONDS = 60


class _InMemoryStore:
    """Single-process dev/test fallback. Not safe across multiple workers."""

    def __init__(self) -> None:
        self._tokens: dict[str, tuple[str, str, float]] = {}

    def mint(self, thread_id: str, owner_hash: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(24)
        expiry = time.monotonic() + ttl_seconds
        self._tokens[token] = (thread_id, owner_hash, expiry)
        return token

    def exchange(self, token: str) -> tuple[str, str] | None:
        # pop is atomic under the GIL: two concurrent exchanges of the same
        # token can't both succeed.
        entry = self._tokens.pop(token, None)
        if entry is None:
            return None
        thread_id, owner_hash, expiry = entry
        if time.monotonic() > expiry:
            return None
        return thread_id, owner_hash

    def reset(self) -> None:
        self._tokens.clear()


class _RedisStore:
    """Redis-backed store for multi-worker/multi-replica deployments."""

    def __init__(self, redis_url: str) -> None:
        self._client: redis.Redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def mint(self, thread_id: str, owner_hash: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(24)
        value = f"{thread_id}{_SEPARATOR}{owner_hash}"
        self._client.set(token, value, ex=ttl_seconds, nx=True)
        return token

    def exchange(self, token: str) -> tuple[str, str] | None:
        # GETDEL is atomic single-use fetch-and-delete (Redis-server >= 6.2) —
        # no separate read-then-delete.
        raw = self._client.getdel(token)
        if raw is None:
            return None
        # decode_responses=True guarantees str at runtime; the stub's return
        # type is still the generic AnyStr default, so narrow explicitly.
        value = cast("str", raw)
        thread_id, _, owner_hash = value.partition(_SEPARATOR)
        return thread_id, owner_hash


_in_memory_store = _InMemoryStore()


def _get_store() -> "_InMemoryStore | _RedisStore":
    """Redis-backed when REDIS_URL is set, else the in-memory fallback.

    Client construction is guarded on REDIS_URL (not the module-level import,
    which is unconditional — the `redis` dep is already present).
    """
    if settings.REDIS_URL:
        return _RedisStore(settings.REDIS_URL)
    return _in_memory_store


def mint_handoff_token(
    thread_id: str, owner_hash: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:
    """Mint a single-use, short-TTL token binding thread_id to owner_hash.

    owner_hash is always the minting caller's _hash_user(cookie_user) — never
    None, since mint is authed-only (the endpoint rejects anon threads before
    calling this).
    """
    return _get_store().mint(thread_id, owner_hash, ttl_seconds)


def exchange_handoff_token(token: str) -> tuple[str, str] | None:
    """Atomically fetch-and-delete a token, returning (thread_id, owner_hash).

    Returns None if the token is absent, expired, or already used. Single-use
    is enforced by the atomic delete-on-fetch, not by a separate check.
    """
    return _get_store().exchange(token)


def _reset_store_for_test() -> None:
    """Clear the in-memory store between tests."""
    _in_memory_store.reset()
