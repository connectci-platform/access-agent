"""A signed-in user's consent to personalized answers (D8-3124).

Personalization is opt-in. The answer lives in the user's ACCESS profile in
Drupal (D8-3123) and is read and written through the allocations server's
unlisted tools, as the user. The model never sees those tools.

``consent_for`` returns ``None`` whenever it can't tell: no user, an error, a
timeout, a reply it can't read. Callers treat that like "not allowed and don't
ask": no profile and no offer. The state is cached per user briefly, and a
write replaces the cached entry with Drupal's answer so the change applies on
the user's next message in this process. The TTL is short because other
worker processes, and the Drupal profile form, change answers it doesn't see.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from ..tools.mcp_client import MCPClient

logger = logging.getLogger(__name__)

SERVER = "allocations"
READ_TOOL = "get_assistant_consent"
WRITE_TOOL = "set_assistant_consent"

LOOKUP_TIMEOUT_S = 3.0
# Short, because a change on the Drupal profile form never reaches this
# cache: it is the longest a turn can run on an old answer. A change through
# this agent replaces the entry at once.
SUCCESS_TTL_S = 10.0
FAILURE_TTL_S = 10.0
MAX_CACHED_USERS = 1024
# After an offer or "Not now", the next offer waits this long.
OFFER_INTERVAL = timedelta(days=30)

State = Literal["allow", "deny", "unset"]
Choice = Literal["allow", "deny", "not_now"]
_STATES = frozenset({"allow", "deny", "unset"})
_CHOICES = frozenset({"allow", "deny", "not_now"})

_clock = time.monotonic


@dataclass(frozen=True)
class Consent:
    state: State
    asked_at: datetime | None


def should_offer(consent: Consent | None, *, now: datetime) -> bool:
    """Whether the user may be asked: never answered, and not asked lately."""
    if consent is None or consent.state != "unset":
        return False
    return consent.asked_at is None or now - consent.asked_at >= OFFER_INTERVAL


def _parse(data: Any) -> Consent | None:
    if not isinstance(data, dict) or data.get("consent") not in _STATES:
        return None
    raw = data.get("asked_at")
    asked: datetime | None = None
    if raw is not None:
        if not isinstance(raw, str):
            return None
        try:
            asked = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if asked.tzinfo is None:
            asked = asked.replace(tzinfo=UTC)
    return Consent(state=data["consent"], asked_at=asked)


class _Cache:
    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[float, Consent | None]] = OrderedDict()

    def get(self, user: str) -> tuple[bool, Consent | None]:
        entry = self._entries.get(user)
        if entry is None or _clock() >= entry[0]:
            self._entries.pop(user, None)
            return False, None
        self._entries.move_to_end(user)
        return True, entry[1]

    def put(self, user: str, value: Consent | None) -> None:
        ttl = SUCCESS_TTL_S if value is not None else FAILURE_TTL_S
        self._entries[user] = (_clock() + ttl, value)
        self._entries.move_to_end(user)
        while len(self._entries) > MAX_CACHED_USERS:
            self._entries.popitem(last=False)

    def forget(self, user: str) -> None:
        self._entries.pop(user, None)

    def reset(self) -> None:
        self._entries.clear()


_cache = _Cache()


async def _call(user: str, tool: str, args: dict[str, Any]) -> Consent | None:
    try:
        async with asyncio.timeout(LOOKUP_TIMEOUT_S):
            result = await MCPClient().call_tool(SERVER, tool, args, acting_user=user)
    except TimeoutError:
        logger.info("Assistant consent unavailable: %s timed out", tool)
        return None
    except Exception as exc:
        logger.warning("Assistant consent unavailable: %s", type(exc).__name__)
        return None
    if not result.success:
        logger.info("Assistant consent unavailable: %s failed", tool)
        return None
    parsed = _parse(result.data)
    if parsed is None:
        logger.info("Assistant consent unavailable: %s reply unreadable", tool)
    return parsed


async def consent_for(user: str | None) -> Consent | None:
    """The user's consent, or ``None`` when there is no user or it can't tell.

    Never raises, except to pass on the caller's cancellation.
    """
    if user is None:
        return None
    hit, value = _cache.get(user)
    if hit:
        return value
    value = await _call(user, READ_TOOL, {})
    _cache.put(user, value)
    return value


async def record_consent(user: str, choice: str) -> Consent | None:
    """Store the user's answer. Returns the stored state, or ``None`` on failure.

    ``not_now`` records that the user was asked and leaves the state unset;
    Drupal refuses it once the user has chosen allow or deny.
    """
    if choice not in _CHOICES:
        raise ValueError(f"not a consent choice: {choice!r}")
    value = await _call(user, WRITE_TOOL, {"consent": choice})
    if value is None:
        # The old entry may now be wrong either way; read it fresh next time.
        _cache.forget(user)
    else:
        _cache.put(user, value)
    return value


def _reset_cache_for_test() -> None:
    _cache.reset()
