"""Whether a signed-in run uses the user's allocations, and when to offer it (D8-3124).

Personalization is opt-in. A run looks up the user's allocations only when
their consent is ``allow``, or ``unset`` with Allow once chosen in this
conversation. Deny always wins, and a consent that can't be read counts as
no.

The offer is decided after the turn, from what the model did rather than from
a prompt rule: a turn that used the compute-resources or software-discovery
tools was a question the user's own resources would change. The ask is
recorded in Drupal before the offer goes out, which is what keeps the next
one 30 days away, across devices, even if the user ignores the card. If it
can't be recorded, no offer is made.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .consent import Consent, consent_for, record_consent, should_offer
from .profile_source import allocated_profile_for

if TYPE_CHECKING:
    from datetime import datetime

    from .profile import UserProfile

OFFER_SERVERS = frozenset({"compute-resources", "software-discovery"})

# Per user, never pruned: a few bytes each, one per signed-in user who was
# ever offered personalization in this process.
_offer_locks: dict[str, asyncio.Lock] = {}


async def personalized_profile(
    user: str | None, *, once: bool
) -> tuple[Consent | None, UserProfile | None]:
    """The user's consent and, when it allows, their allocations profile."""
    if user is None:
        return None, None
    consent = await consent_for(user)
    if consent is None:
        return None, None
    if consent.state == "allow" or (consent.state == "unset" and once):
        return consent, await allocated_profile_for(user)
    return consent, None


def _used_resource_tools(final_state: dict[str, Any]) -> bool:
    for raw in final_state.get("tool_results") or []:
        result = raw if isinstance(raw, dict) else raw.model_dump()
        if result.get("server") in OFFER_SERVERS and not result.get("refused"):
            return True
    return False


async def offer_after_turn(
    user: str | None,
    consent: Consent | None,
    final_state: dict[str, Any],
    *,
    once: bool,
    now: datetime,
) -> bool:
    """Whether to show the personalization card after this turn.

    True only after the ask has been recorded.
    """
    if user is None or once or not should_offer(consent, now=now):
        return False
    if not _used_resource_tools(final_state):
        return False
    # One offer per user at a time: two tabs finishing together would both
    # see the consent read before either recorded the ask. The second reads
    # again under the lock and sees the first's ask.
    lock = _offer_locks.setdefault(user, asyncio.Lock())
    async with lock:
        if not should_offer(await consent_for(user), now=now):
            return False
        return await record_consent(user, "not_now") is not None
