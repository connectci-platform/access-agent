"""The allocated-resources profile for a signed-in run, built server-side.

The browser sends nothing for this. For a run with an acting user, the agent
calls the allocations MCP tool ``get_my_rp_accounts`` as that user, maps each
account to an ``AllocatedResource``, and caches the result per user for a few
minutes so a thread doesn't call the tool on every message.

``None`` and ``[]`` mean different things downstream (see
``render_allocated_resources``): ``[]`` renders "Allocated resources: none".
So only a lookup that succeeded and found no accounts yields ``[]``; anything
that couldn't tell (an error, a timeout, ``state: "syncing"``, a malformed
response) yields no profile at all, which is exactly a run without one.

``rp_slug`` comes only from the resource-groups vocabulary, by the account's
ACCESS resource id. It is never derived from the display name or the resource
id itself: normalizing ``delta-gpu.ncsa.access-ci.org`` gives a slug UKY
rejects (the ``_normalize_rp_name`` trap profile.py warns about).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from ..services.rp_cache import get_rp_cache
from ..tools.mcp_client import MCPClient
from .profile import NAME_RE, RESOURCE_ID_RE, RP_SLUG_RE, AllocatedResource, UserProfile
from .tools.access_documents import _normalize_rp_name

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

ALLOCATIONS_SERVER = "allocations"
ACCOUNTS_TOOL = "get_my_rp_accounts"

# Runs before the turn's own timer starts, so it never eats the turn budget,
# but it does delay the first event. Keep it short.
LOOKUP_TIMEOUT_S = 3.0
SUCCESS_TTL_S = 300.0
# A failure is cached too, briefly, so an allocations outage costs one timeout
# per user per window rather than one per message.
FAILURE_TTL_S = 30.0
MAX_CACHED_USERS = 1024
MAX_RESOURCES = 32  # UserProfile.allocated_resources max_length

# States from Drupal's /api/1.0/rp-accounts that say what the user holds.
# "syncing" (a cold first lookup) and anything else say nothing.
_STATES_WITH_ROWS = frozenset({"rows_fresh", "rows_stale"})
_STATE_NO_ROWS = "no_rows_fresh"

_NAME_MAX = 64
_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9 .\-_/()]+")
_NAME_RE = re.compile(NAME_RE)
_RESOURCE_ID_RE = re.compile(RESOURCE_ID_RE)
_RP_SLUG_RE = re.compile(RP_SLUG_RE)
_RESOURCE_ID_MAX = 128
_RP_SLUG_MAX = 32

_clock = time.monotonic


def _clean_name(raw: object) -> str | None:
    """The display name reduced to what ``AllocatedResource.name`` accepts."""
    if not isinstance(raw, str):
        return None
    name = " ".join(_NAME_DISALLOWED.sub(" ", raw).split())
    name = name.lstrip(" .-_/()")[:_NAME_MAX].rstrip()
    return name if _NAME_RE.match(name) else None


def _clean_resource_id(raw: object) -> str | None:
    if isinstance(raw, str) and len(raw) <= _RESOURCE_ID_MAX and _RESOURCE_ID_RE.match(raw):
        return raw
    return None


def _slug_for(resource_id: str | None, slugs_for: Callable[[str], list[str]]) -> str | None:
    """The doc-search slug of the one group holding ``resource_id``, if any.

    The group slug is normalized the way ``search_access_documents`` normalizes
    every ``rp_name`` it sends (Drupal's ``bridges-2`` goes out as
    ``bridges2``), so the prompt names what the search actually uses.
    """
    if resource_id is None:
        return None
    groups = slugs_for(resource_id)
    if len(groups) != 1:
        return None
    slug = _normalize_rp_name(groups[0])
    if len(slug) <= _RP_SLUG_MAX and _RP_SLUG_RE.match(slug):
        return slug
    return None


def _resource_from_account(
    account: object, slugs_for: Callable[[str], list[str]]
) -> AllocatedResource | None:
    if not isinstance(account, dict):
        return None
    name = _clean_name(account.get("rp_display_name"))
    if name is None:
        return None
    resource_id = _clean_resource_id(account.get("resource_id"))
    # Every field is already checked against its pattern and length, so this
    # can't raise; a row that couldn't be made valid was dropped above.
    return AllocatedResource(
        name=name, rp_slug=_slug_for(resource_id, slugs_for), resource_id=resource_id
    )


def resources_from_accounts(
    data: Any, slugs_for: Callable[[str], list[str]]
) -> list[AllocatedResource] | None:
    """Map a ``get_my_rp_accounts`` response to allocated resources.

    Returns ``[]`` only for a fresh, empty account list, and ``None`` when the
    response can't say what the user holds. Rows that don't validate are
    dropped; if none survive, that is ``None`` too, not ``[]``.
    """
    if not isinstance(data, dict):
        return None
    state = data.get("state")
    accounts = data.get("accounts")
    if not isinstance(accounts, list):
        return None
    if not accounts:
        return [] if state == _STATE_NO_ROWS else None
    if state not in _STATES_WITH_ROWS:
        return None
    resources = [r for a in accounts if (r := _resource_from_account(a, slugs_for)) is not None]
    return resources[:MAX_RESOURCES] or None


async def _lookup(acting_user: str) -> UserProfile | None:
    rp_cache = get_rp_cache()
    result, _ = await asyncio.gather(
        MCPClient().call_tool(ALLOCATIONS_SERVER, ACCOUNTS_TOOL, {}, acting_user=acting_user),
        rp_cache.ensure_loaded(),
    )
    if not result.success:
        logger.info(
            "Allocations profile unavailable: lookup failed (timed_out=%s)", result.timed_out
        )
        return None
    if not rp_cache.list_slugs():
        logger.info("Allocations profile unavailable: resource-group vocabulary not loaded")
        return None
    resources = resources_from_accounts(result.data, rp_cache.slugs_for_resource_id)
    if resources is None:
        logger.info("Allocations profile unavailable: response did not list accounts")
        return None
    return UserProfile(allocated_resources=resources)


class _ProfileCache:
    """Per-user, short-lived, bounded. Keyed by the verified acting user only."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[float, UserProfile | None]] = OrderedDict()

    def get(self, acting_user: str) -> tuple[bool, UserProfile | None]:
        entry = self._entries.get(acting_user)
        if entry is None:
            return False, None
        expires_at, profile = entry
        if _clock() >= expires_at:
            del self._entries[acting_user]
            return False, None
        return True, profile

    def put(self, acting_user: str, profile: UserProfile | None, ttl: float) -> None:
        self._entries[acting_user] = (_clock() + ttl, profile)
        self._entries.move_to_end(acting_user)
        while len(self._entries) > MAX_CACHED_USERS:
            self._entries.popitem(last=False)

    def reset(self) -> None:
        self._entries.clear()


_cache = _ProfileCache()


async def allocated_profile_for(acting_user: str | None) -> UserProfile | None:
    """The profile for a run acting for ``acting_user``, or ``None``.

    Anonymous runs make no lookup. Never raises, except to pass on the run's
    own cancellation; every failure is a run without a profile.
    """
    if acting_user is None:
        return None
    hit, profile = _cache.get(acting_user)
    if hit:
        return profile
    try:
        async with asyncio.timeout(LOOKUP_TIMEOUT_S):
            profile = await _lookup(acting_user)
    except TimeoutError:
        logger.info("Allocations profile unavailable: lookup timed out")
        profile = None
    except Exception as exc:
        logger.warning("Allocations profile unavailable: %s", type(exc).__name__)
        profile = None
    _cache.put(acting_user, profile, SUCCESS_TTL_S if profile is not None else FAILURE_TTL_S)
    return profile


def _reset_cache_for_test() -> None:
    _cache.reset()
