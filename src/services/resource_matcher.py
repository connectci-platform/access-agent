"""Match resource-group mentions in turn text.

Pure module: no cache, no I/O. The caller supplies the group list (slug +
title from RPSectionCache); ``match_resources`` returns the slugs whose name
appears in the text. Matching is case-insensitive with light normalization —
hyphens/spaces between name parts are interchangeable and optional, so
"bridges2", "bridges 2", and "Bridges-2" all match the "Bridges-2" group.
Word boundaries prevent substring hits ("anvilteam.org" does not match Anvil).
"""

import re
from collections.abc import Iterable

from .rp_cache import RPInfo, get_rp_cache

# Letter-runs and digit-runs of a name; separators between them are dropped
# and re-allowed as optional [-\s] when matching.
_PART_RE = re.compile(r"[a-z]+|\d+", re.IGNORECASE)


def _name_pattern(name: str) -> re.Pattern[str] | None:
    """Compile a name like 'Bridges-2' into r'\\bBridges[-\\s]?2\\b'."""
    parts = _PART_RE.findall(name)
    if not parts:
        return None
    body = r"[-\s]?".join(re.escape(p) for p in parts)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def match_resources(text: str, groups: Iterable[RPInfo]) -> list[str]:
    """Slugs of the resource groups mentioned in ``text``, sorted, deduped."""
    if not text:
        return []
    matched: set[str] = set()
    for group in groups:
        for name in (group.title, group.slug):
            pattern = _name_pattern(name)
            if pattern is not None and pattern.search(text):
                matched.add(group.slug)
                break
    return sorted(matched)


async def resources_for_turn(query_text: str, answer: str) -> list[str]:
    """Resource slugs mentioned in a turn (question + answer).

    Best-effort: cache failures degrade to an empty list — this feeds the
    turn report and must never raise or block the response path.
    """
    try:
        cache = get_rp_cache()
        await cache.ensure_loaded()
        groups = cache.list_groups()
    except Exception:
        return []
    return match_resources(f"{query_text}\n{answer}", groups)
