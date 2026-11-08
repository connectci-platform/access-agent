"""Resource Provider section cache.

Maintains an in-memory cache of resource groups and which documentation
sections each one has populated.  Used by the capabilities endpoint to
build RP-scoped responses.

Data source: Drupal ``/api/resource-groups`` endpoint, which exposes one
entry per ``resource_group`` node with:

- ``slug``: stable URL slug derived from the group title via
  ``Html::getClass()`` — matches the ``data-resource-context`` attribute
  set by aspTheme on the embedded QA bot div.
- ``title``: clean human-readable group title.
- ``populated_sections``: union of populated section types across all
  member CIDER variants, computed server-side.
- ``variants``: the member CIDER variants, each with its ACCESS
  ``global_resource_id``. Kept so a resource a user holds an allocation on
  maps to its group by id, never by name.

Refreshed on first access, then every ``settings.RP_CACHE_TTL_SECONDS``
(default 1800 = 30 min).
"""

import logging
import time
from dataclasses import dataclass, field

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


# All known documentation sections (matches the keys returned by Drupal).
ALL_SECTIONS = [
    "login",
    "file_transfer",
    "storage",
    "queue_specs",
    "top_software",
    "datasets",
]


@dataclass
class RPInfo:
    """Cached info about a resource group."""

    slug: str
    title: str
    populated_sections: list[str] = field(default_factory=list)
    # ACCESS global resource ids of the group's member variants.
    resource_ids: list[str] = field(default_factory=list)


class RPSectionCache:
    """In-memory cache of resource groups and their populated sections.

    Fetches from Drupal's ``/api/resource-groups`` on first access, then
    refreshes every ``settings.RP_CACHE_TTL_SECONDS``.  If the fetch
    fails, serves stale data.
    """

    def __init__(self) -> None:
        self._cache: dict[str, RPInfo] = {}
        self._last_refresh: float = 0
        self._refreshing: bool = False

    async def _fetch_resources(self) -> dict[str, RPInfo]:
        """Fetch resource groups from Drupal."""
        result: dict[str, RPInfo] = {}

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(settings.DRUPAL_RESOURCE_GROUPS_URL)
            response.raise_for_status()
            data = response.json()

        groups = data.get("groups", [])
        logger.info(f"RP cache: fetched {len(groups)} resource groups from Drupal")

        for group in groups:
            slug = group.get("slug")
            title = group.get("title", "")
            populated = group.get("populated_sections", [])

            if not slug:
                continue

            result[slug] = RPInfo(
                slug=slug,
                title=title,
                populated_sections=populated,
                resource_ids=_variant_resource_ids(group.get("variants")),
            )

        return result

    async def refresh(self) -> None:
        """Refresh the cache from Drupal. Safe to call concurrently."""
        if self._refreshing:
            return

        self._refreshing = True
        try:
            new_cache = await self._fetch_resources()
            self._cache = new_cache
            self._last_refresh = time.time()
            populated_count = sum(1 for r in new_cache.values() if r.populated_sections)
            logger.info(
                f"RP section cache refreshed: {len(new_cache)} groups, "
                f"{populated_count} with populated sections"
            )
        except Exception:
            logger.exception("RP cache refresh failed — serving stale data")
        finally:
            self._refreshing = False

    def _needs_refresh(self) -> bool:
        """Check if the cache needs a refresh."""
        if not self._cache:
            return True
        return (time.time() - self._last_refresh) > settings.RP_CACHE_TTL_SECONDS

    async def ensure_loaded(self) -> None:
        """Ensure cache is loaded, refreshing if needed."""
        if self._needs_refresh():
            await self.refresh()

    def get(self, slug: str) -> RPInfo | None:
        """Look up RP info by slug. Returns None for unknown slugs."""
        return self._cache.get(slug)

    def is_valid_slug(self, slug: str) -> bool:
        """Check if a slug is in the cache."""
        return slug in self._cache

    def list_slugs(self) -> list[str]:
        """All cached RP slugs."""
        return list(self._cache.keys())

    def list_groups(self) -> list[RPInfo]:
        """All cached RP groups."""
        return list(self._cache.values())

    def slugs_for_resource_id(self, resource_id: str) -> list[str]:
        """Slugs of every group with a member variant whose ACCESS global
        resource id is ``resource_id``. More than one means the mapping is
        ambiguous; the caller decides what to do with that."""
        return [rp.slug for rp in self._cache.values() if resource_id in rp.resource_ids]


def _variant_resource_ids(variants: object) -> list[str]:
    """The non-empty ``global_resource_id`` of each variant entry."""
    if not isinstance(variants, list):
        return []
    ids = (v.get("global_resource_id") for v in variants if isinstance(v, dict))
    return [rid for rid in ids if isinstance(rid, str) and rid]


# Singleton
_cache: RPSectionCache | None = None


def get_rp_cache() -> RPSectionCache:
    """Get the singleton RP section cache."""
    global _cache
    if _cache is None:
        _cache = RPSectionCache()
    return _cache
