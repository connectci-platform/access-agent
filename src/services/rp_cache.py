"""Resource Provider section cache.

Maintains an in-memory cache of which documentation sections each RP has
populated.  Used by the capabilities endpoint to build RP-scoped responses.

Data source: Drupal ``/api/resources`` list endpoint + ``/api/resources/{id}``
detail endpoint.  The list provides resource IDs and titles; the detail
endpoint provides per-section fields.  The cache checks which fields have
content to determine populated sections.

Refreshed on first access, then every ``settings.RP_CACHE_TTL_SECONDS`` (default 1800 = 30 min).
"""

import logging
import time
from dataclasses import dataclass, field

import httpx

from ..config import settings

logger = logging.getLogger(__name__)



# Mapping from Drupal API field names to our section identifiers
DRUPAL_FIELD_TO_SECTION = {
    "ssh_logins": "login",
    "file_transfer": "file_transfer",
    "storage": "storage",
    "queue_specs": "queue_specs",
    "top_software": "top_software",
    "datasets": "datasets",
}

# All known documentation sections
ALL_SECTIONS = list(DRUPAL_FIELD_TO_SECTION.values())



@dataclass
class RPInfo:
    """Cached info about a resource provider."""

    slug: str
    title: str
    populated_sections: list[str] = field(default_factory=list)


def _extract_slug(global_resource_id: str) -> str | None:
    """Extract a short slug from a global_resource_id.

    e.g. 'delta-cpu.ncsa.access-ci.org' -> 'delta'
         'anvil.purdue.access-ci.org' -> 'anvil'

    Takes the first segment before the dot, then strips common suffixes
    like -cpu, -gpu, -storage.
    """
    if not global_resource_id:
        return None
    host = global_resource_id.split(".")[0]
    # Strip common suffixes to get the base resource name
    for suffix in ("-cpu", "-gpu", "-storage"):
        if host.endswith(suffix):
            host = host[: -len(suffix)]
            break
    return host


def _has_content(value: object) -> bool:
    """Check if a Drupal API field value has content."""
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        return len(value.strip()) > 0
    return bool(value)


class RPSectionCache:
    """In-memory cache of RP documentation sections.

    Fetches from Drupal's ``/api/resources`` on first access, then refreshes
    every ``settings.RP_CACHE_TTL_SECONDS``.  If the fetch fails, serves stale data.
    """

    def __init__(self) -> None:
        self._cache: dict[str, RPInfo] = {}
        self._last_refresh: float = 0
        self._refreshing: bool = False

    async def _fetch_resources(self) -> dict[str, RPInfo]:
        """Fetch resource list from Drupal and build section cache."""
        result: dict[str, RPInfo] = {}

        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: fetch resource list
            list_response = await client.get(settings.DRUPAL_RESOURCES_URL)
            list_response.raise_for_status()
            list_data = list_response.json()
            resources = list_data.get("resources", [])

            logger.info(f"RP cache: fetched {len(resources)} resources from Drupal")

            # Step 2: for each resource, extract slug and fetch detail
            for resource in resources:
                resource_id = resource.get("resource_id")
                global_rid = resource.get("global_resource_id", "")
                title = resource.get("title", "")
                slug = _extract_slug(global_rid)

                if not slug or not resource_id:
                    continue

                # Skip if we already have this slug (first one wins —
                # e.g. delta-cpu before delta-gpu)
                if slug in result:
                    continue

                try:
                    detail_response = await client.get(
                        f"{settings.DRUPAL_RESOURCES_URL}/{resource_id}"
                    )
                    detail_response.raise_for_status()
                    detail = detail_response.json()

                    # Check which section fields have content
                    populated = []
                    for drupal_field, section_id in DRUPAL_FIELD_TO_SECTION.items():
                        if _has_content(detail.get(drupal_field)):
                            populated.append(section_id)

                    # Clean up title — strip parenthetical like "NCSA Delta CPU (Delta CPU)"
                    clean_title = title
                    if "(" in clean_title:
                        clean_title = clean_title.split("(")[0].strip()
                    # Strip org prefix like "NCSA " or "Purdue "
                    # Use the short slug capitalized as fallback
                    if not populated:
                        # No sections populated — still cache the resource
                        # so we know the slug is valid (support + analytics
                        # always show)
                        pass

                    result[slug] = RPInfo(
                        slug=slug,
                        title=clean_title,
                        populated_sections=populated,
                    )

                except httpx.HTTPError as e:
                    logger.warning(f"RP cache: failed to fetch detail for {slug} ({resource_id}): {e}")
                    continue

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
                f"RP section cache refreshed: {len(new_cache)} resources, "
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


# Singleton
_cache: RPSectionCache | None = None


def get_rp_cache() -> RPSectionCache:
    """Get the singleton RP section cache."""
    global _cache
    if _cache is None:
        _cache = RPSectionCache()
    return _cache
