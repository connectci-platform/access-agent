"""Resource Provider section cache.

Maintains an in-memory cache of which documentation sections each RP has
populated.  Used by the capabilities endpoint to build RP-scoped responses.

Data source: Drupal ``/api/resources`` endpoint (future).  Until Drupal
ships ``populated_sections``, the cache is seeded with hardcoded data for
known resources.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Valid UKY RP slugs (from uky-resource-scoped-rag-spec.md)
VALID_RP_SLUGS = frozenset({
    "aces", "ama27", "anvil", "aws", "bridges2", "cloudbank", "delta",
    "deltaai", "derecho", "expanse", "fabric", "googlecloud", "granite",
    "ibmcloud", "jetstream2", "kyric", "launch", "microsoftazure",
    "neocortex", "nexus", "osn", "osg", "pnrp", "ranch", "repacss",
    "sage", "sgx3", "stampede3", "voyager",
})

# All known documentation sections
ALL_SECTIONS = [
    "login", "file_transfer", "storage", "queue_specs", "top_software", "datasets",
]


@dataclass
class RPInfo:
    """Cached info about a resource provider."""

    slug: str
    title: str
    populated_sections: list[str] = field(default_factory=list)


# Hardcoded seed data — representative resources for development/testing.
# Replace with live fetch from Drupal /api/resources once populated_sections
# is available in production.
SEED_DATA: dict[str, RPInfo] = {
    "delta": RPInfo(
        slug="delta",
        title="Delta",
        populated_sections=["login", "file_transfer", "storage", "queue_specs", "top_software", "datasets"],
    ),
    "anvil": RPInfo(
        slug="anvil",
        title="Anvil",
        populated_sections=["login", "storage", "queue_specs", "top_software"],
    ),
    "bridges2": RPInfo(
        slug="bridges2",
        title="Bridges-2",
        populated_sections=["login", "file_transfer", "storage", "queue_specs", "top_software"],
    ),
    "expanse": RPInfo(
        slug="expanse",
        title="Expanse",
        populated_sections=["login", "file_transfer", "storage", "queue_specs", "top_software"],
    ),
    "jetstream2": RPInfo(
        slug="jetstream2",
        title="Jetstream2",
        populated_sections=["login", "storage"],
    ),
    "stampede3": RPInfo(
        slug="stampede3",
        title="Stampede3",
        populated_sections=["login", "file_transfer", "storage", "queue_specs", "top_software"],
    ),
    "derecho": RPInfo(
        slug="derecho",
        title="Derecho",
        populated_sections=["login", "file_transfer", "storage", "queue_specs", "top_software"],
    ),
    "neocortex": RPInfo(
        slug="neocortex",
        title="Neocortex",
        populated_sections=["login", "storage", "queue_specs"],
    ),
    "kyric": RPInfo(
        slug="kyric",
        title="KyRIC",
        populated_sections=["login", "storage", "queue_specs"],
    ),
}


class RPSectionCache:
    """In-memory cache of RP documentation sections.

    Currently uses hardcoded seed data.  When Drupal's ``/api/resources``
    returns ``populated_sections``, swap ``_load_seed()`` for a live fetch
    with TTL refresh.
    """

    def __init__(self) -> None:
        self._cache: dict[str, RPInfo] = {}
        self._load_seed()

    def _load_seed(self) -> None:
        """Load hardcoded seed data."""
        self._cache = dict(SEED_DATA)
        logger.info(f"RP section cache loaded with {len(self._cache)} resources (seed data)")

    def get(self, slug: str) -> RPInfo | None:
        """Look up RP info by slug. Returns None for unknown slugs."""
        return self._cache.get(slug)

    def is_valid_slug(self, slug: str) -> bool:
        """Check if a slug is in the cache (has documentation data)."""
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
