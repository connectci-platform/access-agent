"""Drupal user profile fetcher for personalization.

Fetches user context from Drupal JSON:API for the /capabilities/personalized
endpoint.  Each data source is fetched independently so partial failures
return partial data rather than failing entirely.

Data sources (all existing Drupal fields, no new data collection):
- Active allocations: ``field_cider_resources`` on the user entity
- Affinity groups + coordinator status: ``mcp_my_affinity_groups`` view
- Institution: ``field_institution`` on the user entity
- HPC experience: ``field_hpc_experience`` on the user entity
- Skills and interests: Community Persona flags (taxonomy terms)

Results are cached per-user with a configurable TTL (default 5 minutes).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Assembled user profile from Drupal data."""

    access_id: str
    name: str = ""
    institution: str = ""
    hpc_experience: str = ""
    skills: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    affinity_groups: list[dict[str, Any]] = field(default_factory=list)
    active_allocations: list[dict[str, Any]] = field(default_factory=list)

    def to_context_dict(self) -> dict[str, Any]:
        """Build the ``context`` portion of the /personalized response."""
        ctx: dict[str, Any] = {}
        if self.institution:
            ctx["institution"] = self.institution
        if self.hpc_experience:
            ctx["hpc_experience"] = self.hpc_experience
        if self.skills:
            ctx["skills"] = self.skills
        if self.interests:
            ctx["interests"] = self.interests
        if self.affinity_groups:
            ctx["affinity_groups"] = self.affinity_groups
        if self.active_allocations:
            ctx["active_allocations"] = self.active_allocations
        return ctx

    def highlighted_capabilities(self) -> list[dict[str, str]]:
        """Derive highlighted capabilities from user context."""
        highlights: list[dict[str, str]] = []

        # Coordinators can manage announcements for their groups
        for group in self.affinity_groups:
            if group.get("is_coordinator"):
                highlights.append({
                    "id": "manage_announcements",
                    "label": f"Manage {group['name']} announcements",
                    "reason": f"You coordinate the {group['name']} affinity group",
                })

        # Users with allocations can check usage
        if self.active_allocations:
            resources = [a["resource"] for a in self.active_allocations if a.get("resource")]
            if resources:
                resource_list = ", ".join(resources[:3])
                suffix = f" and {len(resources) - 3} more" if len(resources) > 3 else ""
                highlights.append({
                    "id": "check_usage",
                    "label": f"Check your usage on {resource_list}{suffix}",
                    "reason": f"You have active allocations on {resource_list}{suffix}",
                })

        return highlights

    def to_system_prompt_section(self) -> str:
        """Format profile as a text block for injection into agent system prompts.

        Returns an empty string if the profile has no meaningful context.
        """
        lines: list[str] = []

        if self.name:
            lines.append(f"- Name: {self.name}")
        if self.institution:
            lines.append(f"- Institution: {self.institution}")
        if self.hpc_experience:
            lines.append(f"- HPC Experience: {self.hpc_experience}")
        if self.skills:
            lines.append(f"- Skills: {', '.join(self.skills)}")
        if self.interests:
            lines.append(f"- Interests: {', '.join(self.interests)}")
        if self.affinity_groups:
            group_strs = []
            for g in self.affinity_groups:
                label = g["name"]
                if g.get("is_coordinator"):
                    label += " (coordinator)"
                group_strs.append(label)
            lines.append(f"- Affinity Groups: {', '.join(group_strs)}")
        if self.active_allocations:
            alloc_strs = []
            for a in self.active_allocations:
                s = a.get("resource", "")
                if a.get("project"):
                    s += f" ({a['project']})"
                alloc_strs.append(s)
            lines.append(f"- Active Allocations: {', '.join(alloc_strs)}")

        if not lines:
            return ""

        return "## USER CONTEXT (from profile)\n\n" + "\n".join(lines)


@dataclass
class _CacheEntry:
    """Cached user profile with timestamp."""

    profile: UserProfile
    fetched_at: float


class DrupalProfileFetcher:
    """Fetches and caches user profile data from Drupal JSON:API.

    Each data source is fetched independently.  If one call fails,
    the others still populate the profile.
    """

    def __init__(self) -> None:
        self._cache: dict[str, _CacheEntry] = {}

    async def get_profile(self, access_id: str, jwt_cookie: str) -> UserProfile:
        """Get a user's profile, using cache if fresh enough."""
        entry = self._cache.get(access_id)
        if entry and (time.time() - entry.fetched_at) < settings.PROFILE_CACHE_TTL_SECONDS:
            return entry.profile

        profile = await self._fetch_profile(access_id, jwt_cookie)
        self._cache[access_id] = _CacheEntry(profile=profile, fetched_at=time.time())
        return profile

    def invalidate(self, access_id: str) -> None:
        """Remove a user from the cache."""
        self._cache.pop(access_id, None)

    async def _fetch_profile(self, access_id: str, jwt_cookie: str) -> UserProfile:
        """Fetch all profile data from Drupal, tolerating partial failures."""
        profile = UserProfile(access_id=access_id)
        base_url = settings.DRUPAL_BASE_URL

        cookies = {"SESSaccess_auth": jwt_cookie}
        headers = {"Accept": "application/vnd.api+json"}

        async with httpx.AsyncClient(timeout=15, cookies=cookies, headers=headers) as client:
            # Fetch user entity fields and affinity groups in parallel
            import asyncio

            user_task = asyncio.create_task(
                self._fetch_user_fields(client, base_url, profile)
            )
            groups_task = asyncio.create_task(
                self._fetch_affinity_groups(client, base_url, profile)
            )
            await asyncio.gather(user_task, groups_task, return_exceptions=True)

        logger.info(
            f"Fetched profile for {access_id}: "
            f"{len(profile.affinity_groups)} groups, "
            f"{len(profile.active_allocations)} allocations"
        )
        return profile

    async def _fetch_user_fields(
        self, client: httpx.AsyncClient, base_url: str, profile: UserProfile
    ) -> None:
        """Fetch user entity fields: name, institution, HPC experience, allocations."""
        try:
            # JSON:API user endpoint — the JWT cookie authenticates as the current user.
            # /jsonapi/user/user filters to the authenticated user when no ID is given.
            # We use a sparse fieldset to minimize payload.
            url = (
                f"{base_url}/jsonapi/user/user"
                "?filter[status]=1"
                "&fields[user--user]=display_name,field_institution,field_hpc_experience,field_cider_resources"
                "&page[limit]=1"
            )
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            users = data.get("data", [])
            if not users:
                logger.warning("No user entity returned from Drupal JSON:API")
                return

            user = users[0]
            attrs = user.get("attributes", {})

            profile.name = attrs.get("display_name", "")

            # Institution — may be a string or a structured field
            institution = attrs.get("field_institution")
            if isinstance(institution, str):
                profile.institution = institution
            elif isinstance(institution, dict):
                profile.institution = institution.get("value", "")

            # HPC experience
            hpc = attrs.get("field_hpc_experience")
            if isinstance(hpc, str):
                profile.hpc_experience = hpc
            elif isinstance(hpc, dict):
                profile.hpc_experience = hpc.get("value", "")

            # Active allocations from field_cider_resources
            cider = attrs.get("field_cider_resources")
            if isinstance(cider, list):
                for resource in cider:
                    if isinstance(resource, dict):
                        profile.active_allocations.append({
                            "resource": resource.get("name", resource.get("title", "")),
                            "project": resource.get("project", ""),
                        })
                    elif isinstance(resource, str):
                        profile.active_allocations.append({"resource": resource, "project": ""})

        except Exception:
            logger.warning("Failed to fetch user fields from Drupal", exc_info=True)

    async def _fetch_affinity_groups(
        self, client: httpx.AsyncClient, base_url: str, profile: UserProfile
    ) -> None:
        """Fetch affinity group memberships and coordinator status."""
        try:
            # The mcp_my_affinity_groups view is a JSON:API Views endpoint
            # that returns groups filtered by the authenticated user.
            url = f"{base_url}/jsonapi/views/mcp_my_affinity_groups/page_1"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            for group in data.get("data", []):
                attrs = group.get("attributes", {})
                profile.affinity_groups.append({
                    "name": attrs.get("title", ""),
                    "is_coordinator": attrs.get("field_is_coordinator", False),
                })

        except Exception:
            logger.warning("Failed to fetch affinity groups from Drupal", exc_info=True)


# Module-level singleton
_fetcher: DrupalProfileFetcher | None = None


def get_profile_fetcher() -> DrupalProfileFetcher:
    """Get the singleton profile fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = DrupalProfileFetcher()
    return _fetcher
