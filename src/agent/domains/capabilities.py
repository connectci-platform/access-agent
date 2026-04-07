"""Capability registry — aggregates all capabilities and serves them to the API and agent.

Categories define UI grouping.  General capabilities cover the RAG+tools pipeline.
Domain capabilities come from DomainAgentConfig.capabilities.  Everything is
aggregated here so consumers have one place to query.

For resource-scoped responses, the registry builds RP-specific capabilities
from the section-to-question mapping and the RPSectionCache.
"""

import logging
from typing import Any

from .config import Capability, Category

logger = logging.getLogger(__name__)

# ── Section-to-question mapping for RP-scoped capabilities ──────────────

SECTION_QUESTION_MAP: dict[str, dict[str, str]] = {
    "login": {
        "label": "Login",
        "description": "Get help logging in to {title}",
        "example_query": "How do I log in to {title}?",
    },
    "file_transfer": {
        "label": "File transfer",
        "description": "Learn about file transfer options on {title}",
        "example_query": "How do I transfer files to {title}?",
    },
    "storage": {
        "label": "Storage",
        "description": "Explore storage options and quotas on {title}",
        "example_query": "What storage is available on {title}?",
    },
    "queue_specs": {
        "label": "Job submission",
        "description": "Learn about queues and job submission on {title}",
        "example_query": "How do I submit a job on {title}?",
    },
    "top_software": {
        "label": "Software",
        "description": "See frequently used software on {title}",
        "example_query": "What software is available on {title}?",
    },
    "datasets": {
        "label": "Datasets",
        "description": "Browse datasets available on {title}",
        "example_query": "What datasets are available on {title}?",
    },
}

# ── Categories ────────────────────────────────────────────────────────────

CATEGORIES: list[Category] = [
    Category("general", "Ask a question", 0),
    Category("support", "Create a ticket", 1),
    Category("content", "Manage content", 2),
    Category("explore", "Explore resources", 3),
    Category("analytics", "Check usage", 4),
]

# ── General capabilities (non-domain pipeline) ───────────────────────────

GENERAL_CAPABILITIES: list[Capability] = [
    Capability(
        "ask_question",
        "Ask a question",
        "Get answers about ACCESS resources, policies, and services",
        "general",
        requires_auth=False,
    ),
    Capability(
        "check_allocations",
        "Check allocations",
        "Look up allocation details and status",
        "explore",
        requires_auth=False,
        example_query="What allocations are available for new researchers?",
    ),
    Capability(
        "search_software",
        "Search software",
        "Find software available on ACCESS resources",
        "explore",
        requires_auth=False,
        example_query="Is Python available on Delta?",
    ),
    Capability(
        "check_system_status",
        "Check system status",
        "See current outages and resource status",
        "explore",
        requires_auth=False,
        example_query="Are there any system outages right now?",
    ),
    Capability(
        "browse_events",
        "Browse events",
        "Find upcoming trainings, workshops, and office hours",
        "explore",
        requires_auth=False,
        example_query="Find upcoming workshops and training events",
    ),
    Capability(
        "browse_affinity_groups",
        "Browse affinity groups",
        "Explore community affinity groups",
        "explore",
        requires_auth=False,
        example_query="Show me affinity groups for machine learning",
    ),
    Capability(
        "check_usage",
        "Check usage (XDMoD)",
        "View resource usage and performance data",
        "analytics",
        requires_auth=False,
        example_query="Show my resource usage on Delta last month",
    ),
    Capability(
        "search_nsf_awards",
        "Search NSF awards",
        "Look up NSF award information",
        "explore",
        requires_auth=False,
        example_query="NSF awards for computational biology",
    ),
]


# ── Registry ──────────────────────────────────────────────────────────────


class CapabilityRegistry:
    """Aggregates capabilities from domain configs and general pipeline.

    Built once at startup, then serves fast in-memory lookups for the API
    and agent system prompt.
    """

    def __init__(
        self,
        capabilities: list[Capability],
        categories: list[Category],
        disabled_ids: set[str] | None = None,
    ) -> None:
        disabled = disabled_ids or set()

        # Filter out disabled, then index by id
        self._capabilities = {c.id: c for c in capabilities if c.id not in disabled and c.enabled}
        self._categories = {c.id: c for c in categories}

        if disabled:
            logger.info("Disabled capabilities: %s", ", ".join(sorted(disabled)))
        logger.info(
            "Capability registry: %d capabilities across %d categories",
            len(self._capabilities),
            len({c.category for c in self._capabilities.values()}),
        )

    # ── Queries ───────────────────────────────────────────────────────

    def get_all(self) -> list[Capability]:
        """All enabled capabilities."""
        return list(self._capabilities.values())

    def get_by_id(self, capability_id: str) -> Capability | None:
        """Look up a single capability."""
        return self._capabilities.get(capability_id)

    def get_categories(self) -> list[Category]:
        """Categories sorted by display order."""
        # Only return categories that have at least one capability
        active_cats = {c.category for c in self._capabilities.values()}
        return sorted(
            [c for c in self._categories.values() if c.id in active_cats],
            key=lambda c: c.order,
        )

    def get_by_category(self, authenticated: bool) -> list[dict[str, Any]]:
        """Capabilities grouped by category, ready for API serialization.

        Returns a list of category dicts, each with a 'capabilities' list.
        Auth-required capabilities are included for anonymous users but
        marked with 'locked': True.
        """
        categories = self.get_categories()
        caps_by_cat: dict[str, list[Capability]] = {}
        for cap in self._capabilities.values():
            caps_by_cat.setdefault(cap.category, []).append(cap)

        result = []
        for cat in categories:
            cat_caps = caps_by_cat.get(cat.id, [])
            if not cat_caps:
                continue
            result.append(
                {
                    "id": cat.id,
                    "label": cat.label,
                    "order": cat.order,
                    "capabilities": [
                        {
                            "id": c.id,
                            "label": c.label,
                            "description": c.description,
                            **(
                                {"requires_auth": True, "locked": True}
                                if c.requires_auth and not authenticated
                                else {}
                            ),
                            **(
                                {"example_query": c.example_query}
                                if c.example_query
                                else {}
                            ),
                        }
                        for c in cat_caps
                    ],
                }
            )
        return result

    # ── Resource-scoped capabilities ─────────────────────────────────

    def get_by_category_scoped(
        self, slug: str, authenticated: bool
    ) -> dict[str, Any] | None:
        """Build RP-scoped capabilities response.

        Returns a dict with ``resource_context``, ``categories``, and
        ``is_authenticated`` — or None if the slug isn't in the cache.
        """
        from ...services.rp_cache import get_rp_cache

        cache = get_rp_cache()
        rp_info = cache.get(slug)
        if rp_info is None:
            return None

        title = rp_info.title

        # Build resource_docs category from populated sections
        resource_caps = []
        for section in rp_info.populated_sections:
            mapping = SECTION_QUESTION_MAP.get(section)
            if not mapping:
                continue
            resource_caps.append({
                "id": "ask_about_resource",
                "label": mapping["label"],
                "description": mapping["description"].format(title=title),
                "example_query": mapping["example_query"].format(title=title),
                "section": section,
            })

        categories: list[dict[str, Any]] = []

        if resource_caps:
            categories.append({
                "id": "resource_docs",
                "label": f"About {title}",
                "order": 0,
                "capabilities": resource_caps,
            })

        # Always-present: support
        support_cap = self._capabilities.get("open_ticket")
        if support_cap:
            cap_dict: dict[str, Any] = {
                "id": support_cap.id,
                "label": support_cap.label,
                "description": support_cap.description,
                "example_query": f"Get help with a {title} issue",
            }
            if support_cap.requires_auth and not authenticated:
                cap_dict["requires_auth"] = True
                cap_dict["locked"] = True
            categories.append({
                "id": "support",
                "label": "Create a ticket",
                "order": 1,
                "capabilities": [cap_dict],
            })

        # Always-present: analytics
        usage_cap = self._capabilities.get("check_usage")
        if usage_cap:
            cap_dict = {
                "id": usage_cap.id,
                "label": usage_cap.label,
                "description": usage_cap.description,
                "example_query": f"Check my usage on {title}",
            }
            if usage_cap.requires_auth and not authenticated:
                cap_dict["requires_auth"] = True
                cap_dict["locked"] = True
            categories.append({
                "id": "analytics",
                "label": "Check usage",
                "order": 2,
                "capabilities": [cap_dict],
            })

        return {
            "resource_context": {"slug": slug, "title": title},
            "categories": categories,
            "is_authenticated": authenticated,
        }

    # ── Agent self-knowledge ──────────────────────────────────────────

    def get_system_prompt_section(self, authenticated: bool) -> str:
        """Generate a capability summary for the agent's system prompt."""
        categories = self.get_categories()
        caps_by_cat: dict[str, list[Capability]] = {}
        for cap in self._capabilities.values():
            # Skip locked capabilities for anonymous users in the prompt
            if cap.requires_auth and not authenticated:
                continue
            caps_by_cat.setdefault(cap.category, []).append(cap)

        lines = [
            "## What You Can Do",
            "You have the following capabilities. When relevant, mention these to help users discover features.",
            "",
        ]
        for cat in categories:
            cat_caps = caps_by_cat.get(cat.id, [])
            if not cat_caps:
                continue
            labels = ", ".join(c.label for c in cat_caps)
            auth_note = ""
            if all(c.requires_auth for c in cat_caps):
                auth_note = " (requires login)"
            lines.append(f"**{cat.label}{auth_note}:** {labels}")

        return "\n".join(lines)

    # ── Capability-to-query mapping ───────────────────────────────────

    def infer_capability_id(self, domain: str | None, tools_used: list[str] | None = None) -> str:
        """Infer the primary capability ID from classification results.

        Used by the usage logger to tag queries with the capability they exercised.
        """
        if domain:
            # Domain agent — map domain name to default capability.
            # More nuanced mapping (read vs write intent) can be added later.
            domain_default = {
                "announcements": "search_announcements",
                "jsm": "open_ticket",
            }
            cap_id = domain_default.get(domain)
            if cap_id and cap_id in self._capabilities:
                return cap_id

        # General pipeline — map by MCP server name in tool calls
        if tools_used:
            server_to_cap = {
                "allocations": "check_allocations",
                "software-discovery": "search_software",
                "system-status": "check_system_status",
                "events": "browse_events",
                "affinity-groups": "browse_affinity_groups",
                "xdmod": "check_usage",
                "xdmod-data": "check_usage",
                "nsf-awards": "search_nsf_awards",
                "announcements": "search_announcements",
            }
            for tool in tools_used:
                # Tool names are like "server_name__tool_name" — extract server
                server = tool.split("__")[0] if "__" in tool else tool
                if server in server_to_cap:
                    cap_id = server_to_cap[server]
                    if cap_id in self._capabilities:
                        return cap_id

        return "ask_question"


# ── Singleton ─────────────────────────────────────────────────────────────

_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry, building it on first call."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def _build_registry() -> CapabilityRegistry:
    """Build the registry from domain configs + general capabilities."""
    from ...config import settings
    from .registry import get_domain_registry

    # Collect all capabilities
    all_caps: list[Capability] = list(GENERAL_CAPABILITIES)
    domain_registry = get_domain_registry()
    for domain_name in domain_registry.list_domains():
        config = domain_registry.get(domain_name)
        if config:
            all_caps.extend(config.capabilities)

    # Parse disabled list from env
    disabled: set[str] = set()
    if settings.DISABLED_CAPABILITIES:
        disabled = {s.strip() for s in settings.DISABLED_CAPABILITIES.split(",") if s.strip()}

    return CapabilityRegistry(
        capabilities=all_caps,
        categories=CATEGORIES,
        disabled_ids=disabled,
    )
