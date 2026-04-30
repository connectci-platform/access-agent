"""Capability registry — single source of truth for what the agent can do.

Every capability declares a ``backend`` (``McpBackend`` or ``RagBackend``)
that says what system actually serves it.  The registry aggregates
backends across enabled capabilities and exposes derived queries that
three runtime components read from:

1. **Tool catalog loader** (``src.tools.registry.ToolRegistry``) filters
   the MCP catalog to only servers whose owning capability is enabled.
2. **RAG answer node** consults ``enabled_rag_endpoints()`` before calling
   the RAG service, and ``scoped_rag_enabled()`` before doing scoped lookups.
3. **Graph router** (``route_after_rag``) and **domain agent node** check
   ``is_domain_enabled()`` before dispatching to a domain agent.

The operator interface is two env vars with deny-wins semantics:

- ``ENABLED_CAPABILITIES``: optional comma-separated allow-list
- ``DISABLED_CAPABILITIES``: always-wins comma-separated deny-list

Disabled capabilities disappear from the UI, the system prompt, the tool
catalog, RAG routing, domain routing, and usage attribution — all from
the same registry, so the layers cannot drift.

Categories define UI grouping.  General capabilities cover the RAG+tools
pipeline; domain capabilities come from DomainAgentConfig.capabilities.
For resource-scoped responses, the registry also builds RP-specific
capabilities from the section-to-question mapping and the RPSectionCache.
"""

import logging
from typing import Any

from .config import Capability, Category, McpBackend, RagBackend

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

# ── Attribution fallback order ────────────────────────────────────────────

# Used by CapabilityRegistry.infer_capability_id when neither a domain nor a
# tool identifies the exercised capability. Prefers RAG general, then RAG
# XDMoD, then exploratory capabilities. Ensures attribution is predictable
# under unusual operator allow-lists.
_ATTRIBUTION_FALLBACK_ORDER: tuple[str, ...] = (
    "ask_question",
    "ask_xdmod_question",
    "ask_about_resource",
    "check_allocations",
    "check_system_status",
)

# ── Write-capable capabilities ────────────────────────────────────────────

# Capabilities whose backend performs writes (POST/PUT/DELETE) against an
# external system. Source of truth for the READ_ONLY guard; enumerated in
# docs/security/write-capability-audit.md. If a new write-capable capability
# is added, it MUST be added here AND in the audit document.
WRITE_CAPABILITY_IDS: frozenset[str] = frozenset(
    {
        "manage_announcements",  # announcements domain: create/update/delete
        "open_ticket",  # jsm domain: create support ticket
        "report_login_problem",  # jsm domain: create login-issue ticket
        "report_security",  # jsm domain: create security-concern ticket
    }
)

# MCP tool names corresponding to the write capabilities above. The legacy
# chain enforces READ_ONLY by removing write capabilities from the registry,
# but the tool_calling_loop builds tools directly from the MCP catalog and
# never sees the registry — so it needs an explicit deny-list of MCP tool
# names. Keep in sync with the capabilities above.
WRITE_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # announcements domain (manage_announcements)
        "create_announcement",
        "update_announcement",
        "delete_announcement",
        # jsm domain (open_ticket / report_login_problem / report_security)
        "create_support_ticket",
        "create_login_ticket",
        "report_security_incident",
    }
)

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
    # ── RAG-backed capabilities ────────────────────────────────────────
    Capability(
        "ask_question",
        "Ask a question",
        "Get answers about ACCESS resources, policies, and services",
        "general",
        backend=RagBackend(endpoint="general"),
        requires_auth=False,
    ),
    Capability(
        "ask_xdmod_question",
        "Ask about usage metrics",
        "Get answers about ACCESS usage, XDMoD, and performance data",
        "analytics",
        backend=RagBackend(endpoint="xdmod"),
        requires_auth=False,
        example_query="How do I read an XDMoD chart for my allocation?",
    ),
    Capability(
        "ask_about_resource",
        "Ask about a specific resource",
        "Get answers scoped to a specific ACCESS resource's documentation",
        "explore",
        backend=RagBackend(endpoint="general", scoped=True),
        requires_auth=False,
    ),
    # ── MCP-backed capabilities ────────────────────────────────────────
    Capability(
        "check_allocations",
        "Check allocations",
        "Look up allocation details and status",
        "explore",
        backend=McpBackend(servers=("allocations",)),
        requires_auth=False,
        example_query="What allocations are available for new researchers?",
    ),
    Capability(
        "search_software",
        "Search software",
        "Find software available on ACCESS resources",
        "explore",
        backend=McpBackend(servers=("software-discovery",)),
        requires_auth=False,
        example_query="Is Python available on Delta?",
    ),
    Capability(
        "check_system_status",
        "Check system status",
        "See current outages and resource status",
        "explore",
        backend=McpBackend(servers=("system-status",)),
        requires_auth=False,
        example_query="Are there any system outages right now?",
    ),
    Capability(
        "browse_events",
        "Browse events",
        "Find upcoming trainings, workshops, and office hours",
        "explore",
        backend=McpBackend(servers=("events",)),
        requires_auth=False,
        example_query="Find upcoming workshops and training events",
    ),
    Capability(
        "browse_affinity_groups",
        "Browse affinity groups",
        "Explore community affinity groups",
        "explore",
        backend=McpBackend(servers=("affinity-groups",)),
        requires_auth=False,
        example_query="Show me affinity groups for machine learning",
    ),
    Capability(
        "check_usage",
        "Check usage (XDMoD)",
        "View resource usage and performance data",
        "analytics",
        backend=McpBackend(servers=("xdmod", "xdmod-data")),
        requires_auth=False,
        example_query="Show my resource usage on Delta last month",
    ),
    Capability(
        "search_nsf_awards",
        "Search NSF awards",
        "Look up NSF award information",
        "explore",
        backend=McpBackend(servers=("nsf-awards",)),
        requires_auth=False,
        example_query="NSF awards for computational biology",
    ),
]


# ── Registry ──────────────────────────────────────────────────────────────


class CapabilityRegistry:
    """Aggregates capabilities from domain configs and general pipeline.

    Built once at startup, then serves fast in-memory lookups for the API,
    agent system prompt, tool catalog, RAG routing, and usage attribution.

    The registry is the single source of truth for what the agent can do.
    Caller is responsible for passing the filtered capability list (after
    applying ENABLED/DISABLED env vars); the registry itself just indexes
    what it's given.
    """

    def __init__(
        self,
        capabilities: list[Capability],
        categories: list[Category],
    ) -> None:
        # Caller applies ENABLED/DISABLED filtering; we just index what's given.
        self._capabilities = {c.id: c for c in capabilities if c.enabled}
        self._categories = {c.id: c for c in categories}

    # ── Queries ───────────────────────────────────────────────────────

    def get_all(self) -> list[Capability]:
        """All enabled capabilities."""
        return list(self._capabilities.values())

    def get_by_id(self, capability_id: str) -> Capability | None:
        """Look up a single capability."""
        return self._capabilities.get(capability_id)

    # ── Backend-derived queries (runtime enforcement) ────────────────

    def enabled_mcp_servers(self) -> set[str]:
        """Union of MCP servers across enabled capabilities.

        Used by the tool catalog loader to filter out tools from servers
        whose owning capabilities are disabled.
        """
        servers: set[str] = set()
        for cap in self._capabilities.values():
            if isinstance(cap.backend, McpBackend):
                servers.update(cap.backend.servers)
        return servers

    def enabled_rag_endpoints(self) -> set[str]:
        """Set of RAG endpoints ('general', 'xdmod') enabled.

        Used by rag_answer_node to skip calls to disabled endpoints.
        """
        endpoints: set[str] = set()
        for cap in self._capabilities.values():
            if isinstance(cap.backend, RagBackend):
                endpoints.add(cap.backend.endpoint)
        return endpoints

    def scoped_rag_enabled(self) -> bool:
        """Whether resource-scoped RAG is enabled.

        True if any enabled capability has a RagBackend with scoped=True.
        """
        return any(
            isinstance(cap.backend, RagBackend) and cap.backend.scoped
            for cap in self._capabilities.values()
        )

    def is_domain_enabled(self, domain_name: str) -> bool:
        """True if any capability belonging to the given domain is enabled.

        Used by the graph router to bypass domain agent routing for
        domains that have no enabled capabilities.
        """
        from .registry import get_domain_registry

        config = get_domain_registry().get(domain_name)
        if not config:
            return False
        domain_cap_ids = {c.id for c in config.capabilities}
        return any(cap_id in self._capabilities for cap_id in domain_cap_ids)

    def capability_for_server(self, server: str) -> Capability | None:
        """Find the first enabled capability whose McpBackend includes this server.

        Used by usage attribution (infer_capability_id) to map an
        observed tool call back to the capability that owns it.

        When multiple capabilities share the same server (e.g. search and
        manage announcements both back on "announcements"), this returns
        the one declared first in registration order. **Convention:**
        declare read capabilities before write capabilities in
        GENERAL_CAPABILITIES and DomainAgentConfig so that read ops (the
        common case) win attribution by default.
        """
        for cap in self._capabilities.values():
            if isinstance(cap.backend, McpBackend) and server in cap.backend.servers:
                return cap
        return None

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
                            **({"example_query": c.example_query} if c.example_query else {}),
                        }
                        for c in cat_caps
                    ],
                }
            )
        return result

    # ── Resource-scoped capabilities ─────────────────────────────────

    def _build_scoped_cap_dict(
        self,
        capability_id: str,
        example_query: str,
        authenticated: bool,
    ) -> dict[str, Any] | None:
        """Build a single capability dict for an RP-scoped category.

        Returns None if the capability isn't in the registry.
        """
        cap = self._capabilities.get(capability_id)
        if not cap:
            return None
        cap_dict: dict[str, Any] = {
            "id": cap.id,
            "label": cap.label,
            "description": cap.description,
            "example_query": example_query,
        }
        if cap.requires_auth and not authenticated:
            cap_dict["requires_auth"] = True
            cap_dict["locked"] = True
        return cap_dict

    def _build_scoped_welcome(self, title: str, populated_sections: list[str]) -> str:
        """Build a contextual welcome message from a resource group's sections."""
        section_labels = [
            SECTION_QUESTION_MAP[s]["label"].lower()
            for s in populated_sections
            if s in SECTION_QUESTION_MAP
        ]
        if not section_labels:
            return f"Hi! Ask me anything about {title} or ACCESS."

        if len(section_labels) == 1:
            topics = section_labels[0]
        elif len(section_labels) == 2:
            topics = f"{section_labels[0]} and {section_labels[1]}"
        else:
            topics = ", ".join(section_labels[:-1]) + f", and {section_labels[-1]}"
        return (
            f"Hi! I can help with questions about {topics} on {title} "
            f"— or ask me anything about ACCESS."
        )

    async def get_by_category_scoped(self, slug: str, authenticated: bool) -> dict[str, Any] | None:
        """Build RP-scoped capabilities response.

        Returns a dict with ``resource_context``, ``categories``, and
        ``is_authenticated`` — or None if the slug isn't in the cache.
        """
        from ...services.rp_cache import get_rp_cache

        cache = get_rp_cache()
        await cache.ensure_loaded()
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
            resource_caps.append(
                {
                    "id": "ask_about_resource",
                    "label": mapping["label"],
                    "description": mapping["description"].format(title=title),
                    "example_query": mapping["example_query"].format(title=title),
                    "section": section,
                }
            )

        categories: list[dict[str, Any]] = []

        if resource_caps:
            categories.append(
                {
                    "id": "resource_docs",
                    "label": f"About {title}",
                    "order": 0,
                    "capabilities": resource_caps,
                }
            )

        # Always-present: support
        support_cap = self._build_scoped_cap_dict(
            "open_ticket", f"Get help with a {title} issue", authenticated
        )
        if support_cap:
            categories.append(
                {
                    "id": "support",
                    "label": "Create a ticket",
                    "order": 1,
                    "capabilities": [support_cap],
                }
            )

        # Always-present: analytics
        usage_cap = self._build_scoped_cap_dict(
            "check_usage", f"Check my usage on {title}", authenticated
        )
        if usage_cap:
            categories.append(
                {
                    "id": "analytics",
                    "label": "Check usage",
                    "order": 2,
                    "capabilities": [usage_cap],
                }
            )

        return {
            "resource_context": {"slug": slug, "title": title},
            "categories": categories,
            "is_authenticated": authenticated,
            "welcome_message": self._build_scoped_welcome(title, rp_info.populated_sections),
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

        Used by the usage logger to tag queries with the capability they
        exercised. Walks the registry's backend declarations — disabled
        capabilities stop appearing in attribution automatically because
        they're not in self._capabilities.
        """
        # Domain agent — return the first enabled capability in that domain
        # (read-before-write since read ops are far more common). Fall back
        # to any enabled capability in the domain.
        if domain:
            from .registry import get_domain_registry

            config = get_domain_registry().get(domain)
            if config:
                domain_cap_ids = [c.id for c in config.capabilities]
                # Prefer the first capability declared in the domain config
                # that's still enabled — this is usually the "read" op.
                for cap_id in domain_cap_ids:
                    if cap_id in self._capabilities:
                        return cap_id

        # General pipeline — walk the registry's McpBackend declarations to
        # find the capability whose server was called. The first tool's
        # server wins (consistent with old behavior).
        if tools_used:
            for tool in tools_used:
                # Tool names are like "server_name__tool_name" — extract server
                server = tool.split("__")[0] if "__" in tool else tool
                cap = self.capability_for_server(server)
                if cap is not None:
                    return cap.id

        # Fallback chain: prefer a generic RAG capability, then any
        # exploratory capability, finally a sentinel. Explicit ordering so
        # operators using unusual allow-lists get predictable attribution.
        for cap_id in _ATTRIBUTION_FALLBACK_ORDER:
            if cap_id in self._capabilities:
                return cap_id

        # Last resort: no preferred capability enabled. Log and return a
        # sentinel so usage analytics can filter it explicitly.
        logger.warning("infer_capability_id: no fallback capability enabled, returning 'unknown'")
        return "unknown"


# ── Singleton ─────────────────────────────────────────────────────────────

_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry, building it on first call."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def _build_registry() -> CapabilityRegistry:
    """Build the registry from domain configs + general capabilities.

    Applies ENABLED_CAPABILITIES (optional allow-list) and
    DISABLED_CAPABILITIES (always-wins deny-list) with deny-wins
    semantics: if a capability appears in both, it is disabled.
    """
    from ...config import settings
    from .registry import get_domain_registry

    # Collect all capabilities
    all_caps: list[Capability] = list(GENERAL_CAPABILITIES)
    domain_registry = get_domain_registry()
    for domain_name in domain_registry.list_domains():
        config = domain_registry.get(domain_name)
        if config:
            all_caps.extend(config.capabilities)

    # Parse ENABLED/DISABLED env vars
    enabled_set: set[str] | None = None
    if settings.ENABLED_CAPABILITIES:
        enabled_set = {s.strip() for s in settings.ENABLED_CAPABILITIES.split(",") if s.strip()}
    disabled_set: set[str] = set()
    if settings.DISABLED_CAPABILITIES:
        disabled_set = {s.strip() for s in settings.DISABLED_CAPABILITIES.split(",") if s.strip()}

    # Phase 1 safety guard: READ_ONLY forcibly adds every write capability to
    # the disabled set. This runs BEFORE the filter loop, so the deny-wins
    # semantics of the existing filter automatically picks it up.
    if settings.READ_ONLY:
        disabled_set = disabled_set | set(WRITE_CAPABILITY_IDS)
        logger.warning(
            "READ_ONLY=true active — write capabilities disabled: %s",
            ", ".join(sorted(WRITE_CAPABILITY_IDS)),
        )

    # Apply filter: allow-list first (if set), then deny-list always wins
    filtered: list[Capability] = []
    for cap in all_caps:
        if enabled_set is not None and cap.id not in enabled_set:
            continue
        if cap.id in disabled_set:
            continue
        filtered.append(cap)

    registry = CapabilityRegistry(capabilities=filtered, categories=CATEGORIES)

    # Operator-facing log: what did the filter resolve to?
    logger.info(
        "Capability filter: ENABLED=%s, DISABLED=%s",
        settings.ENABLED_CAPABILITIES or "(all)",
        settings.DISABLED_CAPABILITIES or "(none)",
    )
    active_ids = sorted(c.id for c in registry.get_all())
    logger.info(
        "Registry: %d capabilities active (of %d defined)",
        len(active_ids),
        len(all_caps),
    )
    if active_ids:
        logger.info("  Active: %s", ", ".join(active_ids))
    mcp_servers = sorted(registry.enabled_mcp_servers())
    logger.info("  → MCP servers: %s", ", ".join(mcp_servers) or "(none)")
    rag_endpoints = sorted(registry.enabled_rag_endpoints())
    logger.info("  → RAG endpoints: %s", ", ".join(rag_endpoints) or "(none)")
    logger.info("  → Scoped RAG: %s", "yes" if registry.scoped_rag_enabled() else "no")
    if not rag_endpoints:
        logger.warning("No RAG capabilities enabled — agent will rely on MCP tools only")

    return registry
