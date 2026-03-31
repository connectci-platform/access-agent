"""Domain agent configuration and capability data models."""

from dataclasses import dataclass, field


@dataclass
class Capability:
    """A single capability the agent can perform.

    Capabilities are the user-facing unit of functionality.  They drive
    the UI buttons, the agent's self-knowledge, and per-query analytics.
    IDs are opaque strings — no internal details leak to the client.
    """

    id: str
    """Opaque identifier, e.g. 'search_announcements'."""

    label: str
    """User-facing short label, e.g. 'Search announcements'."""

    description: str
    """User-facing description, e.g. 'Find ACCESS news and announcements'."""

    category: str
    """Category ID for UI grouping, e.g. 'explore'."""

    requires_auth: bool = False
    """Whether this capability requires an authenticated user."""

    enabled: bool = True
    """Toggle without redeploy (overridden by DISABLED_CAPABILITIES env var)."""


@dataclass
class Category:
    """Groups capabilities into UI sections."""

    id: str
    """Category identifier, e.g. 'explore'."""

    label: str
    """User-facing label, e.g. 'Explore resources'."""

    order: int
    """Display order in the UI (lower = first)."""


@dataclass
class DomainAgentConfig:
    """Configuration for a domain-specific react agent.

    Each domain (announcements, jsm, etc.) gets its own config that controls
    which MCP tools it can access and how it behaves.
    """

    name: str
    """Domain identifier (e.g. 'announcements', 'jsm')."""

    mcp_servers: list[str]
    """Which MCP servers' tools belong to this domain."""

    system_prompt: str
    """Domain-specific system prompt. May contain {acting_user} placeholder."""

    capabilities: list[Capability] = field(default_factory=list)
    """Capabilities this domain agent provides."""

    max_iterations: int = 10
    """Max react loop iterations before forcing a response."""

    temperature: float = 0.3
    """LLM temperature for the domain agent."""

    model_name: str | None = None
    """Override the default LLM model. None uses the provider default."""
