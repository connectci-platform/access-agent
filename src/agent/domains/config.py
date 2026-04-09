"""Domain agent configuration and capability data models."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class McpBackend:
    """A capability backed by one or more MCP servers.

    All tools from the listed servers are considered part of this
    capability.  When the capability is disabled, those servers' tools
    are filtered from the tool catalog so the planner never sees them.
    """

    servers: tuple[str, ...]


@dataclass(frozen=True)
class RagBackend:
    """A capability backed by a RAG endpoint.

    ``endpoint`` selects which RAG service to consult ('general' for
    ACCESS docs, 'xdmod' for usage metrics).  ``scoped=True`` means the
    RAG call is resource-scoped to a specific RP via the user's
    resource_context.
    """

    endpoint: Literal["general", "xdmod"]
    scoped: bool = False


Backend = McpBackend | RagBackend


@dataclass
class Capability:
    """A single capability the agent can perform.

    Capabilities are the user-facing unit of functionality.  They drive
    the UI buttons, the agent's self-knowledge, and per-query analytics.
    IDs are stable semantic names (e.g. ``search_announcements``) used
    in operator env vars, usage logs, and the capabilities API — but do
    NOT reveal internal implementation details like MCP tool names or
    server hostnames.

    The ``backend`` field declares what system actually serves the
    capability.  The capability registry aggregates backends across
    enabled capabilities and uses them to gate runtime behavior: the MCP
    tool catalog, RAG routing, domain agent routing, and usage
    attribution.  A capability with ``backend=None`` is pure prompt
    behavior with no external call (reserved for future use).
    """

    id: str
    """Stable semantic identifier, e.g. 'search_announcements'."""

    label: str
    """User-facing short label, e.g. 'Search announcements'."""

    description: str
    """User-facing description, e.g. 'Find ACCESS news and announcements'."""

    category: str
    """Category ID for UI grouping, e.g. 'explore'."""

    backend: Backend | None = None
    """What system serves this capability.  None = pure prompt behavior."""

    requires_auth: bool = False
    """Whether this capability requires an authenticated user."""

    enabled: bool = True
    """Toggle without redeploy (overridden by DISABLED_CAPABILITIES env var)."""

    example_query: str = ""
    """Example query shown in discovery responses, e.g. 'Is Python available on Delta?'"""


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
