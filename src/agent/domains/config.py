"""Domain agent configuration."""

from dataclasses import dataclass


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

    max_iterations: int = 10
    """Max react loop iterations before forcing a response."""

    temperature: float = 0.3
    """LLM temperature for the domain agent."""

    model_name: str | None = None
    """Override the default LLM model. None uses the provider default."""
