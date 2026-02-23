"""Domain agent registry — lookup domain configs by name."""

import logging

from .config import DomainAgentConfig

logger = logging.getLogger(__name__)


class DomainAgentRegistry:
    """Registry of domain agent configurations."""

    def __init__(self) -> None:
        self._domains: dict[str, DomainAgentConfig] = {}

    def register(self, config: DomainAgentConfig) -> None:
        """Register a domain agent config."""
        self._domains[config.name] = config
        logger.debug(f"Registered domain agent: {config.name}")

    def get(self, name: str) -> DomainAgentConfig | None:
        """Look up a domain config by name."""
        return self._domains.get(name)

    def list_domains(self) -> list[str]:
        """List registered domain names."""
        return list(self._domains.keys())


# Lazy singleton
_registry: DomainAgentRegistry | None = None


def get_domain_registry() -> DomainAgentRegistry:
    """Get the global domain registry, registering all domains on first call."""
    global _registry
    if _registry is None:
        _registry = DomainAgentRegistry()
        _register_all_domains(_registry)
    return _registry


def _register_all_domains(registry: DomainAgentRegistry) -> None:
    """Import and register all domain configs."""
    from .announcements import ANNOUNCEMENTS_CONFIG
    from .jsm import JSM_CONFIG

    registry.register(ANNOUNCEMENTS_CONFIG)
    registry.register(JSM_CONFIG)
    logger.info(
        f"Registered {len(registry.list_domains())} domain agents: {registry.list_domains()}"
    )
