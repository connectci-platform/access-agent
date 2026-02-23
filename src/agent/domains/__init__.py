"""Domain agent configurations and registry."""

from .config import DomainAgentConfig
from .registry import get_domain_registry

__all__ = ["DomainAgentConfig", "get_domain_registry"]
