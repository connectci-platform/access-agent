"""Domain agent configurations and registry."""

from .capabilities import get_capability_registry
from .config import Capability, Category, DomainAgentConfig
from .registry import get_domain_registry

__all__ = [
    "Capability",
    "Category",
    "DomainAgentConfig",
    "get_capability_registry",
    "get_domain_registry",
]
