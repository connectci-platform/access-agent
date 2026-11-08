"""Announcements domain agent configuration."""

from .config import Capability, DomainAgentConfig, McpBackend

ANNOUNCEMENTS_CONFIG = DomainAgentConfig(
    name="announcements",
    capabilities=[
        Capability(
            id="search_announcements",
            label="Search announcements",
            description="Find ACCESS news and announcements",
            example_query="Recent announcements about Expanse",
            category="explore",
            backend=McpBackend(servers=("announcements",)),
            requires_auth=False,
        ),
        Capability(
            id="manage_announcements",
            label="Manage your announcements",
            description="Create, update, and delete announcements you've authored",
            example_query="Show my draft announcements",
            category="content",
            backend=McpBackend(servers=("announcements",)),
            requires_auth=True,
        ),
    ],
)
