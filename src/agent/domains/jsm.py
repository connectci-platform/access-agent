"""JSM (Jira Service Management) domain agent configuration."""

from .config import Capability, DomainAgentConfig, McpBackend

JSM_CONFIG = DomainAgentConfig(
    name="jsm",
    capabilities=[
        Capability(
            id="open_ticket",
            label="Open a help ticket",
            description="Create a support ticket for technical issues",
            example_query="I want to create a support ticket",
            category="support",
            backend=McpBackend(servers=("jsm",)),
            requires_auth=False,
        ),
        Capability(
            id="report_login_problem",
            label="Report a login problem",
            description="Get help with ACCESS or resource login issues",
            example_query="I need help logging in to Anvil",
            category="support",
            backend=McpBackend(servers=("jsm",)),
            requires_auth=False,
        ),
        Capability(
            id="report_security",
            label="Report a security issue",
            description="Report a security concern to the ACCESS team",
            example_query="I need to report a security issue",
            category="support",
            backend=McpBackend(servers=("jsm",)),
            requires_auth=False,
        ),
    ],
)
