"""Agent-side LangChain middleware (create_agent hooks).

Distinct from ``langchain.agents.middleware`` (the upstream package this
re-exports/extends) — this package holds middleware authored for the loop's
``create_agent`` build, currently the anon write/private-read gate.
"""

from .requires_auth import REQUIRES_AUTH_SENTINEL, RequiresAuthMiddleware

__all__ = ["REQUIRES_AUTH_SENTINEL", "RequiresAuthMiddleware"]
