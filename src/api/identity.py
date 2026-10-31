"""Caller identity from the Bearer agent token, as one FastAPI dependency.

Every identity-bearing endpoint depends on ``request_identity``, so a refused
token is answered before the handler body runs: before Turnstile, before any
ownership write, and before a streaming response starts. A request with no
``Authorization`` header is anonymous. Cookies are never read.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from ..agent_token import (
    AgentIdentity,
    AgentTokenUnavailableError,
    AgentTokenVerifier,
    InvalidAgentTokenError,
)
from ..config import settings

# The verified raw token for this request, kept for the token exchange and
# nowhere else: not in agent state, checkpoints, spans or logs.
_agent_token: ContextVar[str | None] = ContextVar("agent_token", default=None)

_INVALID_TOKEN = {"WWW-Authenticate": 'Bearer error="invalid_token"'}

# A token must outlive the longest turn it could start, or a tool call late in
# the turn would act on an expired identity.
_LIFETIME_MARGIN_S = 30


def current_agent_token() -> str | None:
    """The verified Bearer token of the request being served, if any."""
    return _agent_token.get()


async def request_identity(request: Request) -> AgentIdentity | None:
    """The verified caller, or None for a request without ``Authorization``.

    Async on purpose: a sync dependency runs in a threadpool with a copied
    context, so the ContextVar set here would never reach the handler or the
    streamed generator.
    """
    _agent_token.set(None)
    header = request.headers.get("authorization")
    if header is None:
        return None

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid token", headers=_INVALID_TOKEN)

    verifier: AgentTokenVerifier = request.app.state.agent_token_verifier
    try:
        identity = await verifier.verify(token)
    except InvalidAgentTokenError:
        raise HTTPException(
            status_code=401, detail="Invalid token", headers=_INVALID_TOKEN
        ) from None
    except AgentTokenUnavailableError:
        # The verifier already logs each failed key-set fetch; logging every
        # request during an outage would flood.
        raise HTTPException(status_code=503, detail="Sign-in is unavailable") from None

    if identity.exp - time.time() < settings.AGENT_TURN_TIMEOUT_S + _LIFETIME_MARGIN_S:
        raise HTTPException(
            status_code=401, detail="Token expires too soon", headers=_INVALID_TOKEN
        )

    _agent_token.set(token)
    return identity


# The parameter type for an endpoint that takes the caller's identity.
Identity = Annotated[AgentIdentity | None, Depends(request_identity)]


def require_identity(identity: AgentIdentity | None) -> str:
    """The caller's ACCESS ID, or 401 for an anonymous caller."""
    if identity is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return identity.sub
