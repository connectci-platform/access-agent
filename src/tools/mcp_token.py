"""Acting-user tokens for the MCP tool servers (D8-2946).

The tool servers that act for a researcher (events, announcements,
allocations, and jsm for who a ticket is from) verify who the call is for
themselves, instead of trusting
``X-Acting-User``. They accept a token the portal mints at
``/api/token-exchange``: the agent, as the ``access-assistant`` confidential
client, exchanges the caller's verified agent token for one addressed to the
MCP servers (``aud``) with scope ``mcp:write`` and the same ``sub``.

The agent token itself is never forwarded: its audience is the agent.

Tokens are cached per ACCESS ID until shortly before they expire, and
concurrent tool calls for one user share one exchange. Neither token is ever
logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
import jwt

from ..telemetry.http import create_async_client

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..config import Settings

logger = logging.getLogger(__name__)

SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
MCP_SCOPE = "mcp:write"

# A cached token is replaced this long before it expires, so one that is
# handed to a tool call outlives the call (the longest per-tool timeout is 60s).
REFRESH_MARGIN_S = 90
# Sized well past the number of researchers signed in at once.
MAX_CACHED = 2000
# The portal mints for 900 seconds; anything longer is not what we asked for.
MAX_LIFETIME_S = 900
# A failed exchange is answered from memory this long, so a misconfiguration
# doesn't spend the portal's 60-an-hour allowance on every tool call.
FAILURE_TTL_S = 30


class McpTokenUnavailableError(Exception):
    """No acting-user token could be obtained. The message is safe to show."""


@dataclass(frozen=True)
class _Cached:
    token: str
    expires_at: float


class McpTokenExchanger:
    def __init__(
        self,
        *,
        exchange_url: str,
        client_id: str,
        client_secret: str,
        audience: str,
        verify_tls: bool = True,
        clock: Callable[[], float] = time.time,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._exchange_url = exchange_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._audience = audience
        self._verify_tls = verify_tls
        self._clock = clock
        self._transport = transport
        self._cache: OrderedDict[str, _Cached] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[_Cached]] = {}
        self._failures: OrderedDict[str, tuple[float, str]] = OrderedDict()

    @classmethod
    def from_settings(cls, settings: Settings) -> McpTokenExchanger:
        return cls(
            exchange_url=settings.mcp_token_exchange_url,
            client_id=settings.TOKEN_EXCHANGE_CLIENT_ID,
            client_secret=settings.TOKEN_EXCHANGE_CLIENT_SECRET,
            audience=settings.mcp_token_audience,
            verify_tls=settings.ENVIRONMENT not in ("local", "docker"),
        )

    @property
    def configured(self) -> bool:
        return bool(self._exchange_url and self._client_id and self._client_secret)

    async def token_for(self, acting_user: str, agent_token: str | None) -> str:
        """An MCP token for ``acting_user``, from the cache or a fresh exchange.

        ``agent_token`` is the verified Bearer token of the request being
        served, which must belong to ``acting_user``. It is required even for
        a cache hit, so only a request signed in as that user gets the token.
        """
        if not self.configured:
            raise McpTokenUnavailableError("the token exchange is not configured")
        if not agent_token:
            raise McpTokenUnavailableError("the request carries no sign-in token to exchange")
        # The agent token was verified for this request; a cached token is
        # only ever handed to a request signed in as the same user.
        if _subject(agent_token) != acting_user:
            raise McpTokenUnavailableError("the sign-in token belongs to another user")
        failed = self._failures.get(acting_user)
        if failed is not None and self._clock() - failed[0] < FAILURE_TTL_S:
            raise McpTokenUnavailableError(failed[1])
        cached = self._cache.get(acting_user)
        if cached is not None and cached.expires_at - self._clock() > REFRESH_MARGIN_S:
            self._cache.move_to_end(acting_user)
            return cached.token

        task = self._inflight.get(acting_user)
        if task is None:
            task = asyncio.get_running_loop().create_task(self._exchange(acting_user, agent_token))
            self._inflight[acting_user] = task
            task.add_done_callback(lambda t: self._forget(acting_user, t))
        # Shielded so one cancelled tool call doesn't cancel the exchange the
        # others are waiting on.
        entry = await asyncio.shield(task)
        return entry.token

    def refused(self, acting_user: str) -> None:
        """A tool server refused this user's token: drop it, and for
        FAILURE_TTL_S answer as a failed exchange instead of minting another,
        so a server that keeps refusing can't spend the portal's allowance
        (shared with XDMoD) on every call."""
        self._cache.pop(acting_user, None)
        self._remember_failure(acting_user, "the tool server refused the sign-in token")

    def _remember_failure(self, acting_user: str, reason: str) -> None:
        self._failures[acting_user] = (self._clock(), reason)
        self._failures.move_to_end(acting_user)
        while len(self._failures) > MAX_CACHED:
            self._failures.popitem(last=False)

    def _forget(self, acting_user: str, task: asyncio.Task[_Cached]) -> None:
        if self._inflight.get(acting_user) is task:
            del self._inflight[acting_user]
        if not task.cancelled() and isinstance(task.exception(), McpTokenUnavailableError):
            self._remember_failure(acting_user, str(task.exception()))
        else:
            self._failures.pop(acting_user, None)

    async def _exchange(self, acting_user: str, agent_token: str) -> _Cached:
        # RFC 6749 2.3.1: each half is form-encoded before base64.
        auth = httpx.BasicAuth(quote(self._client_id, safe=""), quote(self._client_secret, safe=""))
        data = {
            "grant_type": GRANT_TYPE,
            "subject_token": agent_token,
            "subject_token_type": SUBJECT_TOKEN_TYPE,
            "audience": self._audience,
            "scope": MCP_SCOPE,
        }
        try:
            async with self._client() as client:
                response = await client.post(self._exchange_url, data=data, auth=auth)
        except httpx.HTTPError as e:
            logger.error("AUTH_INFRA: MCP token exchange failed: %s", type(e).__name__)
            raise McpTokenUnavailableError(
                "the portal's token exchange could not be reached"
            ) from None

        if response.status_code != 200:
            # The RFC 6749 error code, never the body's other fields.
            code = _error_code(response)
            logger.error(
                "AUTH_INFRA: MCP token exchange refused: status %s, error %s",
                response.status_code,
                code,
            )
            raise McpTokenUnavailableError(f"the portal refused the token exchange ({code})")

        token = _access_token(response)
        claims = _unverified_claims(token)
        # The servers verify the signature; this only guards against caching
        # someone else's token, or one for another service, under this user.
        if claims.get("sub") != acting_user or claims.get("aud") != self._audience:
            logger.error(
                "AUTH_INFRA: MCP token exchange returned a token for another subject or audience"
            )
            raise McpTokenUnavailableError("the portal returned an unexpected token")
        exp = claims.get("exp")
        now = self._clock()
        if (
            not isinstance(exp, int)
            or isinstance(exp, bool)
            or not now < exp <= now + MAX_LIFETIME_S + 60
        ):
            raise McpTokenUnavailableError("the portal returned an unexpected token")

        entry = _Cached(token=token, expires_at=float(exp))
        self._cache[acting_user] = entry
        self._cache.move_to_end(acting_user)
        while len(self._cache) > MAX_CACHED:
            self._cache.popitem(last=False)
        return entry

    def _client(self) -> httpx.AsyncClient:
        if self._transport is not None:
            return httpx.AsyncClient(transport=self._transport, timeout=10.0)
        if not self._verify_tls:
            # ddev serves a self-signed certificate in local and docker.
            logger.warning("MCP token exchange TLS verification disabled")
        return create_async_client(timeout=10.0, verify=self._verify_tls)


def _error_code(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "unknown"
    code = body.get("error") if isinstance(body, dict) else None
    return code if isinstance(code, str) and code.isascii() and len(code) <= 40 else "unknown"


def _access_token(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise McpTokenUnavailableError("the portal returned an unexpected token")
    token, token_type = body.get("access_token"), body.get("token_type")
    if (
        not isinstance(token, str)
        or not token
        or not isinstance(token_type, str)
        or token_type.lower() != "bearer"
    ):
        raise McpTokenUnavailableError("the portal returned an unexpected token")
    return token


def _subject(agent_token: str) -> str | None:
    try:
        sub = jwt.decode(agent_token, options={"verify_signature": False}).get("sub")
    except jwt.InvalidTokenError:
        return None
    return sub if isinstance(sub, str) else None


def _unverified_claims(token: str) -> dict[str, Any]:
    try:
        claims: dict[str, Any] = jwt.decode(token, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        raise McpTokenUnavailableError("the portal returned an unexpected token") from None
    return claims


_exchanger: McpTokenExchanger | None = None


def get_mcp_token_exchanger() -> McpTokenExchanger:
    """The process-wide exchanger, built from settings on first use."""
    global _exchanger
    if _exchanger is None:
        from ..config import settings

        _exchanger = McpTokenExchanger.from_settings(settings)
    return _exchanger


def reset_mcp_token_exchanger() -> None:
    """Drop the process-wide exchanger and its cache (tests, settings changes)."""
    global _exchanger
    _exchanger = None
