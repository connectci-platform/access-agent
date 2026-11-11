"""Acting-user tokens for the MCP tool servers (D8-2946)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import secrets
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from src.api.identity import _agent_token
from src.config import settings
from src.tools import mcp_token
from src.tools.mcp_client import MCPClient
from src.tools.mcp_token import McpTokenExchanger, McpTokenUnavailableError

if TYPE_CHECKING:
    from pytest_httpx import HTTPXMock

EXCHANGE_URL = "https://portal.test/api/token-exchange"
AUDIENCE = "https://mcp.access-ci.org"
SUB = "jsmith@access-ci.org"
SECRET = "s3cret:with/odd+chars"

_KEY = ec.generate_private_key(ec.SECP256R1())


def agent_jwt(sub: str = SUB) -> str:
    """An agent token for ``sub``; the agent verified it before any exchange."""
    now = int(time.time())
    payload = {
        "sub": sub,
        "aud": "http://localhost:8000",
        "jti": secrets.token_hex(4),
        "exp": now + 900,
    }
    return jwt.encode(payload, _KEY, algorithm="ES256")


AGENT_TOKEN = agent_jwt()


def mcp_jwt(**claims: Any) -> str:
    now = int(time.time())
    payload = {
        "iss": "https://portal.test",
        "sub": SUB,
        "aud": AUDIENCE,
        "azp": "access-assistant",
        "scope": "mcp:write",
        "iat": now,
        "exp": now + 900,
        "jti": "j1",
        **claims,
    }
    return jwt.encode(payload, _KEY, algorithm="ES256", headers={"typ": "at+jwt", "kid": "k"})


class Portal:
    """A stand-in /api/token-exchange that records what it was sent."""

    def __init__(self, status: int = 200, token: str | None = None, body: Any = None) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status
        self.token = token
        self.body = body
        self.delay = 0.0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.body is not None:
            return httpx.Response(self.status, json=self.body)
        if self.status != 200:
            return httpx.Response(
                self.status, json={"error": "invalid_grant", "error_description": "x"}
            )
        token = self.token or mcp_jwt()
        return httpx.Response(
            200,
            json={
                "access_token": token,
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "token_type": "Bearer",
                "expires_in": 900,
            },
        )


def exchanger(portal: Portal, **kwargs: Any) -> McpTokenExchanger:
    return McpTokenExchanger(
        exchange_url=EXCHANGE_URL,
        client_id="access-assistant",
        client_secret=SECRET,
        audience=AUDIENCE,
        transport=httpx.MockTransport(portal.handler),
        **kwargs,
    )


class TestExchanger:
    async def test_sends_an_rfc8693_exchange_for_the_mcp_audience(self):
        portal = Portal()
        token = await exchanger(portal).token_for(SUB, AGENT_TOKEN)

        assert jwt.decode(token, options={"verify_signature": False})["sub"] == SUB
        (request,) = portal.requests
        assert str(request.url) == EXCHANGE_URL
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        assert form == {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": AGENT_TOKEN,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "audience": AUDIENCE,
            "scope": "mcp:write",
        }
        # RFC 6749 2.3.1: id and secret are each form-encoded, then base64.
        scheme, _, creds = request.headers["authorization"].partition(" ")
        assert scheme == "Basic"
        assert base64.b64decode(creds).decode() == "access-assistant:s3cret%3Awith%2Fodd%2Bchars"

    async def test_caches_per_user_until_near_expiry(self):
        portal = Portal()
        now = [time.time()]
        ex = exchanger(portal, clock=lambda: now[0])
        first = await ex.token_for(SUB, AGENT_TOKEN)
        assert await ex.token_for(SUB, agent_jwt()) == first
        assert len(portal.requests) == 1

        now[0] += 900 - mcp_token.REFRESH_MARGIN_S + 1
        await ex.token_for(SUB, AGENT_TOKEN)
        assert len(portal.requests) == 2

    async def test_concurrent_calls_share_one_exchange(self):
        portal = Portal()
        portal.delay = 0.05
        ex = exchanger(portal)
        tokens = await asyncio.gather(*(ex.token_for(SUB, AGENT_TOKEN) for _ in range(5)))
        assert len(set(tokens)) == 1
        assert len(portal.requests) == 1

    async def test_a_cache_hit_still_needs_the_requests_agent_token(self):
        portal = Portal()
        ex = exchanger(portal)
        await ex.token_for(SUB, AGENT_TOKEN)
        with pytest.raises(McpTokenUnavailableError, match="no sign-in token"):
            await ex.token_for(SUB, None)

    @pytest.mark.parametrize(
        ("portal", "match"),
        [
            (Portal(status=400), "invalid_grant"),
            (Portal(status=429), "invalid_grant"),
            (Portal(token=mcp_jwt(sub="someone-else@access-ci.org")), "unexpected token"),
            (Portal(token=mcp_jwt(aud="https://xdmod.access-ci.org")), "unexpected token"),
            (Portal(token=mcp_jwt(exp=int(time.time()) - 10)), "unexpected token"),
            (Portal(body={"access_token": "x", "token_type": "Bearer"}), "unexpected token"),
            (Portal(body={"token_type": "Bearer"}), "unexpected token"),
            (Portal(body=["not", "an", "object"]), "unexpected token"),
        ],
    )
    async def test_failures_raise_and_cache_nothing(self, portal: Portal, match: str):
        ex = exchanger(portal)
        with pytest.raises(McpTokenUnavailableError, match=match):
            await ex.token_for(SUB, AGENT_TOKEN)
        assert ex._cache == {}

    async def test_unreachable_portal(self):
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        ex = McpTokenExchanger(
            exchange_url=EXCHANGE_URL,
            client_id="access-assistant",
            client_secret=SECRET,
            audience=AUDIENCE,
            transport=httpx.MockTransport(refuse),
        )
        with pytest.raises(McpTokenUnavailableError, match="could not be reached"):
            await ex.token_for(SUB, AGENT_TOKEN)

    async def test_unconfigured(self):
        ex = McpTokenExchanger(
            exchange_url=EXCHANGE_URL,
            client_id="access-assistant",
            client_secret="",
            audience=AUDIENCE,
        )
        with pytest.raises(McpTokenUnavailableError, match="not configured"):
            await ex.token_for(SUB, AGENT_TOKEN)

    async def test_tokens_are_never_logged(self, caplog: pytest.LogCaptureFixture):
        token = mcp_jwt()
        await exchanger(Portal(token=token)).token_for(SUB, AGENT_TOKEN)
        await asyncio.gather(
            *(
                _swallow(exchanger(p).token_for(SUB, AGENT_TOKEN))
                for p in (Portal(status=400), Portal(token=mcp_jwt(sub="x@access-ci.org")))
            )
        )
        for record in caplog.records:
            message = record.getMessage()
            for secret in (token, AGENT_TOKEN, SECRET):
                assert secret not in message


async def _swallow(coro: Any) -> None:
    with contextlib.suppress(McpTokenUnavailableError):
        await coro


class TestSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "MCP_TOKEN_EXCHANGE_URL", "")
        monkeypatch.setattr(settings, "MCP_TOKEN_AUDIENCE", "")
        monkeypatch.setattr(settings, "AGENT_TOKEN_ISSUER", "https://support.access-ci.org/")
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        assert settings.mcp_token_exchange_url == "https://support.access-ci.org/api/token-exchange"
        assert settings.mcp_token_audience == "https://mcp.access-ci.org"
        monkeypatch.setattr(settings, "ENVIRONMENT", "local")
        assert settings.mcp_token_audience == "https://mcp.dev.invalid"
        assert settings.MCP_ACTING_USER_TOKEN in ("off", "send", "require")
        assert "jsm" in settings.mcp_servers_acting_for_users

    def test_startup_refuses_token_mode_without_a_secret(self, monkeypatch: pytest.MonkeyPatch):
        from src.agent_token import check_agent_token_config

        monkeypatch.setattr(settings, "ENVIRONMENT", "local")
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        monkeypatch.setattr(settings, "TOKEN_EXCHANGE_CLIENT_SECRET", "")
        with pytest.raises(RuntimeError, match="TOKEN_EXCHANGE_CLIENT_SECRET"):
            check_agent_token_config(settings)
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "off")
        check_agent_token_config(settings)


# --- MCPClient ---------------------------------------------------------------

EVENTS = "http://localhost:3010"
JSM = "http://localhost:3012"
TOKEN = mcp_jwt()


@pytest.fixture
def portal(monkeypatch: pytest.MonkeyPatch) -> Portal:
    p = Portal(token=TOKEN)
    monkeypatch.setattr(mcp_token, "_exchanger", exchanger(p))
    monkeypatch.setattr(settings, "MCP_API_KEY", "key")
    return p


@pytest.fixture
def signed_in():
    reset = _agent_token.set(AGENT_TOKEN)
    yield
    _agent_token.reset(reset)


def client() -> MCPClient:
    c = MCPClient()
    c._server_urls = {**c._server_urls, "events": EVENTS, "jsm": JSM}
    return c


def ok(httpx_mock: HTTPXMock, url: str) -> None:
    httpx_mock.add_response(
        method="POST", url=url, json={"content": [{"type": "text", "text": "{}"}]}
    )


class TestMCPClientActingUserToken:
    async def test_off_sends_only_the_header(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "off")
        ok(httpx_mock, f"{EVENTS}/tools/register_for_event")
        await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        sent = httpx_mock.get_requests()[0]
        assert sent.headers["x-acting-user"] == SUB
        assert "authorization" not in sent.headers
        assert portal.requests == []

    @pytest.mark.parametrize("mode", ["send", "require"])
    async def test_sends_the_exchanged_token_with_the_header(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock, mode: str
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", mode)
        ok(httpx_mock, f"{EVENTS}/tools/register_for_event")
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert result.success
        sent = httpx_mock.get_requests()[0]
        assert sent.headers["authorization"] == f"Bearer {TOKEN}"
        assert sent.headers["x-acting-user"] == SUB
        assert sent.headers["x-api-key"] == "key"
        # The agent token itself never reaches a tool server.
        assert AGENT_TOKEN not in str(sent.headers)

    async def test_jsm_gets_the_token_to_file_under_the_signed_in_account(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        # D8-3069: the ticket server files a signed-in user's ticket under the
        # email and name on the verified token, not the model's arguments.
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        ok(httpx_mock, f"{JSM}/tools/create_support_ticket")
        await client().call_tool("jsm", "create_support_ticket", {}, acting_user=SUB)
        assert httpx_mock.get_requests()[0].headers["authorization"] == f"Bearer {TOKEN}"

    async def test_anonymous_calls_exchange_nothing(
        self, monkeypatch, portal, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        ok(httpx_mock, f"{EVENTS}/tools/search_events")
        await client().call_tool("events", "search_events", {})
        sent = httpx_mock.get_requests()[0]
        assert "authorization" not in sent.headers
        assert "x-acting-user" not in sent.headers

    async def test_send_falls_back_to_the_header_when_the_exchange_fails(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "send")
        portal.status = 400
        portal.token = None
        ok(httpx_mock, f"{EVENTS}/tools/register_for_event")
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert result.success
        sent = httpx_mock.get_requests()[0]
        assert "authorization" not in sent.headers
        assert sent.headers["x-acting-user"] == SUB

    async def test_require_fails_the_write_clearly_and_sends_nothing(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        portal.status = 400
        portal.token = None
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert not result.success
        assert "sign-in couldn't be confirmed" in (result.error or "")
        assert httpx_mock.get_requests() == []

    async def test_require_without_a_signed_in_request_sends_nothing(
        self, monkeypatch, portal, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert not result.success
        assert httpx_mock.get_requests() == []
        assert portal.requests == []


class TestEdges:
    async def test_a_non_integer_exp_is_refused(self):
        with pytest.raises(McpTokenUnavailableError, match="unexpected token"):
            await exchanger(Portal(token=mcp_jwt(exp="soon"))).token_for(SUB, AGENT_TOKEN)

    async def test_a_non_json_refusal_reports_unknown(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="<html>bad gateway</html>")

        ex = McpTokenExchanger(
            exchange_url=EXCHANGE_URL,
            client_id="access-assistant",
            client_secret=SECRET,
            audience=AUDIENCE,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(McpTokenUnavailableError, match=r"\(unknown\)"):
            await ex.token_for(SUB, AGENT_TOKEN)

    async def test_a_non_json_success_is_refused(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        ex = McpTokenExchanger(
            exchange_url=EXCHANGE_URL,
            client_id="access-assistant",
            client_secret=SECRET,
            audience=AUDIENCE,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(McpTokenUnavailableError, match="unexpected token"):
            await ex.token_for(SUB, AGENT_TOKEN)

    async def test_the_real_client_skips_tls_checks_only_when_told(
        self, caplog: pytest.LogCaptureFixture
    ):
        for verify in (True, False):
            ex = McpTokenExchanger(
                exchange_url=EXCHANGE_URL,
                client_id="c",
                client_secret="s",
                audience=AUDIENCE,
                verify_tls=verify,
            )
            async with ex._client() as http:
                assert isinstance(http, httpx.AsyncClient)
        assert "TLS verification disabled" in caplog.text

    def test_the_process_exchanger_comes_from_settings(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "MCP_TOKEN_EXCHANGE_URL", EXCHANGE_URL)
        monkeypatch.setattr(settings, "MCP_TOKEN_AUDIENCE", AUDIENCE)
        monkeypatch.setattr(settings, "TOKEN_EXCHANGE_CLIENT_SECRET", "s")
        mcp_token.reset_mcp_token_exchanger()
        try:
            ex = mcp_token.get_mcp_token_exchanger()
            assert ex is mcp_token.get_mcp_token_exchanger()
            assert ex.configured
            assert ex._audience == AUDIENCE
            assert ex._exchange_url == EXCHANGE_URL
        finally:
            mcp_token.reset_mcp_token_exchanger()

    def test_no_issuer_means_no_exchange_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "MCP_TOKEN_EXCHANGE_URL", "")
        monkeypatch.setattr(settings, "AGENT_TOKEN_ISSUER", "")
        assert settings.mcp_token_exchange_url == ""

    async def test_the_cache_is_bounded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(mcp_token, "MAX_CACHED", 1)

        def handler(request: httpx.Request) -> httpx.Response:
            form = parse_qs(request.content.decode())
            sub = jwt.decode(form["subject_token"][0], options={"verify_signature": False})["sub"]
            return httpx.Response(
                200, json={"access_token": mcp_jwt(sub=sub), "token_type": "Bearer"}
            )

        ex = McpTokenExchanger(
            exchange_url=EXCHANGE_URL,
            client_id="access-assistant",
            client_secret=SECRET,
            audience=AUDIENCE,
            transport=httpx.MockTransport(handler),
        )
        await ex.token_for("a@access-ci.org", agent_jwt("a@access-ci.org"))
        await ex.token_for(SUB, AGENT_TOKEN)
        assert list(ex._cache) == [SUB]

    async def test_a_cache_hit_needs_an_agent_token_for_the_same_user(self):
        portal = Portal()
        ex = exchanger(portal)
        await ex.token_for(SUB, AGENT_TOKEN)
        with pytest.raises(McpTokenUnavailableError, match="another user"):
            await ex.token_for(SUB, agent_jwt("someone-else@access-ci.org"))
        with pytest.raises(McpTokenUnavailableError, match="another user"):
            await ex.token_for(SUB, "not-a-jwt")
        assert len(portal.requests) == 1

    async def test_a_failed_exchange_is_not_retried_for_a_while(self):
        portal = Portal(status=400)
        now = [time.time()]
        ex = exchanger(portal, clock=lambda: now[0])
        for _ in range(3):
            with pytest.raises(McpTokenUnavailableError, match="invalid_grant"):
                await ex.token_for(SUB, AGENT_TOKEN)
        assert len(portal.requests) == 1
        now[0] += mcp_token.FAILURE_TTL_S + 1
        portal.status, portal.token = 200, None
        await ex.token_for(SUB, AGENT_TOKEN)
        assert len(portal.requests) == 2


def refuse(httpx_mock: HTTPXMock, status: int) -> None:
    headers = {"www-authenticate": 'Bearer error="invalid_token"'} if status == 401 else {}
    httpx_mock.add_response(
        method="POST",
        url=f"{EVENTS}/tools/register_for_event",
        status_code=status,
        headers=headers,
        json={"error": "refused"},
        match_headers={"Authorization": f"Bearer {TOKEN}"},
    )


class TestServerRefusesTheToken:
    @pytest.mark.parametrize("status", [401, 503])
    async def test_send_drops_the_token_and_retries_once_with_the_header(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock, status: int
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "send")
        refuse(httpx_mock, status)
        ok(httpx_mock, f"{EVENTS}/tools/register_for_event")
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert result.success
        first, second = httpx_mock.get_requests()
        assert first.headers["authorization"] == f"Bearer {TOKEN}"
        assert "authorization" not in second.headers
        assert second.headers["x-acting-user"] == SUB
        assert SUB not in mcp_token._exchanger._cache

    async def test_require_drops_the_token_and_fails_clearly(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        refuse(httpx_mock, 401)
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert not result.success
        assert "didn't accept the researcher's sign-in" in (result.error or "")
        assert len(httpx_mock.get_requests()) == 1
        assert SUB not in mcp_token._exchanger._cache

    async def test_an_ordinary_error_is_not_retried(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "send")
        httpx_mock.add_response(
            method="POST", url=f"{EVENTS}/tools/register_for_event", status_code=500, json={}
        )
        result = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert not result.success
        assert len(httpx_mock.get_requests()) == 1
        assert SUB in mcp_token._exchanger._cache


class TestProductionStartup:
    def test_production_needs_the_exchange_url_and_audience(self, monkeypatch: pytest.MonkeyPatch):
        from src.agent_token import check_agent_token_config

        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "AGENT_TOKEN_ISSUER", "https://support.access-ci.org")
        monkeypatch.setattr(
            settings,
            "AGENT_TOKEN_JWKS_URL",
            "https://support.access-ci.org/.well-known/exchange-jwks.json",
        )
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "send")
        monkeypatch.setattr(settings, "TOKEN_EXCHANGE_CLIENT_SECRET", "s")
        monkeypatch.setattr(settings, "MCP_TOKEN_EXCHANGE_URL", "")
        monkeypatch.setattr(settings, "MCP_TOKEN_AUDIENCE", "")
        with pytest.raises(RuntimeError, match="MCP_TOKEN_EXCHANGE_URL and MCP_TOKEN_AUDIENCE"):
            check_agent_token_config(settings)
        monkeypatch.setattr(
            settings, "MCP_TOKEN_EXCHANGE_URL", "http://support.access-ci.org/api/token-exchange"
        )
        monkeypatch.setattr(settings, "MCP_TOKEN_AUDIENCE", "https://mcp.access-ci.org")
        with pytest.raises(RuntimeError, match="https"):
            check_agent_token_config(settings)
        monkeypatch.setattr(
            settings, "MCP_TOKEN_EXCHANGE_URL", "https://support.access-ci.org/api/token-exchange"
        )
        check_agent_token_config(settings)

    async def test_remembered_failures_are_bounded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(mcp_token, "MAX_CACHED", 1)
        ex = exchanger(Portal(status=400))
        for sub in ("a@access-ci.org", SUB):
            with pytest.raises(McpTokenUnavailableError):
                await ex.token_for(sub, agent_jwt(sub))
        assert list(ex._failures) == [SUB]


class TestRefusalIsRemembered:
    async def test_send_skips_the_exchange_while_a_refusal_is_fresh(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "send")
        refuse(httpx_mock, 401)
        ok(httpx_mock, f"{EVENTS}/tools/register_for_event")
        ok(httpx_mock, f"{EVENTS}/tools/register_for_event")
        await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        second = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert second.success
        assert len(portal.requests) == 1
        assert "authorization" not in httpx_mock.get_requests()[-1].headers

    async def test_require_fails_while_a_refusal_is_fresh(
        self, monkeypatch, portal, signed_in, httpx_mock: HTTPXMock
    ):
        monkeypatch.setattr(settings, "MCP_ACTING_USER_TOKEN", "require")
        refuse(httpx_mock, 401)
        await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        second = await client().call_tool("events", "register_for_event", {}, acting_user=SUB)
        assert not second.success
        assert "refused" in (second.error or "")
        assert len(portal.requests) == 1
        assert len(httpx_mock.get_requests()) == 1

    async def test_the_refusal_expires(self):
        now = [time.time()]
        portal = Portal()
        ex = exchanger(portal, clock=lambda: now[0])
        await ex.token_for(SUB, AGENT_TOKEN)
        ex.refused(SUB)
        with pytest.raises(McpTokenUnavailableError, match="refused"):
            await ex.token_for(SUB, AGENT_TOKEN)
        now[0] += mcp_token.FAILURE_TTL_S + 1
        await ex.token_for(SUB, AGENT_TOKEN)
        assert len(portal.requests) == 2
