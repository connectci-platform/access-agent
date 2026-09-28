"""Tests for the lifespan-scoped checkpointer.

The pooled checkpointer's setup() must run once at app startup, not once
per request. Before this change, /api/v1/query opened a fresh connection
and ran setup() on every request via the per-request `async with
create_async_checkpointer(...) as cp: await cp.setup()` path.

stream_agent's graph execution is mocked (as in test_auth_e2e.py) so this
test isolates the checkpointer lifecycle from LLM/MCP infrastructure. The
checkpointer itself is a fake async context manager (no real Postgres/
SQLite needed) — only the call-count invariant is asserted, not any
backend specifics.
"""

from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

FAKE_AGENT_RESULT = {
    "final_answer": "Test response",
    "tools_used": [],
    "query_analysis": None,
    "query_classification": None,
}


class SpySaver:
    """Wraps a fake inner saver and counts setup() calls."""

    def __init__(self) -> None:
        self.setup_calls = 0

    async def setup(self) -> None:
        self.setup_calls += 1


class FakeCheckpointerCM:
    """Fake async context manager standing in for create_pooled_checkpointer's return."""

    def __init__(self, saver: SpySaver) -> None:
        self._saver = saver

    async def __aenter__(self) -> SpySaver:
        return self._saver

    async def __aexit__(self, *exc_info: object) -> None:
        return None


async def fake_stream_agent(**kwargs: object):
    yield "updates", {"__end__": FAKE_AGENT_RESULT}


async def test_setup_runs_once_not_per_request(monkeypatch, tmp_path):
    """setup() runs at startup, not per /api/v1/query request."""
    from src.config import settings

    # Non-empty so the lifespan takes the checkpointer-building branch. The
    # checkpointer itself is mocked below, so this URL is never dialed for it —
    # but the widget path's thread-owner claim_thread now reads the SAME setting,
    # so point that at a temp sqlite file (reset its singleton first).
    monkeypatch.setattr(
        settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'owners.db'}", raising=False
    )
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.thread_owners as _to

    _to._store = None

    spy = SpySaver()

    with (
        patch("src.main.create_pooled_checkpointer", return_value=FakeCheckpointerCM(spy)),
        patch("src.api.routes.stream_agent", side_effect=fake_stream_agent),
        patch("src.api.routes.get_registry", new_callable=AsyncMock) as mock_get_registry,
    ):
        mock_reg = AsyncMock()
        mock_reg.catalog = {"tools": [], "quick_lookup": {}}
        mock_get_registry.return_value = mock_reg

        from src.main import app

        async with app.router.lifespan_context(app):
            assert spy.setup_calls == 1
            assert app.state.checkpointer is spy

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                for _ in range(2):
                    response = await client.post(
                        "/api/v1/query",
                        json={"query": "What GPU resources are available?", "session_id": "s1"},
                    )
                    assert response.status_code == 200

            assert spy.setup_calls == 1


async def test_query_endpoint_passes_pooled_checkpointer_to_stream_agent(monkeypatch, tmp_path):
    """The widget stream call site threads app.state.checkpointer into stream_agent."""
    from src.config import settings

    monkeypatch.setattr(
        settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'owners.db'}", raising=False
    )
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.thread_owners as _to

    _to._store = None

    spy = SpySaver()

    with (
        patch("src.main.create_pooled_checkpointer", return_value=FakeCheckpointerCM(spy)),
        patch("src.api.routes.stream_agent", side_effect=fake_stream_agent) as mock_stream_agent,
        patch("src.api.routes.get_registry", new_callable=AsyncMock) as mock_get_registry,
    ):
        mock_reg = AsyncMock()
        mock_reg.catalog = {"tools": [], "quick_lookup": {}}
        mock_get_registry.return_value = mock_reg

        from src.main import app

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json={"query": "What GPU resources are available?", "session_id": "s2"},
                )
                assert response.status_code == 200

        mock_stream_agent.assert_called_once()
        assert mock_stream_agent.call_args.kwargs.get("checkpointer") is spy
