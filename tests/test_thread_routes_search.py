"""Tests for POST /threads/search (the sidebar).

Agent tokens come from the shared ``agent_tokens`` fixture. Unlike the other thread routes, search is authed-only
IN EFFECT: a caller with no verified identity has no user_hash to match, so
their list is empty ([]), not a 401 — see task-6-brief.md.

Turn reports are seeded directly against a temp-sqlite-backed TurnReporter
(mirrors tests/test_turn_reporter.py's TestWrite._reporter pattern) and that
reporter instance is patched into src.api.thread_routes.get_turn_reporter so
the endpoint reads from the same seeded DB.
"""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.turn_reporter import TurnReport, TurnReportBase, TurnReporter, _hash_user

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reporter():
    """A TurnReporter backed by a fresh in-memory sqlite DB."""
    r = TurnReporter()
    r._engine = create_engine("sqlite:///:memory:")
    TurnReportBase.metadata.create_all(r._engine)
    r._session_factory = sessionmaker(bind=r._engine)
    r._initialized = True
    return r


@pytest.fixture(autouse=True)
def _patch_turn_reporter(monkeypatch, reporter):
    """Point the search endpoint at the seeded reporter instance."""
    monkeypatch.setattr("src.api.thread_routes.get_turn_reporter", lambda: reporter)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def bearer_for(agent_tokens):
    return agent_tokens.headers


@pytest.fixture
def bearer(bearer_for):
    return bearer_for("me@x")


@pytest.fixture
def seed_turn(reporter):
    """Insert a turn_reports row directly with a known user_hash.

    created_at increments per call (a counter, offset backward from now) so
    ordering by max(created_at) desc is deterministic across rows inserted
    within the same test.
    """
    counter = {"n": 0}

    def _seed(
        *,
        session_id: str,
        acting_user: str,
        turn_index: int | None,
        query_text: str,
    ) -> None:
        counter["n"] += 1
        created_at = datetime.now(UTC) - timedelta(seconds=1000 - counter["n"])
        # Core insert, not TurnReport(...) + session.add(...): the ORM column
        # default (turn_index default=1) fires whenever the mapped attribute
        # is None at flush time, so an ORM insert can't express a genuine NULL
        # turn_index (the "failed first turn" case this query must handle).
        # A Core insert bypasses the ORM-level default and writes real NULL.
        with reporter._engine.begin() as conn:
            conn.execute(
                insert(TurnReport).values(
                    session_id=session_id,
                    turn_index=turn_index,
                    question_id=f"q-{session_id}-{counter['n']}",
                    query_text=query_text,
                    user_hash=_hash_user(acting_user),
                    created_at=created_at,
                )
            )

    return _seed


# ---------------------------------------------------------------------------
# POST /threads/search
# ---------------------------------------------------------------------------


async def test_search_lists_only_callers_threads(client, bearer_for, seed_turn):
    seed_turn(session_id="mine", acting_user="me@x", turn_index=1, query_text="hello mine")
    seed_turn(session_id="other", acting_user="you@x", turn_index=1, query_text="hello other")

    r = await client.post(
        "/api/v1/threads/search",
        json={"metadata": {"graph_id": "agent"}, "limit": 100},
        headers=bearer_for("me@x"),
    )
    assert r.status_code == 200
    body = r.json()
    ids = [t["thread_id"] for t in body]
    assert ids == ["mine"]
    assert body[0]["values"]["messages"][0]["content"] == "hello mine"
    assert body[0]["values"]["messages"][0]["type"] == "human"


async def test_search_label_falls_back_when_no_turn1(client, bearer_for, seed_turn):
    seed_turn(session_id="s", acting_user="me@x", turn_index=2, query_text="second turn")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        headers=bearer_for("me@x"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body[0]["values"]["messages"][0]["content"] == "second turn"  # not the UUID


async def test_search_label_prefers_turn1_over_later_turns(client, bearer_for, seed_turn):
    # Insert turn 2 first, then turn 1 — label must still be turn 1's text,
    # proving the ordering is by turn_index (with NULLS LAST), not insertion
    # order or created_at alone.
    seed_turn(session_id="s", acting_user="me@x", turn_index=2, query_text="second turn")
    seed_turn(session_id="s", acting_user="me@x", turn_index=1, query_text="first turn")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        headers=bearer_for("me@x"),
    )
    body = r.json()
    assert len(body) == 1
    assert body[0]["values"]["messages"][0]["content"] == "first turn"


async def test_search_null_turn_index_does_not_sort_ahead(client, bearer_for, seed_turn):
    # A failed first turn writes turn_index NULL. NULLS LAST must keep it from
    # sorting ahead of the real turn_index=1 row when picking the label.
    seed_turn(session_id="s", acting_user="me@x", turn_index=None, query_text="failed turn")
    seed_turn(session_id="s", acting_user="me@x", turn_index=1, query_text="real turn one")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        headers=bearer_for("me@x"),
    )
    body = r.json()
    assert len(body) == 1
    assert body[0]["values"]["messages"][0]["content"] == "real turn one"


async def test_search_orders_sessions_by_most_recent_activity(client, bearer_for, seed_turn):
    seed_turn(session_id="older", acting_user="me@x", turn_index=1, query_text="older session")
    seed_turn(session_id="newer", acting_user="me@x", turn_index=1, query_text="newer session")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        headers=bearer_for("me@x"),
    )
    ids = [t["thread_id"] for t in r.json()]
    assert ids == ["newer", "older"]


async def test_search_respects_limit(client, bearer_for, seed_turn):
    for i in range(5):
        seed_turn(session_id=f"s{i}", acting_user="me@x", turn_index=1, query_text=f"q{i}")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 2},
        headers=bearer_for("me@x"),
    )
    assert len(r.json()) == 2


async def test_search_no_token_returns_empty_list_not_401(client, seed_turn):
    seed_turn(session_id="mine", acting_user="me@x", turn_index=1, query_text="hello mine")

    r = await client.post("/api/v1/threads/search", json={"limit": 100})
    assert r.status_code == 200
    assert r.json() == []


async def test_search_response_shape(client, bearer_for, seed_turn):
    seed_turn(session_id="mine", acting_user="me@x", turn_index=1, query_text="hello mine")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        headers=bearer_for("me@x"),
    )
    thread = r.json()[0]
    assert thread["thread_id"] == "mine"
    assert "created_at" in thread
    assert "updated_at" in thread
    assert thread["metadata"] == {}
    assert thread["status"] == "idle"
    assert thread["interrupts"] == {}
    assert thread["values"]["messages"] == [{"type": "human", "content": "hello mine"}]
