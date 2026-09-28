"""Tests for POST /threads/search (the sidebar).

Harness copied from tests/test_thread_routes_cancel.py (JWKS server + ES256
JWT cookie minting). Unlike the other thread routes, search is authed-only
IN EFFECT: a caller with no verified identity has no user_hash to match, so
their list is empty ([]), not a 401 — see task-6-brief.md.

Turn reports are seeded directly against a temp-sqlite-backed TurnReporter
(mirrors tests/test_turn_reporter.py's TestWrite._reporter pattern) and that
reporter instance is patched into src.api.thread_routes.get_turn_reporter so
the endpoint reads from the same seeded DB.
"""

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from httpx import ASGITransport, AsyncClient
from jwt import algorithms as jwt_algorithms
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker

from src.auth import configure_trusted_issuers
from src.main import app
from src.turn_reporter import TurnReport, TurnReportBase, TurnReporter, _hash_user

# ---------------------------------------------------------------------------
# Test key pair and JWKS server (copied from tests/test_thread_routes_cancel.py)
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "thread-search-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "thread-search-test-kid"

# list_threads_for_user's query relies on window functions (SQLite >= 3.25)
# and explicit NULLS LAST ordering (SQLite >= 3.30) to behave identically to
# Postgres. Skip rather than silently exercise a divergent code path if the
# interpreter's bundled sqlite3 predates that — see task-6-brief.md.
_SQLITE_MIN_VERSION = (3, 30, 0)
pytestmark = pytest.mark.skipif(
    sqlite3.sqlite_version_info < _SQLITE_MIN_VERSION,
    reason=(
        f"list_threads_for_user requires SQLite >= {'.'.join(map(str, _SQLITE_MIN_VERSION))} "
        f"for window functions + explicit NULLS LAST; runtime has {sqlite3.sqlite_version}"
    ),
)


class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(JWKS_RESPONSE)

    def log_message(self, format, *args):
        pass


def _start_jwks_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def _make_jwt(sub: str, issuer: str = ISSUER) -> str:
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": sub,
        "iat": now - 3600,
        "exp": now + 3600,
    }
    return jwt.encode(payload, PRIVATE_PEM, algorithm="ES256", headers={"kid": KID})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _jwks_server():
    server, jwks_url = _start_jwks_server()
    configure_trusted_issuers({ISSUER: jwks_url})
    yield
    server.shutdown()
    configure_trusted_issuers({})


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
def valid_cookie_for():
    def _make(sub: str) -> dict[str, str]:
        return {"SESSaccess_auth": _make_jwt(sub)}

    return _make


@pytest.fixture
def valid_cookie(valid_cookie_for):
    return valid_cookie_for("me@x")


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


async def test_search_lists_only_callers_threads(client, valid_cookie_for, seed_turn):
    seed_turn(session_id="mine", acting_user="me@x", turn_index=1, query_text="hello mine")
    seed_turn(session_id="other", acting_user="you@x", turn_index=1, query_text="hello other")

    r = await client.post(
        "/api/v1/threads/search",
        json={"metadata": {"graph_id": "agent"}, "limit": 100},
        cookies=valid_cookie_for("me@x"),
    )
    assert r.status_code == 200
    body = r.json()
    ids = [t["thread_id"] for t in body]
    assert ids == ["mine"]
    assert body[0]["values"]["messages"][0]["content"] == "hello mine"
    assert body[0]["values"]["messages"][0]["type"] == "human"


async def test_search_label_falls_back_when_no_turn1(client, valid_cookie_for, seed_turn):
    seed_turn(session_id="s", acting_user="me@x", turn_index=2, query_text="second turn")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        cookies=valid_cookie_for("me@x"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body[0]["values"]["messages"][0]["content"] == "second turn"  # not the UUID


async def test_search_label_prefers_turn1_over_later_turns(client, valid_cookie_for, seed_turn):
    # Insert turn 2 first, then turn 1 — label must still be turn 1's text,
    # proving the ordering is by turn_index (with NULLS LAST), not insertion
    # order or created_at alone.
    seed_turn(session_id="s", acting_user="me@x", turn_index=2, query_text="second turn")
    seed_turn(session_id="s", acting_user="me@x", turn_index=1, query_text="first turn")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        cookies=valid_cookie_for("me@x"),
    )
    body = r.json()
    assert len(body) == 1
    assert body[0]["values"]["messages"][0]["content"] == "first turn"


async def test_search_null_turn_index_does_not_sort_ahead(client, valid_cookie_for, seed_turn):
    # A failed first turn writes turn_index NULL. NULLS LAST must keep it from
    # sorting ahead of the real turn_index=1 row when picking the label.
    seed_turn(session_id="s", acting_user="me@x", turn_index=None, query_text="failed turn")
    seed_turn(session_id="s", acting_user="me@x", turn_index=1, query_text="real turn one")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        cookies=valid_cookie_for("me@x"),
    )
    body = r.json()
    assert len(body) == 1
    assert body[0]["values"]["messages"][0]["content"] == "real turn one"


async def test_search_orders_sessions_by_most_recent_activity(client, valid_cookie_for, seed_turn):
    seed_turn(session_id="older", acting_user="me@x", turn_index=1, query_text="older session")
    seed_turn(session_id="newer", acting_user="me@x", turn_index=1, query_text="newer session")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        cookies=valid_cookie_for("me@x"),
    )
    ids = [t["thread_id"] for t in r.json()]
    assert ids == ["newer", "older"]


async def test_search_respects_limit(client, valid_cookie_for, seed_turn):
    for i in range(5):
        seed_turn(session_id=f"s{i}", acting_user="me@x", turn_index=1, query_text=f"q{i}")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 2},
        cookies=valid_cookie_for("me@x"),
    )
    assert len(r.json()) == 2


async def test_search_no_cookie_returns_empty_list_not_401(client, seed_turn):
    seed_turn(session_id="mine", acting_user="me@x", turn_index=1, query_text="hello mine")

    r = await client.post("/api/v1/threads/search", json={"limit": 100})
    assert r.status_code == 200
    assert r.json() == []


async def test_search_response_shape(client, valid_cookie_for, seed_turn):
    seed_turn(session_id="mine", acting_user="me@x", turn_index=1, query_text="hello mine")

    r = await client.post(
        "/api/v1/threads/search",
        json={"limit": 100},
        cookies=valid_cookie_for("me@x"),
    )
    thread = r.json()[0]
    assert thread["thread_id"] == "mine"
    assert "created_at" in thread
    assert "updated_at" in thread
    assert thread["metadata"] == {}
    assert thread["status"] == "idle"
    assert thread["interrupts"] == {}
    assert thread["values"]["messages"] == [{"type": "human", "content": "hello mine"}]
