"""Deleting a user's conversations removes everything stored for them (D8-3126).

Runs against a temp SQLite file, like the stores' own tests: every table the
agent keeps per thread gets a row for two threads, one is deleted, and only
the other's rows remain. The deletion itself is logged without content.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
import sqlalchemy as sa

from src import conversation_delete
from src.config import settings
from src.conversation_delete import (
    ConversationDeletion,
    delete_conversations,
    owned_thread_ids,
)
from src.pending_previews import PreviewBase
from src.thread_owners import ThreadOwnerBase
from src.tool_attachments import AttachmentBase
from src.turn_reporter import TurnReportBase
from src.usage_logger import Base as UsageBase

ALICE = "alice@access-ci.org"
BOB = "bob@access-ci.org"
GONE, KEPT, OTHER = "t-gone", "t-kept", "t-bob"


def _hash(user: str) -> str:
    import hashlib

    return hashlib.sha256(user.encode()).hexdigest()[:16]


def _filler(column: sa.Column) -> Any:
    kind = column.type
    if isinstance(kind, sa.Boolean):
        return False
    if isinstance(kind, sa.Integer | sa.Float):
        return 0
    if isinstance(kind, sa.DateTime):
        return datetime.now(UTC)
    if isinstance(kind, sa.JSON):
        return {}
    return "x"


def _row(table: sa.Table, **given: Any) -> dict[str, Any]:
    """The given values plus a filler for every other required column."""
    row = dict(given)
    for column in table.columns:
        if column.name in row or column.nullable:
            continue
        if column.primary_key and column.autoincrement is True:
            continue
        if column.default is not None or column.server_default is not None:
            continue
        row[column.name] = _filler(column)
    return row


@pytest.fixture
def db(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'd.db'}"
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)
    conversation_delete._reset_engine_for_test()
    engine = sa.create_engine(url)
    for base in (ThreadOwnerBase, AttachmentBase, PreviewBase, TurnReportBase, UsageBase):
        base.metadata.create_all(engine)
    yield engine
    conversation_delete._reset_engine_for_test()


def _seed(engine: sa.Engine, thread_id: str, user: str | None) -> None:
    meta = {
        **ThreadOwnerBase.metadata.tables,
        **AttachmentBase.metadata.tables,
        **PreviewBase.metadata.tables,
        **TurnReportBase.metadata.tables,
        **UsageBase.metadata.tables,
    }
    user_hash = _hash(user) if user else None
    with engine.begin() as conn:
        t = meta["thread_owners"]
        conn.execute(t.insert(), _row(t, thread_id=thread_id, user_hash=user_hash))
        t = meta["thread_claim_keys"]
        conn.execute(t.insert(), _row(t, thread_id=thread_id))
        t = meta["tool_attachments"]
        conn.execute(t.insert(), _row(t, id=f"a-{thread_id}", thread_id=thread_id))
        t = meta["pending_previews"]
        conn.execute(t.insert(), _row(t, id=f"p-{thread_id}", thread_id=thread_id))
        t = meta["turn_ratings"]
        conn.execute(t.insert(), _row(t, thread_id=thread_id, message_id="m1"))
        t = meta["turn_reports"]
        report_id = conn.execute(
            t.insert(), _row(t, session_id=thread_id, user_hash=user_hash, query_text="q?")
        ).inserted_primary_key[0]
        t = meta["report_tool_calls"]
        conn.execute(t.insert(), _row(t, report_id=report_id, tool_name="search_resources"))
        t = meta["usage_logs"]
        conn.execute(t.insert(), _row(t, session_id=thread_id, query_text="q?"))


TABLES = [
    ("thread_owners", "thread_id"),
    ("thread_claim_keys", "thread_id"),
    ("tool_attachments", "thread_id"),
    ("pending_previews", "thread_id"),
    ("turn_ratings", "thread_id"),
    ("turn_reports", "session_id"),
    ("usage_logs", "session_id"),
]


def _threads_in(engine: sa.Engine, table: str, column: str) -> set[str]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(sa.text(f"SELECT {column} FROM {table}"))}


class _Checkpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


async def test_deleting_one_conversation_removes_all_its_rows_and_nothing_else(db):
    for tid, user in ((GONE, ALICE), (KEPT, ALICE), (OTHER, BOB)):
        _seed(db, tid, user)
    checkpointer = _Checkpointer()

    deleted = await delete_conversations([GONE], user=ALICE, scope="one", checkpointer=checkpointer)

    assert deleted == 1
    assert checkpointer.deleted == [GONE]
    for table, column in TABLES:
        assert _threads_in(db, table, column) == {KEPT, OTHER}, table
    with db.connect() as conn:
        calls = conn.execute(sa.text("SELECT COUNT(*) FROM report_tool_calls")).scalar()
    assert calls == 2


async def test_the_deletion_is_logged_without_content(db):
    _seed(db, GONE, ALICE)
    await delete_conversations([GONE], user=ALICE, scope="all", checkpointer=None)

    with db.connect() as conn:
        rows = conn.execute(sa.select(ConversationDeletion.__table__)).mappings().all()
    assert len(rows) == 1
    row = rows[0]
    assert row["user_hash"] == _hash(ALICE)
    assert row["scope"] == "all"
    assert row["thread_count"] == 1
    assert row["deleted_at"] is not None
    assert set(row.keys()) == {"id", "user_hash", "scope", "thread_count", "deleted_at"}


async def test_owned_thread_ids_finds_a_users_threads_from_owners_and_reports(db):
    _seed(db, GONE, ALICE)
    _seed(db, OTHER, BOB)
    # A report for a thread whose owner row is missing still counts as theirs.
    t = TurnReportBase.metadata.tables["turn_reports"]
    with db.begin() as conn:
        conn.execute(
            t.insert(), _row(t, session_id="t-orphan", user_hash=_hash(ALICE), query_text="q")
        )

    assert sorted(owned_thread_ids(ALICE)) == [GONE, "t-orphan"]
    assert owned_thread_ids(BOB) == [OTHER]
    assert owned_thread_ids("nobody@access-ci.org") == []


async def test_missing_tables_are_skipped(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)
    conversation_delete._reset_engine_for_test()
    assert await delete_conversations([GONE], user=ALICE, scope="one", checkpointer=None) == 1
    assert owned_thread_ids(ALICE) == []
    conversation_delete._reset_engine_for_test()


async def test_nothing_to_delete_logs_nothing(db):
    assert await delete_conversations([], user=ALICE, scope="all", checkpointer=None) == 0
    with db.connect() as conn:
        assert not sa.inspect(conn).has_table("conversation_deletions") or (
            conn.execute(sa.text("SELECT COUNT(*) FROM conversation_deletions")).scalar() == 0
        )


async def test_without_a_database_it_raises(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    conversation_delete._reset_engine_for_test()
    with pytest.raises(conversation_delete.DeletionUnavailableError):
        await delete_conversations([GONE], user=ALICE, scope="one", checkpointer=None)
    with pytest.raises(conversation_delete.DeletionUnavailableError):
        owned_thread_ids(ALICE)


async def test_deleted_ids_are_recorded_so_they_cant_come_back(db):
    _seed(db, GONE, ALICE)
    await delete_conversations([GONE], user=ALICE, scope="one", checkpointer=None)

    with db.connect() as conn:
        ids = set(conn.execute(sa.text("SELECT thread_id FROM deleted_threads")).scalars())
    assert ids == {GONE}


async def test_many_threads_are_deleted_in_batches(db, monkeypatch):
    monkeypatch.setattr(conversation_delete, "_CHUNK", 2)
    for i in range(5):
        _seed(db, f"t{i}", ALICE)

    assert (
        await delete_conversations(
            [f"t{i}" for i in range(5)], user=ALICE, scope="all", checkpointer=None
        )
        == 5
    )
    assert _threads_in(db, "thread_owners", "thread_id") == set()


async def test_a_second_delete_reuses_the_log_table(db):
    _seed(db, GONE, ALICE)
    _seed(db, KEPT, ALICE)
    await delete_conversations([GONE], user=ALICE, scope="one", checkpointer=None)
    await delete_conversations([KEPT], user=ALICE, scope="one", checkpointer=None)

    with db.connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM conversation_deletions")).scalar() == 2


def test_the_log_table_race_is_tolerated_only_if_the_table_exists(db, monkeypatch):
    def racing(*a, **k):
        raise sa.exc.ProgrammingError("CREATE TABLE", {}, Exception("exists"))

    monkeypatch.setattr(conversation_delete.DeletionBase.metadata, "create_all", racing)
    engine = conversation_delete._get_engine()
    with pytest.raises(sa.exc.ProgrammingError):
        conversation_delete._ensure_log_table(engine)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE conversation_deletions (id INTEGER)"))
    conversation_delete._ensure_log_table(engine)
