"""Deleting a user's conversations and everything stored for them (D8-3126).

A conversation lives in several tables, all keyed by its thread id: the
LangGraph checkpoints (deleted through the checkpointer), shared files,
pending change previews, ratings, turn reports and their tool calls, usage
logs, the anonymous claim key and the ownership row. ``delete_conversations``
removes all of it in one transaction, then logs that a deletion happened: the
user's hash, when, how many conversations, and whether it was one or all.
The log holds no content, so a deletion can be accounted for without keeping
what was deleted.

The tables are created by their own stores on first use, so a table that
doesn't exist yet simply has nothing to delete.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base

from .config import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

Scope = Literal["one", "all"]

# (table, column holding the thread id), children before the rows they hang
# off. report_tool_calls is deleted through its reports, before them.
_THREAD_TABLES: tuple[tuple[str, str], ...] = (
    ("tool_attachments", "thread_id"),
    ("pending_previews", "thread_id"),
    ("turn_ratings", "thread_id"),
    ("turn_reports", "session_id"),
    ("usage_logs", "session_id"),
    ("thread_claim_keys", "thread_id"),
    ("thread_owners", "thread_id"),
)

DeletionBase = declarative_base()


class ConversationDeletion(DeletionBase):  # type: ignore[valid-type,misc]
    """That a deletion happened, never what was deleted."""

    __tablename__ = "conversation_deletions"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_hash = sa.Column(sa.String(16), index=True)
    scope = sa.Column(sa.String(8), nullable=False)
    thread_count = sa.Column(sa.Integer, nullable=False)
    deleted_at = sa.Column(sa.DateTime, nullable=False, default=lambda: datetime.now(UTC))


class DeletionUnavailableError(RuntimeError):
    """No database is configured, so nothing can be found or deleted."""


_engine: sa.Engine | None = None


def _get_engine() -> sa.Engine:
    global _engine
    if not settings.DATABASE_URL:
        raise DeletionUnavailableError("DATABASE_URL is not set")
    if _engine is None:
        url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
        _engine = sa.create_engine(url)
    return _engine


def _hash_user(user: str | None) -> str | None:
    # The same hash thread_owners and turn_reports store; None when signed out.
    return hashlib.sha256(user.encode()).hexdigest()[:16] if user else None


def owned_thread_ids(user: str) -> list[str]:
    """Every thread the user owns or has turn reports for."""
    engine = _get_engine()
    user_hash = _hash_user(user)
    found: set[str] = set()
    with engine.connect() as conn:
        tables = sa.inspect(conn)
        if tables.has_table("thread_owners"):
            found.update(
                conn.execute(
                    sa.text("SELECT thread_id FROM thread_owners WHERE user_hash = :h"),
                    {"h": user_hash},
                ).scalars()
            )
        if tables.has_table("turn_reports"):
            found.update(
                conn.execute(
                    sa.text(
                        "SELECT DISTINCT session_id FROM turn_reports "
                        "WHERE user_hash = :h AND session_id IS NOT NULL"
                    ),
                    {"h": user_hash},
                ).scalars()
            )
    return sorted(found)


def _delete_rows(thread_ids: Sequence[str], user: str | None, scope: Scope) -> None:
    engine = _get_engine()
    ids: sa.BindParameter[list[str]] = sa.bindparam("ids", expanding=True)
    with engine.begin() as conn:
        tables = sa.inspect(conn)
        if tables.has_table("report_tool_calls") and tables.has_table("turn_reports"):
            conn.execute(
                sa.text(
                    "DELETE FROM report_tool_calls WHERE report_id IN "
                    "(SELECT id FROM turn_reports WHERE session_id IN :ids)"
                ).bindparams(ids),
                {"ids": list(thread_ids)},
            )
        for table, column in _THREAD_TABLES:
            if tables.has_table(table):
                conn.execute(
                    sa.text(f"DELETE FROM {table} WHERE {column} IN :ids").bindparams(ids),
                    {"ids": list(thread_ids)},
                )
        DeletionBase.metadata.create_all(conn)
        conn.execute(
            sa.insert(ConversationDeletion.__table__).values(
                user_hash=_hash_user(user),
                scope=scope,
                thread_count=len(thread_ids),
                deleted_at=datetime.now(UTC),
            )
        )


async def delete_conversations(
    thread_ids: Sequence[str],
    *,
    user: str | None,
    scope: Scope,
    checkpointer: Any | None,
) -> int:
    """Delete the threads' checkpoints and every row stored for them.

    The caller has already checked that ``user`` may delete each thread
    (``None``: a signed-out caller holding the thread's claim key). The
    checkpoints go first: if the row deletion then fails, the conversation
    can't be reopened and the call can simply be repeated. Returns how many
    conversations were deleted.
    """
    if not thread_ids:
        return 0
    _get_engine()
    if checkpointer is not None:
        for thread_id in thread_ids:
            await checkpointer.adelete_thread(thread_id)
    await asyncio.to_thread(_delete_rows, list(thread_ids), user, scope)
    logger.info("Deleted %d conversation(s) (scope=%s)", len(thread_ids), scope)
    return len(thread_ids)


def _reset_engine_for_test() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
