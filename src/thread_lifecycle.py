"""Writes in flight on a thread, and threads that were deleted (D8-3126).

A deleted conversation must not come back. Several writes can land on a
thread after its run lock is released: the turn report, a stopped turn's
save, a rating. Each goes through ``writing(thread_id)``, which registers the
write and then refuses it if the thread was deleted. A delete marks the
threads deleted first, then ``settle`` waits for the writes already running,
then removes the data. A write that starts after the mark sees it and skips;
one that started before is waited for. The agent runs as one process (see
thread_runs.py), so the in-memory registry covers every writer.

Deleted thread ids are also kept in ``deleted_threads``, so after a restart
the id can't be claimed again as a new conversation.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base

from .config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

logger = logging.getLogger(__name__)

LifecycleBase = declarative_base()


class DeletedThread(LifecycleBase):  # type: ignore[valid-type,misc]
    """A thread id that was deleted. Holds no content."""

    __tablename__ = "deleted_threads"

    thread_id = sa.Column(sa.String(100), primary_key=True)
    deleted_at = sa.Column(sa.DateTime, nullable=False, default=lambda: datetime.now(UTC))


class ThreadDeletedError(Exception):
    """The thread was deleted; the write must not happen."""


_deleted: set[str] = set()
_inflight: dict[str, int] = {}
_idle: dict[str, asyncio.Event] = {}
_engine: sa.Engine | None = None
_tables_ready = False


def _get_engine() -> sa.Engine | None:
    global _engine, _tables_ready
    if not settings.DATABASE_URL:
        return None
    if _engine is None:
        url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
        _engine = sa.create_engine(url)
    if not _tables_ready:
        try:
            LifecycleBase.metadata.create_all(_engine)
        except sa.exc.SQLAlchemyError:
            # Another worker or request may have created it at the same moment.
            if not sa.inspect(_engine).has_table("deleted_threads"):
                raise
        _tables_ready = True
    return _engine


def ensure_tables(connection: sa.Connection) -> None:
    """Create ``deleted_threads`` inside a caller's transaction if needed."""
    if not sa.inspect(connection).has_table("deleted_threads"):
        LifecycleBase.metadata.create_all(connection)


def mark_deleted(thread_ids: Iterable[str]) -> None:
    """From now on, writes on these threads are refused in this process."""
    _deleted.update(thread_ids)


def unmark_deleted(thread_ids: Iterable[str]) -> None:
    """A delete that didn't go through: let the threads be written again."""
    _deleted.difference_update(thread_ids)


def record_deleted_sync(thread_ids: Iterable[str], connection: sa.Connection | None = None) -> None:
    """Store the ids, so a restart still knows them. Inside ``connection``'s
    transaction when given."""
    ids = list(dict.fromkeys(thread_ids))
    if not ids:
        return
    if connection is None:
        engine = _get_engine()
        if engine is None:
            return
        with engine.begin() as conn:
            record_deleted_sync(ids, conn)
        return
    ensure_tables(connection)
    table = DeletedThread.__table__
    existing = set(
        connection.execute(sa.select(table.c.thread_id).where(table.c.thread_id.in_(ids))).scalars()
    )
    now = datetime.now(UTC)
    new = [{"thread_id": tid, "deleted_at": now} for tid in ids if tid not in existing]
    if new:
        connection.execute(sa.insert(table), new)


def is_deleted(thread_id: str) -> bool:
    """Whether the thread was deleted, here or before a restart."""
    if thread_id in _deleted:
        return True
    try:
        engine = _get_engine()
        if engine is None:
            return False
        with engine.connect() as conn:
            found = conn.execute(
                sa.select(DeletedThread.__table__.c.thread_id).where(
                    DeletedThread.__table__.c.thread_id == thread_id
                )
            ).first()
    except sa.exc.SQLAlchemyError:
        # Unreachable database: this process's own deletes are still in
        # memory, and a run on a deleted id has no history to bring back.
        logger.warning("Deleted-thread lookup unavailable")
        return False
    if found:
        _deleted.add(thread_id)
        return True
    return False


@asynccontextmanager
async def writing(thread_id: str) -> AsyncIterator[None]:
    """Register a write on the thread, refusing it if the thread was deleted.

    Registers first and checks second, so a delete that marks the thread in
    between still waits for this write in ``settle``.
    """
    _inflight[thread_id] = _inflight.get(thread_id, 0) + 1
    _idle.setdefault(thread_id, asyncio.Event()).clear()
    try:
        if thread_id in _deleted:
            raise ThreadDeletedError(thread_id)
        yield
    finally:
        remaining = _inflight[thread_id] - 1
        if remaining:
            _inflight[thread_id] = remaining
        else:
            del _inflight[thread_id]
            _idle.pop(thread_id).set()


async def settle(thread_ids: Iterable[str], *, timeout: float) -> bool:
    """Wait for writes already running on these threads. False on timeout."""
    waits = [asyncio.ensure_future(_idle[tid].wait()) for tid in thread_ids if tid in _idle]
    if not waits:
        return True
    _, pending = await asyncio.wait(waits, timeout=timeout)
    for waiting in pending:
        waiting.cancel()
    return not pending


def _reset_memory_for_test() -> None:
    _deleted.clear()
    _inflight.clear()
    _idle.clear()


def _reset_for_test() -> None:
    global _engine, _tables_ready
    _reset_memory_for_test()
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _tables_ready = False
