"""Write previews waiting for the user's Confirm click.

A write tool returns a preview until it is called with ``confirmed: true``.
The agent stores the preview's exact tool and arguments here under a random
id, and the Confirm button sends only that id. The agent then claims the id
once and replays the stored call itself, so what runs is what the user saw:
the model can't alter the arguments between preview and confirmation, and a
double click can't write twice.

The table has its own metadata, created on first use, like
src/tool_attachments.py, and shares the thread-owner store's engine. Failures
are logged and returned as None, never raised: a preview that can't be stored
is shown without a button.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, String, Text, Update, update
from sqlalchemy.orm import declarative_base

from .thread_owners import get_thread_owner_store

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

PREVIEW_TTL = timedelta(hours=24)

PreviewBase = declarative_base()


def _now() -> datetime:
    # Naive UTC: the column has no timezone, and SQLite and Postgres then compare alike.
    return datetime.now(UTC).replace(tzinfo=None)


class PendingPreviewRow(PreviewBase):  # type: ignore[valid-type,misc]
    __tablename__ = "pending_previews"

    id = Column(String(32), primary_key=True)
    thread_id = Column(String(100), nullable=False, index=True)
    tool = Column(String(100), nullable=False)
    args = Column(Text, nullable=False)  # JSON
    created_at = Column(DateTime, default=_now, nullable=False)
    claimed_at = Column(DateTime, nullable=True)


class PendingPreviewStore:
    def __init__(self) -> None:
        self._session_factory: sessionmaker[Session] | None = None

    def _sessions(self) -> sessionmaker[Session] | None:
        if self._session_factory is None:
            factory = get_thread_owner_store().session_factory()
            if factory is None:
                return None
            PreviewBase.metadata.create_all(factory.kw["bind"])
            self._session_factory = factory
        return self._session_factory

    def record(self, thread_id: str, tool: str, args: dict[str, Any]) -> str | None:
        """Store the call and return its id, or None when it can't be stored."""
        try:
            sessions = self._sessions()
            if sessions is None:
                return None
            preview_id = uuid.uuid4().hex
            with sessions() as session, session.begin():
                session.add(
                    PendingPreviewRow(
                        id=preview_id, thread_id=thread_id, tool=tool, args=json.dumps(args)
                    )
                )
            return preview_id
        except Exception:
            logger.warning("couldn't store a pending preview for %s", tool, exc_info=True)
            return None

    def claim(self, thread_id: str, preview_id: str) -> tuple[str, dict[str, Any]] | None:
        """Take the stored call, once. One UPDATE ... RETURNING decides the winner,
        so concurrent clicks can't both get it. None when the id is unknown,
        another thread's, already claimed, older than 24 hours, or on any error."""
        try:
            sessions = self._sessions()
            if sessions is None:
                return None
            now = _now()
            statement: Update = (
                update(PendingPreviewRow)
                .where(
                    PendingPreviewRow.id == preview_id,
                    PendingPreviewRow.thread_id == thread_id,
                    PendingPreviewRow.claimed_at.is_(None),
                    PendingPreviewRow.created_at > now - PREVIEW_TTL,  # type: ignore[arg-type]
                )
                .values(claimed_at=now)
                .returning(PendingPreviewRow.tool, PendingPreviewRow.args)
            )
            with sessions() as session, session.begin():
                row = session.execute(statement).first()
            if row is None:
                return None
            return str(row.tool), dict(json.loads(row.args))
        except Exception:
            logger.warning("couldn't claim a pending preview", exc_info=True)
            return None


_store: PendingPreviewStore | None = None


def get_pending_preview_store() -> PendingPreviewStore:
    global _store
    if _store is None:
        _store = PendingPreviewStore()
    return _store
