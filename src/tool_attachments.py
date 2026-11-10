"""Large tool attachments, stored once per thread instead of in its messages.

A tool message's artifact is checkpointed with the thread, and every later
checkpoint of that thread writes the whole messages channel again: 7 to 17
copies a turn on the dev database. A get_raw_data table of up to 50,000 rows
(about 2.6 MB) would be written that many times on every later turn, and
sent again in every history snapshot a reload reads. So the table is written
here once, and the artifact keeps a small reference: the table's description
without its rows, which the client fetches from
``GET /threads/{thread_id}/attachments/{id}`` when it shows them.

Charts stay inline: a chart payload is a few KB, and the client draws the
chart, its chip and its switcher label from the message the moment it
arrives.

The table is created by ``create_all`` on first use, like thread_owners, and
references thread_owners so that deleting a thread's owner row deletes its
attachments with it.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    event,
    select,
)
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .thread_owners import ThreadOwnerBase

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# structuredContent keys stored here; anything else stays on the message.
STORED_KINDS = ("table",)
REF_SUFFIX = "_ref"


class ToolAttachmentRow(ThreadOwnerBase):  # type: ignore[misc]
    __tablename__ = "tool_attachments"

    id = Column(String(32), primary_key=True)
    thread_id = Column(
        String(100),
        ForeignKey("thread_owners.thread_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind = Column(String(32), nullable=False)
    payload = Column(Text, nullable=False)  # JSON; Postgres compresses it out of line
    size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def reference_for(kind: str, attachment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The payload without its rows, plus the id that fetches them."""
    ref = {k: v for k, v in payload.items() if k != "rows"}
    ref.update(id=attachment_id, kind=kind)
    return ref


class ToolAttachmentStore:
    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def _ensure(self) -> bool:
        if self._session_factory is not None:
            return True
        if not settings.DATABASE_URL:
            return False
        db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
        self._engine = create_engine(db_url)
        if self._engine.dialect.name == "sqlite":
            event.listen(self._engine, "connect", _enable_sqlite_foreign_keys)
        ThreadOwnerBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        return True

    def put(self, thread_id: str, kind: str, payload: dict[str, Any]) -> str | None:
        """Store the payload and return its id, or None when it can't be stored
        (no database, or no owner row for the thread)."""
        if not self._ensure() or self._session_factory is None:
            return None
        text = json.dumps(payload, separators=(",", ":"))
        attachment_id = secrets.token_hex(16)
        session = self._session_factory()
        try:
            session.add(
                ToolAttachmentRow(
                    id=attachment_id,
                    thread_id=thread_id,
                    kind=kind,
                    payload=text,
                    size_bytes=len(text.encode()),
                )
            )
            session.commit()
            return attachment_id
        except Exception:
            session.rollback()
            logger.warning("couldn't store a %s attachment; it stays on the message", kind)
            return None
        finally:
            session.close()

    def get(self, thread_id: str, attachment_id: str) -> str | None:
        """The stored JSON, only when it belongs to this thread."""
        if not self._ensure() or self._session_factory is None:
            return None
        session = self._session_factory()
        try:
            return session.execute(
                select(ToolAttachmentRow.payload).where(
                    ToolAttachmentRow.id == attachment_id,
                    ToolAttachmentRow.thread_id == thread_id,
                )
            ).scalar_one_or_none()
        finally:
            session.close()

    def delete_thread(self, thread_id: str) -> int:
        """Delete a thread's attachments, for a path that deletes the thread's
        checkpoints but keeps its owner row. Returns how many went."""
        if not self._ensure() or self._session_factory is None:
            return 0
        session = self._session_factory()
        try:
            result = session.execute(
                delete(ToolAttachmentRow).where(ToolAttachmentRow.thread_id == thread_id)
            )
            session.commit()
            return int(getattr(result, "rowcount", 0) or 0)
        finally:
            session.close()


def externalize(
    structured: dict[str, Any], thread_id: str | None, store: ToolAttachmentStore | None = None
) -> dict[str, Any]:
    """structuredContent with each stored kind replaced by its reference.

    Without a thread (the widget's /query, the eval) or when the write fails,
    the payload stays inline, as before.
    """
    if not thread_id:
        return structured
    store = store or get_tool_attachment_store()
    out = dict(structured)
    for kind in STORED_KINDS:
        payload = out.get(kind)
        if not isinstance(payload, dict):
            continue
        attachment_id = store.put(thread_id, kind, payload)
        if attachment_id is None:
            continue
        del out[kind]
        out[kind + REF_SUFFIX] = reference_for(kind, attachment_id, payload)
    return out


_store: ToolAttachmentStore | None = None


def get_tool_attachment_store() -> ToolAttachmentStore:
    global _store
    if _store is None:
        _store = ToolAttachmentStore()
    return _store
