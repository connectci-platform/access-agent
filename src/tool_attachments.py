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

Only a full-screen thread run asks MCP servers for structured content, so a
table only ever arrives where it can be stored. If it can't be stored, it is
dropped and the model is told, never left inline.

After each run the thread route deletes the thread's attachments that its
messages no longer refer to (``keep_only``). That one rule covers a turn that
was stopped and replaced, retried or edited, and one that compaction folded
into a summary, along with any path added later.

The table has its own metadata, created on first use, so a failure to make it
(a permissions error, say) can only lose tables: the thread-owner tables, and
every access check that reads them, never depend on it. It shares the owner
store's engine and pool.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, Integer, String, Text, delete, select
from sqlalchemy.orm import declarative_base

from .thread_owners import get_thread_owner_store

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy import Select
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# structuredContent keys stored here; anything else stays on the message.
STORED_KINDS = ("table",)
REF_SUFFIX = "_ref"


AttachmentBase = declarative_base()


class ToolAttachmentRow(AttachmentBase):  # type: ignore[valid-type,misc]
    __tablename__ = "tool_attachments"

    id = Column(String(32), primary_key=True)
    thread_id = Column(String(100), nullable=False, index=True)
    kind = Column(String(32), nullable=False)
    payload = Column(Text, nullable=False)  # JSON; Postgres compresses it out of line
    size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


def reference_for(kind: str, attachment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The payload without its rows, plus the id that fetches them."""
    ref = {k: v for k, v in payload.items() if k != "rows"}
    ref.update(id=attachment_id, kind=kind)
    return ref


def referenced_ids(messages: Iterable[Any]) -> set[str]:
    """The stored attachment ids a thread's messages still refer to."""
    ids: set[str] = set()
    for message in messages:
        artifact = getattr(message, "artifact", None)
        content = artifact.get("structured_content") if isinstance(artifact, dict) else None
        if not isinstance(content, dict):
            continue
        for kind in STORED_KINDS:
            ref = content.get(kind + REF_SUFFIX)
            if isinstance(ref, dict) and isinstance(ref.get("id"), str):
                ids.add(ref["id"])
    return ids


class ToolAttachmentStore:
    """Shares the thread-owner store's engine and pool: one pool per worker,
    and the same database the access check reads."""

    def __init__(self) -> None:
        self._session_factory: sessionmaker[Session] | None = None

    def _sessions(self) -> sessionmaker[Session] | None:
        if self._session_factory is None:
            factory = get_thread_owner_store().session_factory()
            if factory is None:
                return None
            AttachmentBase.metadata.create_all(factory.kw["bind"])
            self._session_factory = factory
        return self._session_factory

    def put(self, thread_id: str, kind: str, payload: dict[str, Any]) -> str | None:
        """Store the payload and return its id, or None when it can't be stored
        (no database, no owner row for the thread, or any database error).
        Never raises: the tool's data was already fetched."""
        try:
            sessions = self._sessions()
            if sessions is None or get_thread_owner_store().resolve_owner(thread_id) is None:
                return None
            text = json.dumps(payload, separators=(",", ":"))
            attachment_id = secrets.token_hex(16)
            with sessions() as session, session.begin():
                session.add(
                    ToolAttachmentRow(
                        id=attachment_id,
                        thread_id=thread_id,
                        kind=kind,
                        payload=text,
                        size_bytes=len(text.encode()),
                    )
                )
            return attachment_id
        except Exception:
            logger.warning("couldn't store a %s attachment", kind, exc_info=True)
            return None

    def get(self, thread_id: str, attachment_id: str) -> str | None:
        """The stored JSON, only when it belongs to this thread. A database
        error reads as not found: the client says the table didn't load and
        offers Retry."""
        try:
            sessions = self._sessions()
            if sessions is None:
                return None
            with sessions() as session:
                return session.execute(
                    select(ToolAttachmentRow.payload).where(
                        ToolAttachmentRow.id == attachment_id,
                        ToolAttachmentRow.thread_id == thread_id,
                    )
                ).scalar_one_or_none()
        except Exception:
            logger.warning("couldn't read an attachment", exc_info=True)
            return None

    def ids_for(self, thread_id: str) -> set[str]:
        """This thread's stored ids; none on a database error, so the cleanup skips."""
        try:
            sessions = self._sessions()
            if sessions is None:
                return set()
            with sessions() as session:
                query: Select[Any] = select(ToolAttachmentRow.id).where(
                    ToolAttachmentRow.thread_id == thread_id
                )
                return {str(i) for i in session.scalars(query)}
        except Exception:
            logger.warning("couldn't list attachments", exc_info=True)
            return set()

    def keep_only(self, thread_id: str, keep: set[str]) -> int:
        """Delete this thread's attachments that aren't in ``keep``. Returns how many went."""
        sessions = self._sessions()
        if sessions is None:
            return 0
        with sessions() as session, session.begin():
            statement = delete(ToolAttachmentRow).where(ToolAttachmentRow.thread_id == thread_id)
            if keep:
                statement = statement.where(ToolAttachmentRow.id.not_in(keep))
            result = session.execute(statement)
            return int(getattr(result, "rowcount", 0) or 0)


# What the model reads when a table couldn't be stored. get_raw_data's text
# said the table is attached, which is now untrue.
NOT_ATTACHED_NOTE = (
    "Note: the full table couldn't be attached for the user, so there is no data "
    "panel or download for it. Answer from the summary above, and say the full "
    "table isn't available this time."
)


def externalize(
    structured: dict[str, Any], thread_id: str | None, store: ToolAttachmentStore | None = None
) -> tuple[dict[str, Any], bool]:
    """structuredContent with each stored kind replaced by its reference, and
    whether a payload had to be dropped.

    A table is never left inline, where every later checkpoint would copy it.
    Without a thread to store it under, or when the write fails, it is dropped
    and the caller tells the model so.
    """
    store = store or get_tool_attachment_store()
    out = dict(structured)
    dropped = False
    for kind in STORED_KINDS:
        payload = out.pop(kind, None)
        if payload is None:
            continue
        attachment_id = (
            store.put(thread_id, kind, payload) if thread_id and isinstance(payload, dict) else None
        )
        if attachment_id is None:
            dropped = True
            continue
        out[kind + REF_SUFFIX] = reference_for(kind, attachment_id, payload)
    return out, dropped


_store: ToolAttachmentStore | None = None


def get_tool_attachment_store() -> ToolAttachmentStore:
    global _store
    if _store is None:
        _store = ToolAttachmentStore()
    return _store
