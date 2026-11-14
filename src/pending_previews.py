"""Write previews waiting for the user's Confirm click.

A write tool returns a preview until it is called with ``confirmed: true``.
The agent stores the preview's exact tool and arguments here under a random
id, and the Confirm button sends only that id. The agent then claims the id
once and replays the stored call itself, so what runs is what the user saw:
the model can't alter the arguments between preview and confirmation, and a
double click can't write twice. A preview belongs to the user who saw it: the
acting user's hash is stored with it, and only that user can claim it.

The table has its own metadata, created on first use, like
src/tool_attachments.py, and shares the thread-owner store's engine. Failures
are logged, never raised: a preview that can't be stored is shown without a
button, and a claim that fails says why.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, String, Text, Update, delete, select, update
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
    # sha256 of the acting user's sub; None for an anonymous caller.
    acting_user = Column(String(64), nullable=True)


class ClaimStatus(StrEnum):
    CLAIMED = "claimed"
    # This user already confirmed it; the write ran (or was attempted) then.
    ALREADY_USED = "already_used"
    # Unknown, another thread's or user's, or older than 24 hours.
    EXPIRED_OR_UNKNOWN = "expired_or_unknown"
    ERROR = "error"


@dataclass(frozen=True)
class Claim:
    status: ClaimStatus
    tool: str | None = None
    args: dict[str, Any] | None = None


def _user_key(acting_user: str | None) -> str | None:
    if acting_user is None:
        return None
    return hashlib.sha256(acting_user.encode()).hexdigest()


def _owned_by(acting_user: str | None) -> Any:
    key = _user_key(acting_user)
    if key is None:
        return PendingPreviewRow.acting_user.is_(None)
    return PendingPreviewRow.acting_user == key


class PendingPreviewStore:
    def __init__(self) -> None:
        self._session_factory: sessionmaker[Session] | None = None
        # Concurrent first calls would otherwise both run create_all, and the
        # loser's CREATE TABLE fails.
        self._init_lock = threading.Lock()

    def _sessions(self) -> sessionmaker[Session] | None:
        if self._session_factory is None:
            with self._init_lock:
                if self._session_factory is None:
                    factory = get_thread_owner_store().session_factory()
                    if factory is None:
                        return None
                    PreviewBase.metadata.create_all(factory.kw["bind"])
                    self._session_factory = factory
        return self._session_factory

    def record(
        self,
        thread_id: str,
        tool: str,
        args: dict[str, Any],
        acting_user: str | None = None,
    ) -> str | None:
        """Store the call and return its id, or None when it can't be stored.

        A run with no thread can't be confirmed (the claim is thread-scoped),
        so nothing is stored for it. The thread's expired previews go in the
        same transaction, so the table holds at most a day of each thread's
        previews."""
        if not thread_id:
            return None
        try:
            sessions = self._sessions()
            if sessions is None:
                return None
            preview_id = uuid.uuid4().hex
            with sessions() as session, session.begin():
                session.execute(
                    delete(PendingPreviewRow).where(
                        PendingPreviewRow.thread_id == thread_id,
                        PendingPreviewRow.created_at <= _now() - PREVIEW_TTL,  # type: ignore[arg-type]
                    )
                )
                session.add(
                    PendingPreviewRow(
                        id=preview_id,
                        thread_id=thread_id,
                        tool=tool,
                        args=json.dumps(args),
                        acting_user=_user_key(acting_user),
                    )
                )
            return preview_id
        except Exception:
            logger.warning("couldn't store a pending preview for %s", tool, exc_info=True)
            return None

    def claim(self, thread_id: str, preview_id: str, acting_user: str | None = None) -> Claim:
        """Take the stored call, once. One UPDATE ... RETURNING decides the winner,
        so concurrent clicks can't both get it. A losing claim then reads the row
        only to say why: ALREADY_USED when this user's preview in this thread was
        claimed before, EXPIRED_OR_UNKNOWN when the id is unknown, another
        thread's or user's, or older than 24 hours, and ERROR on any failure."""
        try:
            sessions = self._sessions()
            if sessions is None:
                return Claim(ClaimStatus.ERROR)
            now = _now()
            owned = (
                PendingPreviewRow.id == preview_id,
                PendingPreviewRow.thread_id == thread_id,
                _owned_by(acting_user),
            )
            statement: Update = (
                update(PendingPreviewRow)
                .where(
                    *owned,
                    PendingPreviewRow.claimed_at.is_(None),
                    PendingPreviewRow.created_at > now - PREVIEW_TTL,  # type: ignore[arg-type]
                )
                .values(claimed_at=now)
                .returning(PendingPreviewRow.tool, PendingPreviewRow.args)
            )
            with sessions() as session, session.begin():
                row = session.execute(statement).first()
                if row is None:
                    claimed_at = session.execute(
                        select(PendingPreviewRow.claimed_at).where(*owned)
                    ).first()
            if row is not None:
                return Claim(ClaimStatus.CLAIMED, str(row.tool), dict(json.loads(row.args)))
            if claimed_at is not None and claimed_at[0] is not None:
                return Claim(ClaimStatus.ALREADY_USED)
            return Claim(ClaimStatus.EXPIRED_OR_UNKNOWN)
        except Exception:
            logger.warning("couldn't claim a pending preview", exc_info=True)
            return Claim(ClaimStatus.ERROR)


_store: PendingPreviewStore | None = None


def get_pending_preview_store() -> PendingPreviewStore:
    global _store
    if _store is None:
        _store = PendingPreviewStore()
    return _store
