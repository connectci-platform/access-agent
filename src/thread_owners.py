from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple

from sqlalchemy import Boolean, Column, DateTime, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)
ThreadOwnerBase = declarative_base()


class ThreadOwner(NamedTuple):
    was_authenticated: bool
    user_hash: str | None


class ThreadOwnerRow(ThreadOwnerBase):  # type: ignore[valid-type,misc]
    __tablename__ = "thread_owners"

    thread_id = Column(String(100), primary_key=True)  # PK == UNIQUE first-writer-wins
    user_hash = Column(String(16), index=True)
    was_authenticated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


def _hash_user(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


class ThreadOwnerStore:
    """Durable, transactional thread-ownership record with atomic first-writer-wins.

    This is the security boundary for thread access — deliberately NOT sourced
    from turn_reports (best-effort mirror) or checkpoint metadata (undocumented
    custom keys).
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def _ensure(self) -> bool:
        if self._session_factory is not None:
            return True
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            return False
        self._engine = create_engine(db_url)
        ThreadOwnerBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        return True

    def claim_thread(self, thread_id: str, acting_user: str | None) -> bool:
        if not self._ensure() or self._session_factory is None:
            return False
        session = self._session_factory()
        try:
            session.add(
                ThreadOwnerRow(
                    thread_id=thread_id,
                    user_hash=_hash_user(acting_user),
                    was_authenticated=acting_user is not None,
                )
            )
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False  # already owned — not the first writer
        finally:
            session.close()

    def resolve_owner(self, thread_id: str) -> ThreadOwner | None:
        if not self._ensure() or self._session_factory is None:
            return None
        session = self._session_factory()
        try:
            row = session.get(ThreadOwnerRow, thread_id)
            if row is None:
                return None
            return ThreadOwner(bool(row.was_authenticated), row.user_hash)  # type: ignore[arg-type]
        finally:
            session.close()

    def check_access(self, thread_id: str, acting_user: str | None) -> bool:
        owner = self.resolve_owner(thread_id)
        if owner is None:
            return False
        if owner.was_authenticated and owner.user_hash:
            return _hash_user(acting_user) == owner.user_hash
        return True  # anon-owned (authed-only endpoints 401 anon upstream)


_store: ThreadOwnerStore | None = None


def get_thread_owner_store() -> ThreadOwnerStore:
    global _store
    if _store is None:
        _store = ThreadOwnerStore()
    return _store
