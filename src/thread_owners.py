from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from sqlalchemy import Boolean, Column, DateTime, String, create_engine, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import settings

if TYPE_CHECKING:
    from sqlalchemy.engine import CursorResult, Engine

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
        if not settings.DATABASE_URL:
            return False
        # SQLAlchemy maps a bare `postgresql://` URL to the psycopg2 driver, which
        # is NOT installed (the project ships psycopg v3 only) — connect would
        # ModuleNotFoundError in prod. Rewrite to the psycopg (v3) driver, the
        # same idiom src/turn_reporter.py and src/usage_logger.py use. A sqlite
        # (or already-`+psycopg`) URL passes through untouched.
        db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
        self._engine = create_engine(db_url)
        ThreadOwnerBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        return True

    def claim_thread(self, thread_id: str, acting_user: str | None) -> bool:
        acting_user = acting_user or None  # empty string is anonymous, not authed-with-null-hash
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

    def upgrade_owner(self, thread_id: str, acting_user: str) -> bool:
        """Flip an anon-owned thread to authed ownership on login.

        A single atomic conditional UPDATE, never resolve-then-write: the
        WHERE was_authenticated == False guard is what makes two concurrent
        upgrades resolve cleanly. The first caller's UPDATE flips the row and
        commits; the second caller's WHERE then matches zero rows (the row is
        no longer was_authenticated == False) and it gets back False, with no
        read-modify-write race window between the two.
        """
        if not self._ensure() or self._session_factory is None:
            return False
        session = self._session_factory()
        try:
            # session.execute() of a Core update() is typed as the generic
            # Result[Any] but is always a CursorResult at runtime (it came
            # from a DML statement, not an ORM-entity select) — cast to reach
            # .rowcount, which Result's base type doesn't declare.
            result = cast(
                "CursorResult[Any]",
                session.execute(
                    update(ThreadOwnerRow)
                    .where(
                        ThreadOwnerRow.thread_id == thread_id,
                        ThreadOwnerRow.was_authenticated == False,  # noqa: E712
                    )
                    .values(user_hash=_hash_user(acting_user), was_authenticated=True)
                ),
            )
            session.commit()
            return bool(result.rowcount > 0)
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
        acting_user = acting_user or None  # empty string is anonymous, not authed-with-null-hash
        owner = self.resolve_owner(thread_id)
        if owner is None:
            return False
        if owner.was_authenticated:
            # Fail-closed: an authed row must carry a user_hash to grant access.
            # Today unreachable (claim_thread/upgrade_owner never set
            # was_authenticated=True without also hashing a truthy
            # acting_user) but a NULL-hash authed row must never fall through
            # to the anon branch's unconditional True — that would be a
            # fail-OPEN on the tenant-isolation security boundary.
            if not owner.user_hash:
                return False
            return _hash_user(acting_user) == owner.user_hash
        # Anon-owned thread: access is granted to ANY caller by design, and that
        # is safe because an anon thread is a capability. It is reachable only by
        # a caller who already presents its session_id, and that session_id is a
        # high-entropy, unguessable token (client-supplied qa_bot_session_<uuid>,
        # or the server's secrets.token_urlsafe fallback — see src/api/routes.py).
        # The unguessable id IS the credential; access does not derive from being
        # authenticated. This deliberately preserves the anon→login→keep-resuming
        # flow (a user who signs in mid-thread keeps their history). Do NOT "fix"
        # this by adding a caller-identity check — that would break resumption
        # without adding security, since the id already gates reachability.
        #
        # CONSEQUENCE THIS COMMENT CANNOT ENFORCE: because the id is the
        # credential for an anon thread, a session_id must be handled with the
        # same care as a password or bearer token everywhere it appears. Do NOT
        # put a session_id in a URL/query string, a redirect, a log line, a
        # support ticket, an analytics/error report, or any place it could be
        # observed — leaking one grants access to that conversation. If you ever
        # need to weaken that (e.g. a session_id genuinely must travel in a URL),
        # the anon capability model here must change first; it is load-bearing.
        return True


_store: ThreadOwnerStore | None = None


def get_thread_owner_store() -> ThreadOwnerStore:
    global _store
    if _store is None:
        _store = ThreadOwnerStore()
    return _store
