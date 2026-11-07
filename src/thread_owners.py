from __future__ import annotations

import hashlib
import hmac
import logging
import re
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from sqlalchemy import Boolean, Column, DateTime, String, create_engine, exists, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import settings

if TYPE_CHECKING:
    from sqlalchemy.engine import CursorResult, Engine

logger = logging.getLogger(__name__)
ThreadOwnerBase = declarative_base()

MIN_CLAIM_KEY_LENGTH = 32
MAX_CLAIM_KEY_LENGTH = 128

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_FULLSCREEN_THREAD_ID = re.compile(_UUID, re.IGNORECASE)
# qa-bot-core's generateSessionId, the widget's only source of session_id.
_WIDGET_SESSION_ID = re.compile(f"qa_bot_session_{_UUID}", re.IGNORECASE)


def is_fullscreen_thread_id(thread_id: str) -> bool:
    """A bare uuid: what POST /threads mints and the client puts in ?threadId=."""
    return _FULLSCREEN_THREAD_ID.fullmatch(thread_id) is not None


def is_widget_session(thread_id: str) -> bool:
    return _WIDGET_SESSION_ID.fullmatch(thread_id) is not None


class ThreadAccess(Enum):
    """How a caller stands with a thread, before any route's rules apply."""

    NEW = auto()  # no such thread yet
    OWNER = auto()  # the signed-in owner
    KEY_HOLDER = auto()  # unclaimed, and the caller has its key
    KEYLESS = auto()  # unclaimed, with no key on record to check
    DENIED = auto()  # someone else's, or the wrong or missing key


class ThreadOwner(NamedTuple):
    was_authenticated: bool
    user_hash: str | None


class ThreadOwnerRow(ThreadOwnerBase):  # type: ignore[valid-type,misc]
    __tablename__ = "thread_owners"

    thread_id = Column(String(100), primary_key=True)  # PK == UNIQUE first-writer-wins
    user_hash = Column(String(16), index=True)
    was_authenticated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class ThreadClaimKeyRow(ThreadOwnerBase):  # type: ignore[valid-type,misc]
    """Hash of the secret the browser sent on a thread's first anonymous run.

    A fullscreen thread id sits in the page URL, so holding it proves nothing.
    This key never leaves the browser tab that started the conversation, which
    makes it the proof that a caller may take ownership. Its own table so that
    create_all deploys it: an ALTER on thread_owners that failed would take
    every ownership read down with it.
    """

    __tablename__ = "thread_claim_keys"

    thread_id = Column(String(100), primary_key=True)
    key_hash = Column(String(64), nullable=False)


def _hash_user(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def _hash_claim_key(claim_key: str | None) -> str | None:
    """The stored form of a claim key, or None for anything too weak to trust.

    SHA-256 alone is enough because the key is random, not a password.
    """
    if not claim_key or not MIN_CLAIM_KEY_LENGTH <= len(claim_key) <= MAX_CLAIM_KEY_LENGTH:
        return None
    return hashlib.sha256(claim_key.encode()).hexdigest()


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

    def claim_thread(
        self, thread_id: str, acting_user: str | None, claim_key: str | None = None
    ) -> bool:
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
            key_hash = _hash_claim_key(claim_key)
            if acting_user is None and key_hash is not None:
                # Same transaction as the owner row: only the first writer's
                # key is kept, so a later visitor can't replace it.
                session.add(ThreadClaimKeyRow(thread_id=thread_id, key_hash=key_hash))
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False  # already owned — not the first writer
        finally:
            session.close()

    def upgrade_widget_session(self, thread_id: str, acting_user: str) -> bool:
        """Flip a widget session to authed ownership when its user signs in.

        The one flip that needs no key: the widget has none, and its id stays
        in the browser. Anything else with no key on record stays locked.
        """
        if not is_widget_session(thread_id):
            return False
        has_key = exists().where(ThreadClaimKeyRow.thread_id == thread_id)
        return self._flip_to_authed(thread_id, acting_user, ~has_key)

    def claim_with_key(self, thread_id: str, acting_user: str, claim_key: str | None) -> bool:
        """Flip an anon-owned thread only on a stored, matching claim key."""
        key_hash = _hash_claim_key(claim_key)
        if key_hash is None:
            return False
        key_matches = exists().where(
            ThreadClaimKeyRow.thread_id == thread_id,
            ThreadClaimKeyRow.key_hash == key_hash,
        )
        return self._flip_to_authed(thread_id, acting_user, key_matches)

    def _flip_to_authed(self, thread_id: str, acting_user: str, allowed: Any) -> bool:
        """A single atomic conditional UPDATE, never resolve-then-write.

        The WHERE was_authenticated == False guard is what makes two concurrent
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
                        allowed,
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

    def owned_by(self, thread_id: str, acting_user: str) -> bool:
        """Whether acting_user is this thread's authenticated owner."""
        owner = self.resolve_owner(thread_id)
        return (
            owner is not None
            and owner.was_authenticated
            and owner.user_hash == _hash_user(acting_user)
        )

    def _stored_key_hash(self, thread_id: str) -> str | None:
        if not self._ensure() or self._session_factory is None:
            return None
        session = self._session_factory()
        try:
            row = session.get(ThreadClaimKeyRow, thread_id)
            return None if row is None else str(row.key_hash)
        finally:
            session.close()

    def access(
        self, thread_id: str, acting_user: str | None, claim_key: str | None = None
    ) -> ThreadAccess:
        """Where the caller stands. The rules live in src/api/thread_access.py.

        The thread id itself counts for nothing here: the fullscreen client
        keeps it in ?threadId=, so anyone can hold it.
        """
        owner = self.resolve_owner(thread_id)
        if owner is None:
            return ThreadAccess.NEW
        if owner.was_authenticated:
            # Fail-closed: an authed row must carry a user_hash to grant access.
            # Today unreachable (no write path sets was_authenticated=True
            # without hashing a truthy acting_user), but a NULL-hash authed row
            # must never match a caller with no identity.
            if owner.user_hash and owner.user_hash == _hash_user(acting_user or None):
                return ThreadAccess.OWNER
            return ThreadAccess.DENIED
        stored = self._stored_key_hash(thread_id)
        if stored is None:
            return ThreadAccess.KEYLESS
        presented = _hash_claim_key(claim_key) or ""
        if hmac.compare_digest(stored, presented):
            return ThreadAccess.KEY_HOLDER
        return ThreadAccess.DENIED


_store: ThreadOwnerStore | None = None


def get_thread_owner_store() -> ThreadOwnerStore:
    global _store
    if _store is None:
        _store = ThreadOwnerStore()
    return _store
