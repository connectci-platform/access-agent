"""A write preview is stored by id and claimed once for confirmation (D8-3088)."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from src.pending_previews import (
    Claim,
    ClaimStatus,
    PendingPreviewRow,
    PendingPreviewStore,
    get_pending_preview_store,
)

ARGS = {"title": "Intro", "dates": ["2026-12-01"], "n": 3}
CLAIMED = Claim(ClaimStatus.CLAIMED, "create_event", ARGS)
USED = Claim(ClaimStatus.ALREADY_USED)
GONE = Claim(ClaimStatus.EXPIRED_OR_UNKNOWN)
ERROR = Claim(ClaimStatus.ERROR)


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.pending_previews as previews
    import src.thread_owners as owners

    owners._store = None
    previews._store = None


def test_a_recorded_preview_claims_back_its_tool_and_args():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    assert preview_id is not None and len(preview_id) == 32
    assert store.claim("t1", preview_id) == CLAIMED


def test_a_second_claim_finds_nothing():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    assert store.claim("t1", preview_id) == CLAIMED
    assert store.claim("t1", preview_id) == USED


def test_another_threads_claim_finds_nothing_and_leaves_it_claimable():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    assert store.claim("t2", preview_id) == GONE
    assert store.claim("t1", preview_id) == CLAIMED


def test_a_preview_older_than_a_day_cant_be_claimed():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24, minutes=1)
    with store._sessions()() as session, session.begin():
        session.execute(
            update(PendingPreviewRow)
            .where(PendingPreviewRow.id == preview_id)
            .values(created_at=old)
        )

    assert store.claim("t1", preview_id) == GONE


def test_recording_removes_the_threads_expired_previews_only():
    store = get_pending_preview_store()
    old_id = store.record("t1", "create_event", ARGS)
    other_id = store.record("t2", "create_event", ARGS)
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24, minutes=1)
    with store._sessions()() as session, session.begin():
        session.execute(update(PendingPreviewRow).values(created_at=old))

    fresh_id = store.record("t1", "create_event", ARGS)

    with store._sessions()() as session:
        ids = set(session.scalars(select(PendingPreviewRow.id)))
    assert ids == {other_id, fresh_id}
    assert old_id not in ids


def test_concurrent_claims_yield_exactly_one_success():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.claim("t1", preview_id), range(8)))

    assert [r for r in results if r.status is ClaimStatus.CLAIMED] == [CLAIMED]
    assert [r for r in results if r != CLAIMED] == [USED] * 7


def test_without_a_database_record_returns_none(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    store = PendingPreviewStore()

    assert store.record("t1", "create_event", ARGS) is None
    assert store.claim("t1", "x" * 32) == ERROR


def test_a_database_error_returns_none_without_raising(monkeypatch):
    store = get_pending_preview_store()
    monkeypatch.setattr(store, "_sessions", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    assert store.record("t1", "create_event", ARGS) is None
    assert store.claim("t1", "x" * 32) == ERROR


def test_an_unknown_id_is_expired_or_unknown():
    assert get_pending_preview_store().claim("t1", "0" * 32) == GONE


def test_a_preview_is_claimable_only_by_the_user_who_saw_it():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS, acting_user="alice")

    assert store.claim("t1", preview_id, acting_user="bob") == GONE
    assert store.claim("t1", preview_id) == GONE
    assert store.claim("t1", preview_id, acting_user="alice") == CLAIMED
    assert store.claim("t1", preview_id, acting_user="bob") == GONE
    assert store.claim("t1", preview_id, acting_user="alice") == USED


def test_an_anonymous_preview_is_not_claimable_by_a_signed_in_user():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    assert store.claim("t1", preview_id, acting_user="alice") == GONE
    assert store.claim("t1", preview_id) == CLAIMED


def test_the_acting_user_is_stored_hashed():
    store = get_pending_preview_store()
    store.record("t1", "create_event", ARGS, acting_user="alice@access-ci.org")

    with store._sessions()() as session:
        [stored] = session.scalars(select(PendingPreviewRow.acting_user))
    assert stored is not None
    assert "alice" not in stored


def test_a_preview_with_no_thread_is_not_recorded():
    store = get_pending_preview_store()

    assert store.record("", "create_event", ARGS, acting_user="alice") is None
    with store._sessions()() as session:
        assert list(session.scalars(select(PendingPreviewRow.id))) == []


def test_concurrent_first_use_records_every_preview():
    store = get_pending_preview_store()

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: store.record("t1", "create_event", ARGS), range(8)))

    assert all(i is not None for i in ids)
