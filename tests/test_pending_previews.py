"""A write preview is stored by id and claimed once for confirmation (D8-3088)."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from src.pending_previews import PendingPreviewRow, PendingPreviewStore, get_pending_preview_store

ARGS = {"title": "Intro", "dates": ["2026-12-01"], "n": 3}


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
    assert store.claim("t1", preview_id) == ("create_event", ARGS)


def test_a_second_claim_finds_nothing():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    assert store.claim("t1", preview_id) is not None
    assert store.claim("t1", preview_id) is None


def test_another_threads_claim_finds_nothing_and_leaves_it_claimable():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    assert store.claim("t2", preview_id) is None
    assert store.claim("t1", preview_id) == ("create_event", ARGS)


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

    assert store.claim("t1", preview_id) is None


def test_concurrent_claims_yield_exactly_one_success():
    store = get_pending_preview_store()
    preview_id = store.record("t1", "create_event", ARGS)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.claim("t1", preview_id), range(8)))

    assert [r for r in results if r is not None] == [("create_event", ARGS)]


def test_without_a_database_record_returns_none(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    store = PendingPreviewStore()

    assert store.record("t1", "create_event", ARGS) is None
    assert store.claim("t1", "x" * 32) is None


def test_a_database_error_returns_none_without_raising(monkeypatch):
    store = get_pending_preview_store()
    monkeypatch.setattr(store, "_sessions", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    assert store.record("t1", "create_event", ARGS) is None
    assert store.claim("t1", "x" * 32) is None
