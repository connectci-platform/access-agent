import pytest

from src.thread_owners import get_thread_owner_store


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    # Mirror how turn_reporter tests point DATABASE_URL at a temp sqlite file.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    import src.thread_owners as m

    m._store = None  # reset singleton


def test_first_writer_claims_second_is_rejected():
    store = get_thread_owner_store()
    assert store.claim_thread("t1", "alice@x") is True
    assert store.claim_thread("t1", "bob@x") is False  # UNIQUE(thread_id) → not first


def test_owner_passes_nonowner_404():
    store = get_thread_owner_store()
    store.claim_thread("t2", "alice@x")
    assert store.check_access("t2", "alice@x") is True
    assert store.check_access("t2", "bob@x") is False


def test_unknown_thread_denied():
    store = get_thread_owner_store()
    assert store.check_access("never-created", "anyone@x") is False
    assert store.resolve_owner("never-created") is None
