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


def test_empty_string_claimant_stored_as_anonymous_not_corrupted():
    store = get_thread_owner_store()
    store.claim_thread("t-empty", "")
    owner = store.resolve_owner("t-empty")
    assert (
        owner.was_authenticated is False
    )  # NOT True-with-null-hash (the corrupted state the fix prevents)
    assert owner.user_hash is None


def test_anon_owned_thread_grants_access_to_any_caller():
    """check_access on an anon-owned thread returns True regardless of caller —
    authed-only endpoints already 401 anonymous callers upstream, so this
    branch relies on that gate rather than re-checking identity here."""
    store = get_thread_owner_store()
    store.claim_thread("t-anon", None)
    assert store.check_access("t-anon", "anyone@x") is True
    assert store.check_access("t-anon", None) is True


def test_resolve_owner_uninitialized_returns_none(monkeypatch):
    """No DATABASE_URL at all: _ensure() fails, resolve_owner short-circuits."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import src.thread_owners as m

    m._store = None
    store = get_thread_owner_store()
    assert store.resolve_owner("whatever") is None
