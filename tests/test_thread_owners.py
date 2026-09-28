import pytest

from src.thread_owners import get_thread_owner_store


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    # Point the store at a temp sqlite file. The store reads settings.DATABASE_URL
    # (not os.environ), so patch the setting, not the env var.
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
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


def test_postgres_url_rewritten_to_psycopg_v3_driver(monkeypatch):
    """A bare postgresql:// DATABASE_URL must connect via the psycopg (v3) driver.

    Guards Finding C1: the project ships psycopg v3 only (no psycopg2). SQLAlchemy
    maps a bare postgresql:// URL to psycopg2 → ModuleNotFoundError at connect in
    prod. _ensure() must rewrite the scheme to postgresql+psycopg://. CI runs on
    sqlite so nothing else catches a regression here; assert on the engine URL the
    store builds without needing a live Postgres.
    """
    import src.thread_owners as m
    from src.config import settings

    m._store = None
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pw@dbhost:5432/langgraph")

    captured = {}

    def _fake_create_engine(url, *a, **k):
        captured["url"] = url
        raise RuntimeError("stop before real connect")  # no live DB in CI

    monkeypatch.setattr(m, "create_engine", _fake_create_engine)

    store = get_thread_owner_store()
    with pytest.raises(RuntimeError):
        store._ensure()

    # v3 driver, not the bare postgresql:// that SQLAlchemy would map to psycopg2.
    assert captured["url"].startswith("postgresql+psycopg://")


def test_resolve_owner_uninitialized_returns_none(monkeypatch):
    """No DATABASE_URL at all: _ensure() fails, resolve_owner short-circuits."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    import src.thread_owners as m

    m._store = None
    store = get_thread_owner_store()
    assert store.resolve_owner("whatever") is None


def test_claim_thread_uninitialized_returns_false(monkeypatch):
    """No DATABASE_URL: _ensure() fails, claim_thread reports not-claimed rather
    than raising (degrades gracefully; ownership just isn't recorded)."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    import src.thread_owners as m

    m._store = None
    store = get_thread_owner_store()
    assert store.claim_thread("whatever", "alice@x") is False
