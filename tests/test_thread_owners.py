import pytest

from src.thread_owners import ThreadAccess, get_thread_owner_store


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
    assert store.access("t2", "alice@x") is ThreadAccess.OWNER
    assert store.access("t2", "bob@x") is ThreadAccess.DENIED


def test_unknown_thread_denied():
    store = get_thread_owner_store()
    assert store.access("never-created", "anyone@x") is ThreadAccess.NEW
    assert store.resolve_owner("never-created") is None


def test_empty_string_claimant_stored_as_anonymous_not_corrupted():
    store = get_thread_owner_store()
    store.claim_thread("t-empty", "")
    owner = store.resolve_owner("t-empty")
    assert (
        owner.was_authenticated is False
    )  # NOT True-with-null-hash (the corrupted state the fix prevents)
    assert owner.user_hash is None


def test_anon_thread_with_no_key_on_record_is_keyless_for_every_caller():
    store = get_thread_owner_store()
    store.claim_thread("t-anon", None)
    assert store.access("t-anon", "anyone@x") is ThreadAccess.KEYLESS
    assert store.access("t-anon", None) is ThreadAccess.KEYLESS
    assert store.access("t-anon", None, "k" * 43) is ThreadAccess.KEYLESS


def test_anon_thread_with_a_key_needs_that_key():
    store = get_thread_owner_store()
    store.claim_thread("t-keyed", None, "k" * 43)
    assert store.access("t-keyed", None, "k" * 43) is ThreadAccess.KEY_HOLDER
    assert store.access("t-keyed", "me@x", "k" * 43) is ThreadAccess.KEY_HOLDER
    assert store.access("t-keyed", None) is ThreadAccess.DENIED
    assert store.access("t-keyed", None, "o" * 43) is ThreadAccess.DENIED


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


WIDGET_ID = "qa_bot_session_00000000-0000-4000-8000-000000000001"


def test_widget_flip_moves_ownership_to_the_signed_in_user():
    import hashlib

    store = get_thread_owner_store()
    store.claim_thread(WIDGET_ID, None)
    assert store.upgrade_widget_session(WIDGET_ID, "me@x") is True
    owner = store.resolve_owner(WIDGET_ID)
    assert owner.was_authenticated is True
    assert owner.user_hash == hashlib.sha256(b"me@x").hexdigest()[:16]


def test_widget_flip_is_noop_on_authed_thread():
    import hashlib

    store = get_thread_owner_store()
    store.claim_thread(WIDGET_ID, "alice@x")
    assert store.upgrade_widget_session(WIDGET_ID, "bob@x") is False
    assert store.resolve_owner(WIDGET_ID).user_hash == hashlib.sha256(b"alice@x").hexdigest()[:16]


def test_second_concurrent_widget_flip_loses_cleanly():
    store = get_thread_owner_store()
    store.claim_thread(WIDGET_ID, None)
    assert store.upgrade_widget_session(WIDGET_ID, "first@x") is True
    assert store.upgrade_widget_session(WIDGET_ID, "second@x") is False


def test_widget_flip_refuses_any_other_id():
    store = get_thread_owner_store()
    store.claim_thread("00000000-0000-4000-8000-000000000001", None)
    store.claim_thread("wave3-persona", None)
    assert store.upgrade_widget_session("00000000-0000-4000-8000-000000000001", "me@x") is False
    assert store.upgrade_widget_session("wave3-persona", "me@x") is False


def test_widget_flip_refuses_a_thread_with_a_key():
    store = get_thread_owner_store()
    store.claim_thread(WIDGET_ID, None, "k" * 43)
    assert store.upgrade_widget_session(WIDGET_ID, "me@x") is False


def test_widget_flip_uninitialized_returns_false(monkeypatch):
    """No DATABASE_URL: _ensure() fails, the flip reports no-flip rather than
    raising."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    import src.thread_owners as m

    m._store = None
    store = get_thread_owner_store()
    assert store.upgrade_widget_session(WIDGET_ID, "alice@x") is False
    assert store.access(WIDGET_ID, "alice@x") is ThreadAccess.NEW
    assert store._stored_key_hash(WIDGET_ID) is None


def test_access_fails_closed_on_authed_row_with_null_hash():
    """Defense-in-depth: an authed row with a NULL user_hash (today unreachable
    since acting_user is truthy whenever was_authenticated is True) must deny
    access, even to a caller with no identity."""
    store = get_thread_owner_store()
    store._ensure()
    assert store._session_factory is not None
    session = store._session_factory()
    try:
        from src.thread_owners import ThreadOwnerRow

        session.add(ThreadOwnerRow(thread_id="t-corrupt", user_hash=None, was_authenticated=True))
        session.commit()
    finally:
        session.close()

    assert store.access("t-corrupt", "anyone@x") is ThreadAccess.DENIED
    assert store.access("t-corrupt", None) is ThreadAccess.DENIED


KEY = "k" * 43


def test_second_claim_cannot_replace_the_key():
    store = get_thread_owner_store()
    store.claim_thread("t-key2", None, KEY)
    assert store.claim_thread("t-key2", None, "o" * 43) is False
    assert store.claim_with_key("t-key2", "thief@x", "o" * 43) is False
    assert store.claim_with_key("t-key2", "me@x", KEY) is True


def test_a_key_outside_the_allowed_length_is_not_stored():
    store = get_thread_owner_store()
    store.claim_thread("t-short", None, "short")
    store.claim_thread("t-long", None, "k" * 129)
    assert store.claim_with_key("t-short", "me@x", "short") is False
    assert store.claim_with_key("t-long", "me@x", "k" * 129) is False


def test_authed_first_run_stores_no_key():
    store = get_thread_owner_store()
    store.claim_thread("t-authed", "alice@x", KEY)
    assert store.claim_with_key("t-authed", "thief@x", KEY) is False


def test_claim_with_key_refuses_a_row_without_a_key():
    store = get_thread_owner_store()
    store.claim_thread("t-legacy", None)
    assert store.claim_with_key("t-legacy", "me@x", KEY) is False
    assert store.claim_with_key("t-legacy", "me@x", None) is False


def test_claim_with_key_flips_once():
    store = get_thread_owner_store()
    store.claim_thread("t-once", None, KEY)
    assert store.claim_with_key("t-once", "me@x", KEY) is True
    assert store.claim_with_key("t-once", "thief@x", KEY) is False
    assert store.owned_by("t-once", "me@x") is True
    assert store.owned_by("t-once", "thief@x") is False


def test_owned_by_is_false_for_anon_and_unknown_threads():
    store = get_thread_owner_store()
    store.claim_thread("t-anon", None)
    assert store.owned_by("t-anon", "me@x") is False
    assert store.owned_by("t-missing", "me@x") is False


def test_key_store_is_inert_without_a_database(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    import src.thread_owners as m

    m._store = None
    store = get_thread_owner_store()
    assert store.claim_with_key("t", "me@x", KEY) is False
