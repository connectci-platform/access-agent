import fakeredis
import pytest

from src.api.handoff_tokens import (
    _reset_store_for_test,
    exchange_handoff_token,
    mint_handoff_token,
)


@pytest.fixture(autouse=True)
def _mem_store(monkeypatch):
    # Force the in-memory path regardless of ambient REDIS_URL. `settings` is
    # constructed once at import, so patch the already-loaded value, not the env.
    from src import config

    monkeypatch.setattr(config.settings, "REDIS_URL", "", raising=False)
    _reset_store_for_test()


@pytest.fixture
def _redis_store(monkeypatch):
    # Route _get_store() down the Redis branch by setting REDIS_URL truthy,
    # then swap redis.Redis.from_url (used in _RedisStore.__init__) for a
    # fakeredis client so no real Redis server is required.
    import redis

    from src import config

    # _get_store() constructs a fresh _RedisStore (and thus calls from_url)
    # on every call, so the fake client must be shared across calls to
    # preserve state between mint and exchange.
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(config.settings, "REDIS_URL", "redis://fake:6379/0", raising=False)
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **kw: fake_client))


def test_mint_then_exchange_returns_thread_and_owner():
    tok = mint_handoff_token("t-1", "ownerhash", ttl_seconds=60)
    assert exchange_handoff_token(tok) == (
        "t-1",
        "ownerhash",
    )  # owner_hash round-trips for the endpoint's identity check


def test_single_use_second_exchange_fails():
    tok = mint_handoff_token("t-1", "ownerhash")
    assert exchange_handoff_token(tok) is not None
    assert exchange_handoff_token(tok) is None  # consumed


def test_unknown_or_expired_token_returns_none():
    assert exchange_handoff_token("never-minted") is None


def test_expired_token_returns_none(monkeypatch):
    tok = mint_handoff_token("t-1", "ownerhash", ttl_seconds=60)

    import time

    # handoff_tokens calls time.monotonic() via `import time`, so patching the
    # time module directly advances its clock without reaching through the
    # handoff_tokens namespace (which mypy rejects as a non-exported attr).
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 61)
    assert exchange_handoff_token(tok) is None


def test_redis_mint_then_exchange_returns_thread_and_owner(_redis_store):
    tok = mint_handoff_token("t-1", "ownerhash", ttl_seconds=60)
    assert exchange_handoff_token(tok) == ("t-1", "ownerhash")


def test_redis_single_use_second_exchange_fails(_redis_store):
    tok = mint_handoff_token("t-1", "ownerhash")
    assert exchange_handoff_token(tok) is not None
    assert exchange_handoff_token(tok) is None  # consumed via GETDEL


def test_redis_unknown_token_returns_none(_redis_store):
    assert exchange_handoff_token("never-minted") is None
