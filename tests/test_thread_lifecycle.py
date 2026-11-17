"""In-flight writes and deleted threads (D8-3126).

A deleted conversation must not come back: a write that starts after the
delete sees the thread as deleted and skips, and the delete waits for writes
already running before it removes anything.
"""

import asyncio

import pytest

from src import thread_lifecycle
from src.config import settings
from src.thread_lifecycle import (
    ThreadDeletedError,
    is_deleted,
    mark_deleted,
    settle,
    unmark_deleted,
    writing,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'l.db'}", raising=False)
    thread_lifecycle._reset_for_test()
    yield
    thread_lifecycle._reset_for_test()


async def test_a_write_on_a_deleted_thread_is_refused():
    mark_deleted(["t1"])
    with pytest.raises(ThreadDeletedError):
        async with writing("t1"):
            pytest.fail("the write ran")
    # Other threads are unaffected.
    async with writing("t2"):
        pass


async def test_settle_waits_for_writes_already_running():
    started = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def writer():
        async with writing("t1"):
            started.set()
            await release.wait()
            order.append("write done")

    task = asyncio.create_task(writer())
    await started.wait()
    mark_deleted(["t1"])
    waiter = asyncio.create_task(settle(["t1"], timeout=1))
    await asyncio.sleep(0)
    assert not waiter.done()
    release.set()
    assert await waiter is True
    order.append("settled")
    await task
    assert order == ["write done", "settled"]


async def test_settle_reports_a_write_that_outlasts_the_timeout():
    release = asyncio.Event()
    started = asyncio.Event()

    async def writer():
        async with writing("t1"):
            started.set()
            await release.wait()

    task = asyncio.create_task(writer())
    await started.wait()
    assert await settle(["t1"], timeout=0.01) is False
    release.set()
    await task
    assert await settle(["t1"], timeout=0.01) is True


async def test_settle_with_nothing_running_is_immediate():
    assert await settle(["t1", "t2"], timeout=0.01) is True


async def test_unmark_lets_writes_through_again():
    mark_deleted(["t1"])
    unmark_deleted(["t1"])
    async with writing("t1"):
        pass


def test_is_deleted_reads_the_stored_record_after_a_restart():
    thread_lifecycle.record_deleted_sync(["t-old"])
    thread_lifecycle._reset_memory_for_test()
    assert is_deleted("t-old") is True
    assert is_deleted("t-new") is False


def test_is_deleted_without_a_database_uses_memory(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    thread_lifecycle._reset_for_test()
    assert is_deleted("t1") is False
    mark_deleted(["t1"])
    assert is_deleted("t1") is True


async def test_two_writes_at_once_settle_only_when_both_finish():
    first, second = asyncio.Event(), asyncio.Event()
    in_both = asyncio.Event()

    async def write(release):
        async with writing("t1"):
            in_both.set()
            await release.wait()

    a = asyncio.create_task(write(first))
    b = asyncio.create_task(write(second))
    await in_both.wait()
    await asyncio.sleep(0)
    first.set()
    await a
    assert await settle(["t1"], timeout=0.01) is False
    second.set()
    await b
    assert await settle(["t1"], timeout=0.01) is True


def test_recording_nothing_or_without_a_database_is_a_no_op(monkeypatch):
    thread_lifecycle.record_deleted_sync([])
    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    thread_lifecycle._reset_for_test()
    thread_lifecycle.record_deleted_sync(["t1"])


def test_an_unreachable_database_reads_as_not_deleted(monkeypatch):
    import sqlalchemy as sa

    def broken():
        raise sa.exc.OperationalError("SELECT", {}, Exception("down"))

    monkeypatch.setattr(thread_lifecycle, "_get_engine", broken)
    assert is_deleted("t1") is False
    mark_deleted(["t1"])
    assert is_deleted("t1") is True


def test_a_table_another_worker_created_first_is_fine(monkeypatch):
    import sqlalchemy as sa

    thread_lifecycle._get_engine()
    thread_lifecycle._tables_ready = False

    def racing(*a, **k):
        raise sa.exc.ProgrammingError("CREATE TABLE", {}, Exception("exists"))

    monkeypatch.setattr(thread_lifecycle.LifecycleBase.metadata, "create_all", racing)
    assert thread_lifecycle._get_engine() is not None


def test_a_table_that_still_isnt_there_is_an_error(monkeypatch, tmp_path):
    import sqlalchemy as sa

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'n.db'}", raising=False)
    thread_lifecycle._reset_for_test()

    def failing(*a, **k):
        raise sa.exc.OperationalError("CREATE TABLE", {}, Exception("denied"))

    monkeypatch.setattr(thread_lifecycle.LifecycleBase.metadata, "create_all", failing)
    with pytest.raises(sa.exc.OperationalError):
        thread_lifecycle._get_engine()
