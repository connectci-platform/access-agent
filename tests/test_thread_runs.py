import asyncio

import pytest
from fastapi import HTTPException

from src.api.thread_runs import (
    acquire_thread_run,
    cancel_run,
    held_run,
    release_thread_run,
)


@pytest.mark.asyncio
async def test_second_acquire_on_same_thread_409():
    acquire_thread_run("t1", "r1")  # synchronous acquire in the handler
    try:
        with pytest.raises(HTTPException) as ei:
            acquire_thread_run("t1", "r2")
        assert ei.value.status_code == 409
    finally:
        release_thread_run("t1", "r1")
    acquire_thread_run("t1", "r3")
    release_thread_run("t1", "r3")  # freed


@pytest.mark.asyncio
async def test_held_run_releases_on_exception():
    acquire_thread_run("t2", "r1")
    with pytest.raises(ValueError):
        async with held_run("t2", "r1"):  # registers current task, releases in finally
            raise ValueError("boom")
    acquire_thread_run("t2", "r2")
    release_thread_run("t2", "r2")  # freed by held_run finally


@pytest.mark.asyncio
async def test_cancel_run_cancels_task():
    started = asyncio.Event()

    async def work():
        acquire_thread_run("t3", "r1")
        async with held_run("t3", "r1"):  # captures asyncio.current_task() internally
            started.set()
            await asyncio.sleep(100)

    task = asyncio.create_task(work())
    await started.wait()
    assert cancel_run("t3", "r1") is True
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_run_no_task_registered_returns_false():
    """Lock acquired on the thread but no cancellable task registered yet (the
    window between acquire_thread_run and held_run entering) → cancel_run returns
    False rather than raising."""
    acquire_thread_run("t-notask", "run-x")  # active mapping set, no task registered
    try:
        assert cancel_run("t-notask", "run-x") is False
    finally:
        release_thread_run("t-notask", "run-x")


@pytest.mark.asyncio
async def test_cancel_run_rejects_wrong_thread():
    """A run_id active on thread B must not be cancellable via a different thread_id.

    run_ids are handed to clients in the metadata SSE event; without the
    thread-scope check, a caller who owns thread A and observed B's run_id could
    cancel B's run. cancel_run must verify the run belongs to the given thread.
    """
    started = asyncio.Event()

    async def work_on_b():
        acquire_thread_run("thread-b", "run-b")
        async with held_run("thread-b", "run-b"):
            started.set()
            await asyncio.sleep(100)

    task = asyncio.create_task(work_on_b())
    await started.wait()

    # Attacker owns thread-a, presents B's real run_id against thread-a.
    assert cancel_run("thread-a", "run-b") is False
    assert not task.done()  # B's run was NOT cancelled

    # The legitimate owner (correct thread_id) still cancels.
    assert cancel_run("thread-b", "run-b") is True
    with pytest.raises(asyncio.CancelledError):
        await task
