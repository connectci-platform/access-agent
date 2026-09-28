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
    assert cancel_run("r1") is True
    with pytest.raises(asyncio.CancelledError):
        await task
