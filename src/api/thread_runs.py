# In-process per-thread run registry. SINGLE-PROCESS ONLY: prod runs one uvicorn
# process (Dockerfile CMD, no --workers; no replicas in docker-compose.prod.yml).
# Under multiple workers/replicas this stops enforcing one-run-per-thread — swap
# the store for a Redis lock (REDIS_URL is already wired) at that point. This is
# a deliberate, documented assumption, not an oversight.
#
# Acquire/release are SPLIT on purpose: acquire runs in the endpoint coroutine so a
# 409 is a real HTTP status; a 409 raised inside the streaming generator would land
# after Starlette flushed 200. release runs in the generator's finally.
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException

_active_by_thread: dict[str, str] = {}  # thread_id -> run_id
_task_by_run: dict[str, "asyncio.Task[object]"] = {}  # run_id -> running task


def acquire_thread_run(thread_id: str, run_id: str) -> None:
    """Synchronous acquire (call in the handler, before StreamingResponse). 409 if busy."""
    if thread_id in _active_by_thread:
        raise HTTPException(status_code=409, detail="A run is already active on this thread")
    _active_by_thread[thread_id] = run_id


def release_thread_run(thread_id: str, run_id: str) -> None:
    if _active_by_thread.get(thread_id) == run_id:
        _active_by_thread.pop(thread_id, None)
    _task_by_run.pop(run_id, None)


@asynccontextmanager
async def held_run(thread_id: str, run_id: str) -> AsyncIterator[None]:
    """Inside the generator: register the cancellable task, guarantee release.

    Captures asyncio.current_task() INTERNALLY (not caller-supplied) so there is no
    locked-but-uncancellable window (Finding 4). Does NOT acquire — acquisition
    already happened synchronously in the handler.
    """
    task = asyncio.current_task()
    assert task is not None
    _task_by_run[run_id] = task
    try:
        yield
    finally:
        release_thread_run(thread_id, run_id)


def cancel_run(thread_id: str, run_id: str) -> bool:
    """Cancel run_id only if it is the run currently active on thread_id.

    The run_id alone is not sufficient authority: run_ids are handed to clients
    in the `metadata` SSE event, so a caller who owns thread A and observed a
    run_id on thread B must not be able to cancel B's run by presenting B's
    run_id against A. Verifying the run belongs to the path thread closes that
    cross-thread cancel (caller ownership of thread_id is already checked in the
    handler). Returns False (→ handler 404) when the run isn't active here.
    """
    if _active_by_thread.get(thread_id) != run_id:
        return False
    task = _task_by_run.get(run_id)
    if task is None:
        return False
    task.cancel()
    return True


@asynccontextmanager
async def run_with_timeout(seconds: int) -> AsyncIterator[None]:
    """Bound a turn. MUST be used inside the generator, around the `async for`, so the
    timeout binds stream CONSUMPTION (not just creation). On expiry asyncio.timeout
    raises TimeoutError; the generator's finally (via held_run) still releases the lock.
    """
    async with asyncio.timeout(seconds):
        yield
