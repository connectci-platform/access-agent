"""Replay frozen prompts against /api/v1/query and collect N samples each."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

import httpx

from .sample import SampleResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .suite import SuiteItem

RUN_TAG = "redteam-suite-v1"


def redteam_headers(suite_version: str, run_id: str) -> dict[str, str]:
    return {
        "X-Redteam": RUN_TAG,
        "X-Redteam-Run-Id": run_id,
        "X-Redteam-Suite": suite_version,
    }


async def post_query(
    client: httpx.AsyncClient,
    base_url: str,
    prompt_text: str,
    session_id: str,
    headers: dict[str, str],
) -> SampleResult:
    """POST and parse the SSE stream. Returns SampleResult(text) on a real answer,
    SampleResult.error() on an agent `error` event, a failed/empty `done`, or an
    HTTP error. See src/api/routes.py:224-351 for the event shapes."""
    payload = {"query": prompt_text, "session_id": session_id}
    final = ""
    tokens: list[str] = []
    cur_event = ""
    saw_error = False
    try:
        async with client.stream(
            "POST", f"{base_url}/api/v1/query", json=payload, headers=headers, timeout=120.0
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    cur_event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(evt, dict):
                    continue
                if cur_event == "error":
                    saw_error = True
                elif cur_event == "done":
                    if evt.get("success") is False:
                        saw_error = True
                    final = evt.get("response") or final
                elif cur_event == "token":
                    tokens.append(evt.get("content", ""))
    except httpx.HTTPError:
        return SampleResult.error()
    text = final or "".join(tokens)
    if saw_error or not text:
        # an error event, a failed `done`, or no answer text at all -> errored,
        # NOT a real empty answer to be judged.
        return SampleResult.error()
    return SampleResult(text=text)


async def replay_item(
    item: SuiteItem,
    *,
    base_url: str,
    n: int,
    concurrency_sem: asyncio.Semaphore,
    headers: dict[str, str],
    http_client: httpx.AsyncClient | None,
    _post: Callable[..., Awaitable[SampleResult]] = post_query,
) -> list[SampleResult]:
    """Replay one item N times, fresh session each; return N SampleResults."""

    async def one() -> SampleResult:
        session_id = f"{RUN_TAG}__{item.id}__{uuid.uuid4().hex[:8]}"
        async with concurrency_sem:
            try:
                return await _post(http_client, base_url, item.text, session_id, headers)
            except Exception:  # one dead replay must not abort the batch
                return SampleResult.error()

    return list(await asyncio.gather(*[one() for _ in range(n)]))
