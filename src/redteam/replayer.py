"""Replay frozen prompts against /api/v1/query and collect N samples each."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx

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
) -> str:
    """POST and parse the SSE stream; return the final answer text.

    Protocol (verified against src/api/routes.py:224-351): named-event SSE,
    each event is `event: <name>\\ndata: <json>\\n\\n`. The final answer is in
    the `done` event under key `response` (routes.py:330). `token` events carry
    incremental `content` and are a fallback if `done` is missing.
    """
    payload = {"query": prompt_text, "session_id": session_id}
    final = ""
    tokens: list[str] = []
    cur_event = ""
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
            if cur_event == "done" and isinstance(evt, dict):
                final = evt.get("response") or final
            elif cur_event == "token" and isinstance(evt, dict):
                tokens.append(evt.get("content", ""))
    return final or "".join(tokens)


async def replay_item(
    item: SuiteItem,
    *,
    base_url: str,
    n: int,
    concurrency_sem: asyncio.Semaphore,
    headers: dict[str, str],
    http_client: httpx.AsyncClient | None,
    _post: Callable[..., Awaitable[str]] = post_query,
) -> list[str]:
    """Replay one item N times, fresh session each; return N response texts."""

    async def one() -> str:
        session_id = f"{RUN_TAG}__{item.id}__{uuid.uuid4().hex[:8]}"
        async with concurrency_sem:
            return await _post(http_client, base_url, item.text, session_id, headers)

    return list(await asyncio.gather(*[one() for _ in range(n)]))
