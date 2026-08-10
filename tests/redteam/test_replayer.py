import asyncio

import pytest

from src.redteam.replayer import redteam_headers, replay_item
from src.redteam.suite import PromptEntry, SuiteItem


def _item():
    e = PromptEntry("id-1", "PLACEHOLDER", "wrapped", "w", "p", "c", "d", "s")
    return SuiteItem(id="id-1", text="PLACEHOLDER", expected="defended", entry=e)


def test_redteam_headers_shape():
    h = redteam_headers("v1-2026-05-08", "run-xyz")
    assert h["X-Redteam-Suite"] == "v1-2026-05-08"
    assert h["X-Redteam-Run-Id"] == "run-xyz"
    assert "X-Redteam" in h


@pytest.mark.asyncio
async def test_replay_item_fresh_session_per_sample():
    seen_sessions = []

    async def fake_post(client, base_url, prompt_text, session_id, headers):
        seen_sessions.append(session_id)
        return "resp"

    sem = asyncio.Semaphore(6)
    out = await replay_item(
        _item(),
        base_url="http://x",
        n=5,
        concurrency_sem=sem,
        headers={},
        http_client=None,
        _post=fake_post,
    )
    assert len(out) == 5
    assert len(set(seen_sessions)) == 5  # every replay a distinct session
