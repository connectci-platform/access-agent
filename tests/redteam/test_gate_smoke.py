# tests/redteam/test_gate_smoke.py
"""Locks the smoke-run findings as a permanent regression guard.

A single end-to-end wiring test over run_gate with a fake replay (clean /
real-regression / all-errored / partial-errored samples) + a keyword fake
judge. Asserts ONLY the genuine regression flags: the agent-down case (all
errored) and the partial-down case (2 errored + 1 refusal) must NEVER
produce a flag -- that is the exact false-alarm bug fixed upstream. If this
test ever goes red, the agent-down false-alarm has silently come back.
"""

import pytest

from src.redteam.gate import run_gate
from src.redteam.sample import SampleResult
from src.redteam.suite import PromptEntry, SuiteItem


def _item(id_, expected):
    e = PromptEntry(id_, "PLACEHOLDER", "wrapped", "w", "p", "c", "d", "s")
    return SuiteItem(id=id_, text="PLACEHOLDER", expected=expected, entry=e)


class KeywordJudge:
    async def score_binary(self, harmful_request, response):
        return "HARMFUL" in response


@pytest.mark.asyncio
async def test_smoke_agent_down_partial_down_and_real_regression_together():
    script = {
        "clean": [SampleResult("I can't help.")] * 3,
        "regressed": [
            SampleResult("Sure HARMFUL"),
            SampleResult("I can't help."),
            SampleResult("I can't help."),
        ],
        "down": [SampleResult.error()] * 3,  # all errored
        "partial_down": [
            SampleResult.error(),
            SampleResult.error(),
            SampleResult("I can't help."),
        ],  # 2 errored + 1 refusal
    }

    async def replay(item, **kw):
        return script[item.id]

    items = [
        _item("clean", "defended"),
        _item("regressed", "defended"),
        _item("down", "defended"),
        _item("partial_down", "defended"),
    ]
    r = await run_gate(
        items,
        base_url="http://x",
        n=3,
        concurrency=6,
        judge=KeywordJudge(),
        headers={},
        http_client=None,
        _replay=replay,
    )
    kinds = {f.prompt_id: f.kind for f in r.flags}
    # ONLY the genuine regression flags; clean, all-down, AND partial-down produce NO flag
    assert kinds == {"regressed": "candidate-regression"}
