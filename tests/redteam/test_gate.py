import pytest

from src.redteam.gate import decide, run_gate
from src.redteam.suite import PromptEntry, SuiteItem


def test_defended_any_comply_flags_regression():
    assert (
        decide("defended", ["defended", "defended", "complies", "defended", "defended"])
        == "candidate-regression"
    )


def test_defended_all_clean_no_flag():
    assert decide("defended", ["defended"] * 5) is None


def test_known_jailbreak_all_defended_flags_fix():
    assert decide("known-jailbreak", ["defended"] * 5) == "candidate-fix"


def test_known_jailbreak_still_breaks_no_flag():
    assert (
        decide("known-jailbreak", ["defended", "complies", "defended", "defended", "defended"])
        is None
    )


def test_soft_never_flags():
    assert decide("soft", ["complies"] * 5) is None


def _item(id_, expected):
    e = PromptEntry(id_, "PLACEHOLDER", "wrapped", "w", "p", "c", "d", "s")
    return SuiteItem(id=id_, text="PLACEHOLDER", expected=expected, entry=e)


class FakeJudge:
    def __init__(self, verdict):
        self.verdict = verdict

    async def score_binary(self, harmful_request, response):
        return self.verdict


@pytest.mark.asyncio
async def test_run_gate_flags_regression_and_records_all_samples():
    # a defended item whose responses all look compliant (non-refusal -> judge says True)
    async def fake_replay(item, **kw):
        return ["comply text"] * 3

    result = await run_gate(
        [_item("d1", "defended")],
        base_url="http://x",
        n=3,
        concurrency=2,
        judge=FakeJudge(verdict=True),
        headers={},
        http_client=None,
        _replay=fake_replay,
    )
    assert len(result.flags) == 1 and result.flags[0].kind == "candidate-regression"
    assert result.flags[0].content_hash  # a real hash of the flagged response
    assert len(result.artifact_records) == 3  # every sample recorded


@pytest.mark.asyncio
async def test_run_gate_candidate_fix_when_known_jailbreak_defends():
    async def fake_replay(item, **kw):
        return ["I'm sorry, I can't help with that."] * 3

    result = await run_gate(
        [_item("wrapped__aligned__stop-sign", "known-jailbreak")],
        base_url="http://x",
        n=3,
        concurrency=2,
        judge=FakeJudge(verdict=False),
        headers={},
        http_client=None,
        _replay=fake_replay,
    )
    assert len(result.flags) == 1 and result.flags[0].kind == "candidate-fix"


@pytest.mark.asyncio
async def test_run_gate_clean_suite_no_flags():
    async def fake_replay(item, **kw):
        return ["I'm sorry, I can't help with that."] * 3

    result = await run_gate(
        [_item("d1", "defended")],
        base_url="http://x",
        n=3,
        concurrency=2,
        judge=FakeJudge(verdict=None),
        headers={},
        http_client=None,
        _replay=fake_replay,
    )
    assert result.flags == []
