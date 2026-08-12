from pathlib import Path

import pytest

from src.redteam import __main__ as cli
from src.redteam.suite import SuiteVersionMismatch


@pytest.mark.asyncio
async def test_run_from_env_aborts_on_version_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("READ_ONLY", "true")
    # a prompts.json whose suite_version differs from the committed baseline
    bad = tmp_path / "prompts.json"
    bad.write_text('{"suite_version":"v2-BAD","pyrit_version":"x","generated_at":"t","prompts":[]}')
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(bad))

    async def _boom(*a, **k):
        raise AssertionError("gate reached despite version mismatch")

    with pytest.raises(SuiteVersionMismatch):
        await cli.run_from_env(_gate=_boom)


@pytest.mark.asyncio
async def test_run_from_env_propagates_judge_outage(monkeypatch, tmp_path):
    from src.redteam.gate import JudgeOutage

    monkeypatch.setenv("READ_ONLY", "true")
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(tmp_path / "art"))
    fixture = Path(__file__).parent / "fixtures" / "prompts.sample.json"
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(fixture))
    # This fixture intentionally carries an extra floor-section id beyond the
    # baseline's expected_prompt_ids (see test_suite.py); not under test here.
    monkeypatch.setattr(cli, "assert_suite_complete", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(
        cli.settings, "EVAL_JUDGE_MODEL", "ccs/Qwen/Qwen3.6-35B-A3B-FP8", raising=False
    )

    # Not under test here — the in-process ASGI app's catalog is unwarmed, which
    # would otherwise trip a real SurfaceOutage before the judge outage under test.
    async def noop_preflight(*a, **k):
        pass

    monkeypatch.setattr(cli, "_surface_preflight", noop_preflight, raising=True)

    async def outage_gate(*a, **k):
        raise JudgeOutage("judge failed on 5/5 judged samples (>= 50%) — aborting")

    with pytest.raises(JudgeOutage):
        await cli.run_from_env(_gate=outage_gate)
