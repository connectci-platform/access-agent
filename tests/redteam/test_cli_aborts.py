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
