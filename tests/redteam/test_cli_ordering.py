import pytest

from src.redteam import __main__ as cli
from src.redteam.gate import GateResult


@pytest.mark.asyncio
async def test_artifact_written_before_return(monkeypatch, tmp_path):
    monkeypatch.setenv("READ_ONLY", "true")
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(tmp_path / "art"))
    # a valid prompts.json matching the baseline version (use the fixture)
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "prompts.sample.json"
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(fixture))

    async def fake_gate(*a, **k):
        return GateResult(
            flags=[], artifact_records=[{"id": "x", "response": "R", "verdict": "defended"}]
        )

    await cli.run_from_env(_gate=fake_gate)
    # artifact file must exist by the time run_from_env returns
    written = list((tmp_path / "art").glob("*.json"))
    assert written, "artifact must be written before run_from_env returns"
