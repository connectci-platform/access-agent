from pathlib import Path

import pytest

from src.redteam import __main__ as cli


def test_artifact_dir_inside_repo_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("READ_ONLY", "true")
    # point the artifact dir at a path inside the repo root -> must be rejected
    repo_root = Path(cli.__file__).resolve().parents[2]
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(repo_root / "src" / "redteam" / "leak"))
    from src.redteam.gate import GateResult

    async def fake_gate(*a, **k):
        return GateResult(
            flags=[],
            artifact_records=[
                {"id": "x", "response": "R", "errored": False, "verdict": "defended"}
            ],
        )

    fixture = Path(__file__).parent / "fixtures" / "prompts.sample.json"
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(fixture))
    import asyncio

    with pytest.raises(RuntimeError, match="artifact"):
        asyncio.run(cli.run_from_env(_gate=fake_gate))
