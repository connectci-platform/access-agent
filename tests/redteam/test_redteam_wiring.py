from pathlib import Path

import pytest

FIXTURE_PROMPTS = Path(__file__).parent / "fixtures" / "prompts.sample.json"


@pytest.mark.asyncio
async def test_run_from_env_unset_base_url_uses_asgi(monkeypatch, tmp_path):
    """REDTEAM_BASE_URL unset -> the client is ASGITransport-backed and lifespan is entered."""
    import src.redteam.__main__ as cli

    # Valid suite + baseline on disk (reuse the existing sample fixture, which
    # matches suite-v1/baseline.json's suite_version and wrapper ids).
    monkeypatch.setenv("READ_ONLY", "true")
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(FIXTURE_PROMPTS))
    monkeypatch.delenv("REDTEAM_BASE_URL", raising=False)
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(tmp_path / "art"))

    seen = {}

    async def fake_gate(items, *, base_url, http_client, **kw):
        seen["base_url"] = base_url
        seen["transport"] = type(http_client._transport).__name__
        from src.redteam.gate import GateResult
        return GateResult(flags=[], artifact_records=[])

    # Skip the completeness manifest + preflight for THIS transport test.
    monkeypatch.setattr(cli, "assert_suite_complete", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cli, "_surface_preflight", None, raising=False)
    await cli.run_from_env(_gate=fake_gate)
    assert seen["transport"] == "ASGITransport"
    assert seen["base_url"] == "http://redteam-asgi"


@pytest.mark.asyncio
async def test_run_from_env_set_base_url_uses_plain_client(monkeypatch, tmp_path):
    import src.redteam.__main__ as cli
    monkeypatch.setenv("READ_ONLY", "true")
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(FIXTURE_PROMPTS))
    monkeypatch.setenv("REDTEAM_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(tmp_path / "art"))
    seen = {}

    async def fake_gate(items, *, base_url, http_client, **kw):
        seen["base_url"] = base_url
        seen["transport"] = type(http_client._transport).__name__
        from src.redteam.gate import GateResult
        return GateResult(flags=[], artifact_records=[])

    monkeypatch.setattr(cli, "assert_suite_complete", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cli, "_surface_preflight", None, raising=False)
    await cli.run_from_env(_gate=fake_gate)
    # httpx 0.28's default async transport class is AsyncHTTPTransport (not
    # ASGITransport) — the plain, unconfigured httpx.AsyncClient() branch.
    assert seen["transport"] == "AsyncHTTPTransport"
    assert seen["base_url"] == "http://localhost:8000"
