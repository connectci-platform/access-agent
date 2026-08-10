import pytest

from src.redteam import __main__ as cli


def test_gate_requires_read_only_env(monkeypatch):
    # The gate entrypoint must refuse to run unless READ_ONLY is truthy.
    monkeypatch.delenv("READ_ONLY", raising=False)
    assert cli.read_only_enabled() is False
    monkeypatch.setenv("READ_ONLY", "true")
    assert cli.read_only_enabled() is True


@pytest.mark.asyncio
async def test_run_from_env_aborts_when_not_read_only(monkeypatch):
    # F9: the abort path in run_from_env itself (not just the helper) must fire,
    # BEFORE any replay — pass a gate that would explode if reached.
    monkeypatch.delenv("READ_ONLY", raising=False)

    async def _boom(*a, **k):
        raise AssertionError("gate reached despite READ_ONLY off")

    with pytest.raises(RuntimeError, match="READ_ONLY"):
        await cli.run_from_env(_gate=_boom)
