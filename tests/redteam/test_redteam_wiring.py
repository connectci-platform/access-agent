from pathlib import Path

import pytest

FIXTURE_PROMPTS = Path(__file__).parent / "fixtures" / "prompts.sample.json"

# Exact match for suite-v1/baseline.json's expected_prompt_ids (no extra/missing
# ids) — needed once assert_suite_complete is wired in, unlike FIXTURE_PROMPTS
# above which intentionally carries an extra floor-section entry.
_COMPLETE_PROMPTS_JSON = (
    '{"suite_version":"v1-2026-05-08","pyrit_version":"0.13.0","prompts":['
    '{"id":"wrapped__aligned__stop-sign","text":"x","section":"wrapped",'
    '"wrapper_id":"aligned","probe_id":"stop-sign","wrapper_category":"dual-persona",'
    '"probe_category":"illegal","wrapper_source":"pyrit-bundled"},'
    '{"id":"wrapped__prefix_injection__system-prompt","text":"x","section":"wrapped",'
    '"wrapper_id":"prefix_injection","probe_id":"system-prompt","wrapper_category":"output-coercion",'
    '"probe_category":"leakage","wrapper_source":"pyrit-bundled"},'
    '{"id":"wrapped__refusal_suppression__stop-sign","text":"x","section":"wrapped",'
    '"wrapper_id":"refusal_suppression","probe_id":"stop-sign","wrapper_category":"output-coercion",'
    '"probe_category":"illegal","wrapper_source":"pyrit-bundled"},'
    '{"id":"wrapped__wikipedia_with_title__credentials","text":"x","section":"wrapped",'
    '"wrapper_id":"wikipedia_with_title","probe_id":"credentials","wrapper_category":"output-coercion",'
    '"probe_category":"leakage","wrapper_source":"pyrit-bundled"}'
    "]}"
)


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
    # Not under test here — match the baseline's judge/scorer pin so this
    # transport-focused test doesn't trip the (Task 3) judge/scorer asserts.
    monkeypatch.setattr(
        cli.settings, "EVAL_JUDGE_MODEL", "ccs/Qwen/Qwen3.6-35B-A3B-FP8", raising=False
    )

    seen = {}

    async def fake_gate(items, *, base_url, http_client, **kw):
        seen["base_url"] = base_url
        seen["transport"] = type(http_client._transport).__name__
        from src.redteam.gate import GateResult

        return GateResult(flags=[], artifact_records=[])

    # Skip the completeness manifest + preflight for THIS transport test.
    monkeypatch.setattr(cli, "assert_suite_complete", lambda *a, **k: None, raising=True)
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
    # Not under test here — match the baseline's judge/scorer pin so this
    # transport-focused test doesn't trip the (Task 3) judge/scorer asserts.
    monkeypatch.setattr(
        cli.settings, "EVAL_JUDGE_MODEL", "ccs/Qwen/Qwen3.6-35B-A3B-FP8", raising=False
    )
    seen = {}

    async def fake_gate(items, *, base_url, http_client, **kw):
        seen["base_url"] = base_url
        seen["transport"] = type(http_client._transport).__name__
        from src.redteam.gate import GateResult

        return GateResult(flags=[], artifact_records=[])

    monkeypatch.setattr(cli, "assert_suite_complete", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(cli, "_surface_preflight", None, raising=False)
    await cli.run_from_env(_gate=fake_gate)
    # httpx 0.28's default async transport class is AsyncHTTPTransport (not
    # ASGITransport) — the plain, unconfigured httpx.AsyncClient() branch.
    assert seen["transport"] == "AsyncHTTPTransport"
    assert seen["base_url"] == "http://localhost:8000"


@pytest.mark.asyncio
async def test_run_from_env_judge_model_mismatch_raises(monkeypatch, tmp_path):
    import src.redteam.__main__ as cli
    from src.redteam.gate import JudgeMismatch

    # baseline judge_model is "ccs/Qwen/..."; force the running judge to differ.
    monkeypatch.setattr(cli.settings, "EVAL_JUDGE_MODEL", "gpt-4o-mini", raising=False)
    monkeypatch.setenv("READ_ONLY", "true")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(_COMPLETE_PROMPTS_JSON)
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(prompts))
    monkeypatch.delenv("REDTEAM_BASE_URL", raising=False)
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(tmp_path / "art"))
    with pytest.raises(JudgeMismatch):
        await cli.run_from_env(_gate=_should_not_run)


async def _should_not_run(*a, **k):
    raise AssertionError("gate ran despite a precondition failure")


@pytest.mark.asyncio
async def test_run_from_env_judge_model_match_reaches_gate(monkeypatch, tmp_path):
    import src.redteam.__main__ as cli
    from src.redteam.gate import GateResult

    monkeypatch.setattr(
        cli.settings, "EVAL_JUDGE_MODEL", "ccs/Qwen/Qwen3.6-35B-A3B-FP8", raising=False
    )
    monkeypatch.setenv("READ_ONLY", "true")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(_COMPLETE_PROMPTS_JSON)
    monkeypatch.setenv("REDTEAM_PROMPTS_PATH", str(prompts))
    monkeypatch.delenv("REDTEAM_BASE_URL", raising=False)
    monkeypatch.setenv("REDTEAM_ARTIFACT_DIR", str(tmp_path / "art"))

    called = {}

    async def fake_gate(items, *, base_url, http_client, **kw):
        called["ran"] = True
        return GateResult(flags=[], artifact_records=[])

    await cli.run_from_env(_gate=fake_gate)
    assert called.get("ran") is True


def test_assert_suite_complete_rejects_missing_and_extra(tmp_path):
    from src.redteam.suite import (
        Baseline,
        PromptEntry,
        Prompts,
        SuiteIncomplete,
        assert_suite_complete,
    )

    def mk(ids):
        return Prompts(
            suite_version="v1",
            pyrit_version="0.13.0",
            entries=[
                PromptEntry(
                    id=i,
                    text="x",
                    section="floor",
                    wrapper_id=None,
                    probe_id=None,
                    wrapper_category=None,
                    probe_category="illegal",
                    wrapper_source=None,
                )
                for i in ids
            ],
        )

    base = Baseline(
        suite_version="v1",
        scorer_version="cascade-v1",
        judge_model="m",
        verdicts={},
        expected_prompt_ids=frozenset({"a", "b"}),
    )
    assert_suite_complete(base, mk(["a", "b"]))  # exact -> ok
    with pytest.raises(SuiteIncomplete):
        assert_suite_complete(base, mk(["a"]))  # missing b
    with pytest.raises(SuiteIncomplete):
        assert_suite_complete(base, mk(["a", "b", "c"]))  # extra c


def _coro(value):
    """Return an async callable that ignores its args and returns `value`."""

    async def _fake(*a, **k):
        return value

    return _fake


def test_main_exit_codes_and_status(monkeypatch, tmp_path):
    import json

    import src.redteam.__main__ as cli
    from src.redteam.gate import GateResult
    from src.redteam.report import Flag

    status = tmp_path / "status.json"
    monkeypatch.setenv("REDTEAM_STATUS_PATH", str(status))
    monkeypatch.setenv("REDTEAM_ISSUE_BODY", str(tmp_path / "body.md"))

    # regression -> exit 3
    flag = Flag("wrapped__aligned__stop-sign", "complies", "abcd", "candidate-regression")
    monkeypatch.setattr(cli, "run_from_env", _coro(GateResult(flags=[flag], artifact_records=[])))
    with pytest.raises(SystemExit) as ex:
        cli.main_argv(["--emit-issue-body"])
    assert ex.value.code == 3
    st = json.loads(status.read_text())
    assert st["disposition"] == "regression" and st["flag_count"] == 1

    # clean -> exit 0
    monkeypatch.setattr(cli, "run_from_env", _coro(GateResult(flags=[], artifact_records=[])))
    with pytest.raises(SystemExit) as ex:
        cli.main_argv(["--emit-issue-body"])
    assert ex.value.code == 0
    assert json.loads(status.read_text())["disposition"] == "clean"
