import os

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_redteam_suite_no_defended_regressions():
    """Full gate against the live agent. Flags -> assertion detail is REDACTED
    (ids + hashes only). Requires: running agent, prod model, READ_ONLY=true,
    fetched prompts.json at REDTEAM_PROMPTS_PATH."""
    if not os.getenv("REDTEAM_PROMPTS_PATH"):
        pytest.skip("REDTEAM_PROMPTS_PATH not set (no fetched suite)")
    assert os.getenv("READ_ONLY", "").lower() in ("1", "true"), "gate must run READ_ONLY"
    from src.redteam import __main__ as cli
    from src.redteam.gate import SurfaceOutage

    # run_from_env does the /health surface preflight in-process before replay;
    # a healthy prod run reaching here (no SurfaceOutage raised) means the
    # preflight passed implicitly.
    try:
        result = await cli.run_from_env()
    except SurfaceOutage as exc:
        pytest.fail(f"surface preflight failed on a run expected to be healthy: {exc}")
    regressions = [f for f in result.flags if f.kind == "candidate-regression"]
    # redacted assertion message — ids + hashes only, never response bodies
    assert not regressions, "candidate regressions: " + "; ".join(
        f"{f.prompt_id} sha256:{f.content_hash}" for f in regressions
    )
