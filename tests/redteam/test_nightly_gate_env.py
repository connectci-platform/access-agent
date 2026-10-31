"""The nightly red-team gate runs the app lifespan in-process with
ENVIRONMENT=production, so its workflow step must carry the settings the
lifespan's startup check requires."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.agent_token import check_agent_token_config

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "nightly.yml"


def _gate_env() -> dict[str, str]:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    steps = [step for job in jobs.values() for step in job.get("steps", [])]
    (gate,) = [step for step in steps if step.get("id") == "gate"]
    return gate["env"]


def test_gate_env_passes_the_lifespan_agent_token_check():
    env = _gate_env()
    assert env["ENVIRONMENT"] == "production"
    check_agent_token_config(
        SimpleNamespace(
            AGENT_TURN_TIMEOUT_S=120,
            ENVIRONMENT=env["ENVIRONMENT"],
            AGENT_TOKEN_ISSUER=env.get("AGENT_TOKEN_ISSUER", ""),
            AGENT_TOKEN_JWKS_URL=env.get("AGENT_TOKEN_JWKS_URL", ""),
        )
    )
