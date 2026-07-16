"""T2 data-residency guard: production must fail closed on external-model config.

Motivated by a real incident: the eval judge silently ran on a commercial
OpenAI model in production because EVAL_JUDGE_BASE_URL (empty = OpenAI
fallback) was never plumbed through the prod compose file. The guard turns
that class of misconfiguration into a loud boot failure instead of a silent
residency violation.
"""

import pytest

from src.config import Settings


def _mk(**overrides):
    # _env_file=None: tests must not inherit the developer's .env.
    return Settings(_env_file=None, **overrides)


def test_production_refuses_external_model_provider():
    with pytest.raises(ValueError, match="residency"):
        _mk(
            ENVIRONMENT="production",
            LLM_PROVIDER="openai",
            EVAL_JUDGE_BASE_URL="https://jump-external.ccs.uky.edu/v1",
        )


def test_production_refuses_judge_openai_fallback():
    # Empty EVAL_JUDGE_BASE_URL means "use OpenAI" — never acceptable in prod.
    with pytest.raises(ValueError, match="judge"):
        _mk(
            ENVIRONMENT="production",
            LLM_PROVIDER="vllm",
            VLLM_BASE_URL="https://jump-external.ccs.uky.edu/v1",
            EVAL_JUDGE_BASE_URL="",
        )


def test_production_boots_with_on_premise_config():
    s = _mk(
        ENVIRONMENT="production",
        LLM_PROVIDER="vllm",
        VLLM_BASE_URL="https://jump-external.ccs.uky.edu/v1",
        EVAL_JUDGE_BASE_URL="https://jump-external.ccs.uky.edu/v1",
    )
    assert s.LLM_PROVIDER == "vllm"


def test_local_dev_unaffected():
    s = _mk(ENVIRONMENT="local", LLM_PROVIDER="openai", EVAL_JUDGE_BASE_URL="")
    assert s.LLM_PROVIDER == "openai"


def test_docker_dev_unaffected():
    s = _mk(ENVIRONMENT="docker", LLM_PROVIDER="openai", EVAL_JUDGE_BASE_URL="")
    assert s.ENVIRONMENT == "docker"
