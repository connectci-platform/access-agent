import pytest

from src.config import settings
from src.llm.providers import active_model_name


@pytest.mark.parametrize(
    "provider,field,value,expected",
    [
        ("openai", "OPENAI_MODEL", "gpt-4o", "gpt-4o"),
        ("vllm", "VLLM_MODEL_NAME", "ccs/Qwen/Qwen3.6-35B-A3B-FP8", "ccs/Qwen/Qwen3.6-35B-A3B-FP8"),
    ],
)
def test_active_model_name_reads_runtime_provider(monkeypatch, provider, field, value, expected):
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, field, value)
    assert active_model_name() == expected


def test_active_model_name_access_ai(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "access_ai")
    assert active_model_name() == "access-llama"
