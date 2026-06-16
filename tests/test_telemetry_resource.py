from src.config import settings
from src.telemetry.setup import _build_resource_attributes


def test_version_and_env_come_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_VERSION", "1.2.3-abc")
    monkeypatch.setattr(settings, "DEPLOY_ENV", "staging")
    attrs = _build_resource_attributes("access-agent")
    assert attrs["service.version"] == "1.2.3-abc"
    assert attrs["deployment.environment"] == "staging"


def test_falls_back_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_VERSION", "")
    monkeypatch.setattr(settings, "DEPLOY_ENV", "")
    monkeypatch.setenv("ENVIRONMENT", "local")
    attrs = _build_resource_attributes("access-agent")
    assert attrs["service.version"] == "unknown"
    assert attrs["deployment.environment"] == "local"
