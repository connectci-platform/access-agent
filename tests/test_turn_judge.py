import asyncio
from types import SimpleNamespace

from src.config import settings
from src.turn_judge import _build_judge_prompt, _parse_judge_response, judge_turn


class TestParse:
    def test_clean_json(self):
        out = _parse_judge_response(
            '{"query_intent":"genuine","refused":false,"is_deflection":false}'
        )
        assert out == {"query_intent": "genuine", "refused": False, "is_deflection": False}

    def test_code_fenced_and_prose(self):
        raw = 'Here you go:\n```json\n{"query_intent":"malicious","refused":true,"is_deflection":false}\n```'
        out = _parse_judge_response(raw)
        assert out["query_intent"] == "malicious"
        assert out["refused"] is True

    def test_invalid_intent_dropped(self):
        out = _parse_judge_response(
            '{"query_intent":"banana","refused":false,"is_deflection":true}'
        )
        assert "query_intent" not in out
        assert out["is_deflection"] is True

    def test_truthy_coercion(self):
        out = _parse_judge_response('{"refused":1,"is_deflection":0}')
        assert out["refused"] is True and out["is_deflection"] is False

    def test_garbage_and_none(self):
        assert _parse_judge_response("not json at all") == {}
        assert _parse_judge_response(None) == {}
        assert _parse_judge_response("[1,2,3]") == {}


class TestPrompt:
    def test_includes_and_truncates(self):
        p = _build_judge_prompt("q" * 5000, "a" * 9000)
        assert "USER QUERY" in p and "ASSISTANT ANSWER" in p
        assert p.count("q") <= 2100  # query truncated to ~2000
        assert p.count("a") <= 4200  # answer truncated to ~4000


class TestJudgeTurn:
    def test_disabled_short_circuits(self, monkeypatch):
        monkeypatch.setattr(settings, "TURN_JUDGE_ENABLED", False)
        assert asyncio.run(judge_turn("hi", "hello")) == {}

    def test_happy_path_mocked_llm(self, monkeypatch):
        monkeypatch.setattr(settings, "TURN_JUDGE_ENABLED", True)

        class _FakeModel:
            async def ainvoke(self, _prompt):
                return SimpleNamespace(
                    content='{"query_intent":"casual","refused":false,"is_deflection":true}'
                )

        monkeypatch.setattr("src.turn_judge.get_llm", lambda **kw: _FakeModel())
        out = asyncio.run(judge_turn("hi", "hey there"))
        assert out == {"query_intent": "casual", "refused": False, "is_deflection": True}

    def test_llm_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "TURN_JUDGE_ENABLED", True)

        def _boom(**kw):
            raise RuntimeError("vllm down")

        monkeypatch.setattr("src.turn_judge.get_llm", _boom)
        assert asyncio.run(judge_turn("q", "a")) == {}
