"""Tests for the LLM eval query interface."""

from src.eval.ask import build_ask_prompt, build_schema_description


class TestBuildSchemaDescription:
    def test_includes_table_names(self):
        desc = build_schema_description()
        assert "eval_runs" in desc
        assert "eval_scores" in desc

    def test_includes_key_columns(self):
        desc = build_schema_description()
        assert "composite_score" in desc
        assert "question_id" in desc
        assert "source" in desc

    def test_schema_description_is_v2(self):
        s = build_schema_description()
        assert "completeness" not in s
        assert "specificity" in s
        assert "answerable" in s
        assert "(1-5)" not in s
        assert "0-2" in s or "3-point" in s


class TestBuildAskPrompt:
    def test_includes_question(self):
        prompt = build_ask_prompt("What's the average correctness score?")
        assert "What's the average correctness score?" in prompt

    def test_includes_schema(self):
        prompt = build_ask_prompt("test")
        assert "eval_runs" in prompt
        assert "eval_scores" in prompt
