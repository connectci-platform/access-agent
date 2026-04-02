"""Tests for eval question loading."""

import json

from src.eval.questions import load_questions


class TestLoadQuestions:
    def test_load_from_json_file(self, tmp_path):
        data = [
            {"id": "q1", "question": "What is ACCESS?", "capability_area": "general"},
            {
                "id": "q2",
                "question": "How do I get an allocation?",
                "capability_area": "allocations",
            },
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        questions = load_questions(str(path))
        assert len(questions) == 2
        assert questions[0].id == "q1"
        assert questions[0].question == "What is ACCESS?"
        assert questions[0].capability_area == "general"

    def test_load_friendly_battery(self):
        questions = load_questions("eval/questions/friendly_battery.json")
        assert len(questions) == 50
        assert all(q.question for q in questions)
        assert all(q.capability_area for q in questions)

    def test_load_real_user_battery(self):
        questions = load_questions("eval/questions/real_user_battery.json")
        assert len(questions) == 50
        assert all(q.question for q in questions)

    def test_no_reference_tags_in_friendly(self):
        """Friendly battery questions should not contain [RU#] or [A3#] tags."""
        questions = load_questions("eval/questions/friendly_battery.json")
        for q in questions:
            assert "[RU" not in q.question, f"Found reference tag in: {q.question}"
            assert "[A3" not in q.question, f"Found reference tag in: {q.question}"
