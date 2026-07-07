"""Tests for the LLM judge."""

from src.eval.judge import parse_judge_response


def _good_payload():
    return """```json
{
  "answerable": "Fair",
  "correctness": {"value": "Correct", "justification": "a"},
  "specificity": {"value": "Actionable", "justification": "b"},
  "relevance": {"value": "On-target", "justification": "c"},
  "citation_quality": {"value": "Good", "justification": "d"},
  "hedging": {"value": "Calibrated", "justification": "e"}
}
```"""


def test_parses_v2_labels_to_ordinals():
    r = parse_judge_response(_good_payload())
    assert r is not None
    assert r.scores == {
        "correctness": 2,
        "specificity": 2,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
    }
    assert r.answerable is True
    assert r.specificity_na is False
    assert abs(r.composite - 1.0) < 1e-9


def test_specificity_na_sets_flag_and_none_score():
    payload = _good_payload().replace('"value": "Actionable"', '"value": "N/A"')
    r = parse_judge_response(payload)
    assert r is not None
    assert r.scores["specificity"] is None
    assert r.specificity_na is True


def test_unfair_answerability_still_parses():
    payload = _good_payload().replace('"answerable": "Fair"', '"answerable": "Unfair"')
    r = parse_judge_response(payload)
    assert r is not None
    assert r.answerable is False


def test_rejects_old_1_5_integer():
    payload = """```json
{"answerable": "Fair",
 "correctness": {"score": 5},
 "specificity": {"value": "Actionable"},
 "relevance": {"value": "On-target"},
 "citation_quality": {"value": "Good"},
 "hedging": {"value": "Calibrated"}}
```"""
    assert parse_judge_response(payload) is None


def test_rejects_out_of_set_label():
    payload = _good_payload().replace('"value": "Correct"', '"value": "Excellent"')
    assert parse_judge_response(payload) is None
