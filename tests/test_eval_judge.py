"""Tests for the LLM judge."""

from unittest.mock import AsyncMock, MagicMock

from src.eval.judge import Judge, parse_judge_response


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


def test_invalid_answerability_rejected():
    # answerable not in ("Fair","Unfair") → parse rejects the whole payload.
    payload = _good_payload().replace('"answerable": "Fair"', '"answerable": "Maybe"')
    assert parse_judge_response(payload) is None


def _completion(content: str, finish_reason: str = "stop") -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


async def test_thinking_judge_strips_trace_and_raises_budget():
    judge = Judge(base_url="http://vllm.local/v1", thinking=True)
    raw = "the model muses about the rubric at length</think>\n" + _good_payload()
    judge.client.chat.completions.create = AsyncMock(return_value=_completion(raw))

    result = await judge.score(query="q", answer="a")

    assert result is not None
    assert result.scores["correctness"] == 2
    kwargs = judge.client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 500 + 3000  # factless base + reasoning headroom
    assert kwargs["extra_body"] is None  # thinking left ON for this endpoint


async def test_truncated_response_retries_with_doubled_budget():
    judge = Judge(base_url="http://vllm.local/v1")
    judge.client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion('{"answerable": "Fair", "correctn', finish_reason="length"),
            _completion(_good_payload()),
        ]
    )

    result = await judge.score(query="q", answer="a")

    assert result is not None
    calls = judge.client.chat.completions.create.call_args_list
    assert calls[0].kwargs["max_tokens"] == 500
    assert calls[1].kwargs["max_tokens"] == 1000
    # On-premise non-thinking judge must disable the reasoning trace via vLLM.
    assert calls[0].kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
