"""Judge.score forwards profile= into build_judge_prompt.

See docs/superpowers/specs/2026-09-21-profile-ab-grading-decisions.md
("The mechanism"): threading profile through the judge is what lets the
"## Request profile" section reach the LLM judge's prompt.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.profile import AllocatedResource, UserProfile
from src.eval.judge import Judge
from src.eval.rubric import build_judge_prompt as _real_build_judge_prompt


def _completion(content: str, finish_reason: str = "stop") -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


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


async def test_score_forwards_profile_to_build_judge_prompt():
    judge = Judge()
    judge.client.chat.completions.create = AsyncMock(return_value=_completion(_good_payload()))
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )

    with patch("src.eval.judge.build_judge_prompt", wraps=_real_build_judge_prompt) as mock_build:
        result = await judge.score(query="q", answer="a", profile=profile)

    assert result is not None
    assert mock_build.call_args.kwargs["profile"] == profile


async def test_score_accounts_for_profile_section_in_max_tokens():
    judge = Judge()
    judge.client.chat.completions.create = AsyncMock(return_value=_completion(_good_payload()))
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )

    await judge.score(query="q", answer="a", profile=profile)

    kwargs = judge.client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 500 + 120  # factless base + profile-section headroom


async def test_score_without_profile_does_not_add_headroom():
    judge = Judge()
    judge.client.chat.completions.create = AsyncMock(return_value=_completion(_good_payload()))

    await judge.score(query="q", answer="a")

    kwargs = judge.client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 500
