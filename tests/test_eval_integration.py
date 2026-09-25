"""Integration test for the eval pipeline (v2 rubric).

Mocks run_agent and the judge LLM to exercise the full pipeline
(run_eval -> judge parse -> scorer write -> models -> report_data
aggregation) without requiring live services.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.profile import AllocatedResource, UserProfile
from src.eval.db import EvalDB
from src.eval.judge import parse_judge_response
from src.eval.report_data import _per_dimension_means, build_report_data
from src.eval.runner import run_question
from src.eval.scorer import run_eval


@pytest.fixture
def mock_db(tmp_path):
    return f"sqlite:///{tmp_path}/test_eval.db"


@pytest.fixture
def tiny_question_set(tmp_path):
    questions = [
        {"id": "t1", "question": "What is ACCESS?", "capability_area": "general"},
        {"id": "t2", "question": "How do I get an allocation?", "capability_area": "allocations"},
        {"id": "t3", "question": "Is Delta down?", "capability_area": "delta"},
    ]
    path = tmp_path / "test_questions.json"
    path.write_text(json.dumps(questions))
    return str(path)


# --- v2 judge payloads (top-level `answerable`, each dimension {"value": <label>}) ---

# q1: fair, all-best answer -> composite 1.0.
MOCK_JUDGE_BEST = json.dumps(
    {
        "answerable": "Fair",
        "correctness": {"value": "Correct", "justification": "Accurate"},
        "specificity": {"value": "Actionable", "justification": "Named resources"},
        "relevance": {"value": "On-target", "justification": "On topic"},
        "citation_quality": {"value": "Good", "justification": "Valid URLs"},
        "hedging": {"value": "Calibrated", "justification": "Well calibrated"},
    }
)

# q2: fair, but the question doesn't call for specifics -> specificity N/A.
MOCK_JUDGE_SPECIFICITY_NA = json.dumps(
    {
        "answerable": "Fair",
        "correctness": {"value": "Correct", "justification": "Accurate"},
        "specificity": {"value": "N/A", "justification": "Process question"},
        "relevance": {"value": "On-target", "justification": "On topic"},
        "citation_quality": {"value": "Good", "justification": "Valid URLs"},
        "hedging": {"value": "Calibrated", "justification": "Well calibrated"},
    }
)

# q3: unfair/unanswerable -> excluded from aggregates by the answerability screen.
MOCK_JUDGE_UNFAIR = json.dumps(
    {
        "answerable": "Unfair",
        "correctness": {"value": "Incorrect", "justification": "Cannot know"},
        "specificity": {"value": "Generic", "justification": "No live status"},
        "relevance": {"value": "Off", "justification": "Unanswerable"},
        "citation_quality": {"value": "Poor", "justification": "No source"},
        "hedging": {"value": "Miscalibrated", "justification": "Overconfident"},
    }
)


def _completion(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


@pytest.mark.asyncio
async def test_full_eval_run(mock_db, tiny_question_set):
    """Full v2 round-trip: three questions, one N/A specificity, one unfair item."""
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "rag_matches": [],
        "tool_results": [],
        "node_trace": [{"node": "classify", "query_type": "static"}],
        "tools_used": [],
    }

    # One judge completion per question, in question order.
    completions = [
        _completion(MOCK_JUDGE_BEST),
        _completion(MOCK_JUDGE_SPECIFICITY_NA),
        _completion(MOCK_JUDGE_UNFAIR),
    ]

    with (
        patch("src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state),
        patch("src.eval.scorer.ToolRegistry") as mock_registry_cls,
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
    ):
        # Mock tool registry. `.tools` is a sync property (name -> definition);
        # set it to a real dict so the snapshot's sorted(registry.tools.keys())
        # works — an AsyncMock child would hand back a coroutine.
        mock_registry = AsyncMock()
        mock_registry.tool_count = 10
        mock_registry.catalog = {"tools": [{"name": "test_tool"}]}
        mock_registry.tools = {"test_tool": object()}
        mock_registry_cls.return_value = mock_registry

        # Mock OpenAI client — a distinct judge response per question.
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = completions
        mock_openai_cls.return_value = mock_client

        summary = await run_eval(
            question_set_path=tiny_question_set,
            database_url=mock_db,
        )

    # All three items were judged and stored (the unfair item is stored, not skipped;
    # the answerability screen only excludes it from downstream aggregates).
    assert summary["questions"] == 3
    assert summary["scored"] == 3
    assert summary["skipped"] == 0

    # Composite is on the v2 [0,1] unit interval.
    assert 0.0 <= summary["composite_score"] <= 1.0

    # v2 dimension set — specificity replaced completeness.
    per_dimension = summary["per_dimension"]
    assert "specificity" in per_dimension
    assert "completeness" not in per_dimension
    assert set(per_dimension) == {
        "correctness",
        "specificity",
        "relevance",
        "citation_quality",
        "hedging",
    }

    # Verify the run row.
    db = EvalDB(mock_db)
    run = db.get_run(summary["run_id"])
    assert run is not None
    assert run.question_count == 3

    scores = db.get_scores_for_run(summary["run_id"])
    assert len(scores) == 3
    assert all(s.source == "judge" for s in scores)
    assert all(s.rubric_version == 2 for s in scores)

    by_qid = {s.question_id: s for s in scores}

    # q1: all-best, fair -> composite 1.0, real specificity value.
    assert by_qid["t1"].answerable is True
    assert by_qid["t1"].specificity == 2
    assert by_qid["t1"].specificity_na is False
    assert abs(by_qid["t1"].composite_score - 1.0) < 1e-9

    # q2: specificity N/A -> None value + flag, still a valid fair row.
    assert by_qid["t2"].answerable is True
    assert by_qid["t2"].specificity is None
    assert by_qid["t2"].specificity_na is True

    # q3: unfair -> stored, but flagged for exclusion by the answerability screen.
    assert by_qid["t3"].answerable is False

    # Answerability screen: report_data aggregation drops the unfair item, so only the
    # two fair rows feed the per-dimension means. The N/A specificity is excluded from
    # the specificity mean (not counted as 0), leaving q1's Actionable (2) alone.
    report = build_report_data(db, since="30d")
    assert report["total_scored"] == 2  # unfair item excluded
    assert report["per_dimension"]["specificity"] == 2.0  # q2 N/A excluded, not zeroed
    assert report["per_dimension"]["correctness"] == 2.0  # both fair rows Correct
    assert "completeness" not in report["per_dimension"]


@pytest.mark.asyncio
async def test_run_eval_persists_skipped_row_for_failed_question(mock_db, tiny_question_set):
    """A question whose agent run raises is not sent to the judge at all — it is
    persisted as a skipped row via persist_skipped_turn and excluded from scoring,
    while the other questions in the set still run and score normally."""
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "rag_matches": [],
        "tool_results": [],
        "node_trace": [{"node": "classify", "query_type": "static"}],
        "tools_used": [],
    }

    async def flaky_agent(*args, **kwargs):
        # tiny_question_set's questions are t1, t2, t3, run in order; fail the
        # second question only, so both the failure path and the surrounding
        # success path are exercised in the same run.
        query = kwargs.get("query") or (args[1] if len(args) > 1 else None)
        if query == "How do I get an allocation?":
            raise RuntimeError("agent blew up")
        return mock_state

    completions = [
        _completion(MOCK_JUDGE_BEST),
        _completion(MOCK_JUDGE_BEST),
    ]

    with (
        patch("src.eval.runner.run_agent", new_callable=AsyncMock, side_effect=flaky_agent),
        patch("src.eval.scorer.get_catalog_aggregator") as mock_aggregator_cls,
        patch("src.eval.scorer.ToolRegistry") as mock_registry_cls,
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
    ):
        # Patch the aggregator so run_eval never reaches the live catalog fetch;
        # fetch_catalog is awaited, so it must be an AsyncMock returning a concrete dict.
        mock_aggregator = MagicMock()
        mock_aggregator.fetch_catalog = AsyncMock(return_value={"tools": [{"name": "test_tool"}]})
        mock_aggregator_cls.return_value = mock_aggregator

        # ToolRegistry holds sync data — MagicMock, not AsyncMock (AsyncMock makes
        # every attribute access a coroutine, which is the CI failure this fixes).
        mock_registry = MagicMock()
        mock_registry.tool_count = 10
        mock_registry.catalog = {"tools": [{"name": "test_tool"}]}
        mock_registry_cls.return_value = mock_registry

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = completions
        mock_openai_cls.return_value = mock_client

        summary = await run_eval(
            question_set_path=tiny_question_set,
            database_url=mock_db,
        )

    assert summary["questions"] == 3
    assert summary["scored"] == 2
    assert summary["skipped"] == 1

    db = EvalDB(mock_db)
    scores = db.get_scores_for_run(summary["run_id"])
    by_qid = {s.question_id: s for s in scores}
    assert by_qid["t2"].source == "skipped"
    assert "agent blew up" in (by_qid["t2"].justifications or {}).get("error", "")
    assert by_qid["t1"].source == "judge"
    assert by_qid["t3"].source == "judge"


@pytest.mark.asyncio
async def test_v2_round_trip(mock_db):
    """Judge payload -> parse -> store -> read, including specificity N/A (spec §9)."""
    db = EvalDB(mock_db)
    run = db.create_run(run_type="pre_production", question_set="t", question_count=1)
    payload = """```json
{"answerable": "Fair",
 "correctness": {"value": "Correct"},
 "specificity": {"value": "N/A"},
 "relevance": {"value": "On-target"},
 "citation_quality": {"value": "Good"},
 "hedging": {"value": "Calibrated"}}
```"""
    r = parse_judge_response(payload)
    assert r is not None
    db.add_score(
        run_id=run.id,
        question_id="q1",
        source="judge",
        correctness=r.scores["correctness"],
        specificity=r.scores["specificity"],
        specificity_na=r.specificity_na,
        answerable=r.answerable,
        rubric_version=2,
        relevance=r.scores["relevance"],
        citation_quality=r.scores["citation_quality"],
        hedging=r.scores["hedging"],
        composite_score=r.composite,
    )
    stored = db.get_scores_for_run(str(run.id))
    assert len(stored) == 1
    assert stored[0].specificity is None
    assert stored[0].specificity_na is True
    assert stored[0].answerable is True

    # _per_dimension_means excludes the N/A specificity from its mean.
    means = _per_dimension_means(stored)
    assert set(means) == {
        "correctness",
        "specificity",
        "relevance",
        "citation_quality",
        "hedging",
    }
    assert means["correctness"] == 2.0
    assert means["specificity"] == 0.0  # only row is N/A -> no values -> 0.0 fallback


@pytest.mark.asyncio
async def test_run_eval_records_profile_in_run_metadata(mock_db, tiny_question_set):
    """eval_runs.metadata carries the resolved profile verbatim (or None)."""
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "rag_matches": [],
        "tool_results": [],
        "node_trace": [{"node": "classify", "query_type": "static"}],
        "tools_used": [],
    }
    completions = [_completion(MOCK_JUDGE_BEST) for _ in range(3)]
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )

    with (
        patch("src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state),
        patch("src.eval.scorer.ToolRegistry") as mock_registry_cls,
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
    ):
        mock_registry = AsyncMock()
        mock_registry.tool_count = 10
        mock_registry.catalog = {"tools": [{"name": "test_tool"}]}
        mock_registry.tools = {"test_tool": object()}
        mock_registry_cls.return_value = mock_registry

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = completions
        mock_openai_cls.return_value = mock_client

        summary = await run_eval(
            question_set_path=tiny_question_set,
            database_url=mock_db,
            profile=profile,
        )

    db = EvalDB(mock_db)
    run = db.get_run(summary["run_id"])
    assert run.metadata_["profile"] == profile.model_dump()

    # run_eval forwards profile all the way to the judge prompt sent to the LLM.
    judge_prompts = [
        call.kwargs["messages"][0]["content"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert all("## Request profile" in p for p in judge_prompts)
    assert all("Delta GPU (rp_name='delta')" in p for p in judge_prompts)

    completions_none = [_completion(MOCK_JUDGE_BEST) for _ in range(3)]
    with (
        patch("src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state),
        patch("src.eval.scorer.ToolRegistry") as mock_registry_cls,
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
    ):
        mock_registry = AsyncMock()
        mock_registry.tool_count = 10
        mock_registry.catalog = {"tools": [{"name": "test_tool"}]}
        mock_registry.tools = {"test_tool": object()}
        mock_registry_cls.return_value = mock_registry

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = completions_none
        mock_openai_cls.return_value = mock_client

        summary_no_profile = await run_eval(
            question_set_path=tiny_question_set,
            database_url=mock_db,
        )

    run_no_profile = db.get_run(summary_no_profile["run_id"])
    assert run_no_profile.metadata_["profile"] is None

    judge_prompts_no_profile = [
        call.kwargs["messages"][0]["content"]
        for call in mock_client.chat.completions.create.call_args_list
    ]
    assert all("## Request profile" not in p for p in judge_prompts_no_profile)


@pytest.mark.asyncio
async def test_run_eval_forwards_and_records_acting_user(mock_db, tiny_question_set):
    """acting_user threads run_eval -> run_question -> run_agent, and is
    recorded in eval_runs.metadata alongside profile (a profile implies an
    authenticated user, so the run's provenance should show who)."""
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "rag_matches": [],
        "tool_results": [],
        "node_trace": [{"node": "classify", "query_type": "static"}],
        "tools_used": [],
    }
    completions = [_completion(MOCK_JUDGE_BEST) for _ in range(3)]

    with (
        patch(
            "src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state
        ) as mock_run_agent,
        patch("src.eval.scorer.ToolRegistry") as mock_registry_cls,
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
    ):
        mock_registry = AsyncMock()
        mock_registry.tool_count = 10
        mock_registry.catalog = {"tools": [{"name": "test_tool"}]}
        mock_registry.tools = {"test_tool": object()}
        mock_registry_cls.return_value = mock_registry

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = completions
        mock_openai_cls.return_value = mock_client

        summary = await run_eval(
            question_set_path=tiny_question_set,
            database_url=mock_db,
            acting_user="jdoe",
        )

    assert all(call.kwargs.get("acting_user") == "jdoe" for call in mock_run_agent.call_args_list)

    db = EvalDB(mock_db)
    run = db.get_run(summary["run_id"])
    assert run.metadata_["acting_user"] == "jdoe"


@pytest.mark.asyncio
async def test_run_question_passes_profile_to_run_agent(mock_db):
    """run_question forwards profile= to run_agent on the agent_full path."""
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "tools_used": [],
    }
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )

    with patch(
        "src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state
    ) as mock_run_agent:
        await run_question(
            "q1",
            "What is ACCESS?",
            tool_catalog={"tools": []},
            profile=profile,
        )

    assert mock_run_agent.call_args.kwargs["profile"] == profile


@pytest.mark.asyncio
async def test_run_question_passes_acting_user_to_run_agent(mock_db):
    """run_question forwards acting_user= to run_agent on the agent_full path."""
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "tools_used": [],
    }

    with patch(
        "src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state
    ) as mock_run_agent:
        await run_question(
            "q1",
            "What is ACCESS?",
            tool_catalog={"tools": []},
            acting_user="jdoe",
        )

    assert mock_run_agent.call_args.kwargs["acting_user"] == "jdoe"


@pytest.mark.asyncio
async def test_raw_rag_with_profile_does_not_raise():
    """run_question with system='raw_rag' + profile succeeds; dispatch omits
    the argument (current prod does no profile-scoped RAG for the baseline)."""
    mock_response = MagicMock()
    mock_response.response = "ACCESS is a program for HPC resources."

    mock_client = AsyncMock()
    mock_client.ask = AsyncMock(return_value=mock_response)

    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )

    with patch("src.eval.runner.get_uky_client", return_value=mock_client):
        result = await run_question(
            "q1",
            "What is ACCESS?",
            tool_catalog={"tools": []},
            system="raw_rag",
            profile=profile,
        )

    assert result.success is True
    mock_client.ask.assert_called_once()
