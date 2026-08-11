"""Test for rejudge_run (v2 rubric replay).

Seeds an original run with one frozen judge score, mocks the judge LLM so no
network call is made, and replays it through rejudge_run. Verifies the new run
is linked back via metadata.rejudged_from and that the rescored row carries the
v2 fields (specificity, specificity_na, answerable, rubric_version=2, composite).
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval import scoring
from src.eval.db import EvalDB
from src.eval.rejudge import rejudge_run

# v2 judge payload: fair answer, actionable specificity, all-best -> composite 1.0.
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


def _completion(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


@pytest.fixture
def mock_db(tmp_path):
    return f"sqlite:///{tmp_path}/test_rejudge.db"


async def test_rejudge_run_replays_and_writes_v2(mock_db):
    """Rejudge an original run: new run links back and rescored row is v2."""
    db = EvalDB(mock_db)

    # Seed an original run with one frozen judge score (question/answer/context).
    original = db.create_run(
        run_type="pre_production",
        agent_commit="abc123",
        agent_branch="main",
        llm_model="qwen",
        judge_model="gpt-4o-mini",
        question_set="tiny",
        question_count=1,
        composite_score=0.5,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="ACCESS is a program for HPC resources.",
        context={"rag_context": "some docs", "tool_results": None, "node_trace": None},
        context_completeness=1.0,
        correctness=1,
        specificity=1,
        specificity_na=False,
        answerable=True,
        rubric_version=2,
        relevance=1,
        citation_quality=1,
        hedging=1,
        composite_score=0.5,
        duration_ms=123,
    )

    # Mock the judge LLM so no network call is made (same pattern as the
    # integration test: patch AsyncOpenAI where Judge constructs it).
    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [_completion(MOCK_JUDGE_BEST)]
        mock_openai_cls.return_value = mock_client

        summary = await rejudge_run(
            original_run_id=str(original.id),
            database_url=mock_db,
        )

    # Summary reflects one rescored item.
    assert summary["rescored"] == 1
    assert summary["skipped"] == 0
    assert summary["errors"] == 0
    assert summary["original_run_id"] == str(original.id)
    new_run_id = str(summary["new_run_id"])
    assert new_run_id != str(original.id)

    # New run is a rejudge run linked back to the original via metadata.
    new_run = db.get_run(new_run_id)
    assert new_run is not None
    assert new_run.run_type == "rejudge"
    assert new_run.metadata_["rejudged_from"] == str(original.id)
    assert new_run.metadata_["system"] == "access-agent"
    assert new_run.composite_score is not None

    # The rescored row carries the v2 fields.
    new_scores = db.get_scores_for_run(new_run_id)
    assert len(new_scores) == 1
    rescored = new_scores[0]
    assert rescored.source == "judge"
    assert rescored.question_id == "q1"
    # Stamped from the module constant, never a literal at the call site.
    assert rescored.rubric_version == scoring.RUBRIC_VERSION
    assert rescored.answerable is True
    assert rescored.specificity == 2  # "Actionable"
    assert rescored.specificity_na is False
    assert rescored.composite_score is not None
    assert abs(rescored.composite_score - 1.0) < 1e-9


async def test_rejudge_replays_required_facts(mock_db):
    """The scorer freezes required_facts in context; rejudge must pass them back
    to the judge so rejudged runs keep fact-grounded verdicts (and the right
    token budget)."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        agent_commit="abc123",
        agent_branch="main",
        llm_model="qwen",
        judge_model="gpt-4o-mini",
        question_set="tiny",
        question_count=1,
        composite_score=0.5,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="ACCESS is a program.",
        context={
            "rag_context": None,
            "tool_results": None,
            "node_trace": None,
            "required_facts": [{"fact_id": 42, "fact_text": "ACCESS allocates HPC resources"}],
        },
        context_completeness=1.0,
        correctness=1,
        specificity=1,
        specificity_na=False,
        answerable=True,
        rubric_version=2,
        relevance=1,
        citation_quality=1,
        hedging=1,
        composite_score=0.5,
        duration_ms=123,
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        await rejudge_run(original_run_id=str(original.id), database_url=mock_db)
        # The prompt sent to the judge must contain the stored fact (and its stable id).
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "ACCESS allocates HPC resources" in sent
        assert "Required Facts" in sent


async def test_rejudge_propagates_mode_metadata(mock_db):
    """A rejudged multiturn run must stay excluded from the dashboard aggregate, so
    metadata.mode rides along to the new run."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=1,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-f-01_t2",
        source="judge",
        question_text="Which have A100s?",
        answer_text="Delta and DeltaAI.",
        context={"rag_context": None, "tool_results": None, "node_trace": None},
        composite_score=0.5,
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    new_run = db.get_run(str(summary["new_run_id"]))
    assert new_run.metadata_["mode"] == "multiturn"
    assert new_run.metadata_["rejudged_from"] == str(original.id)


async def test_rejudge_replays_conversation_history(mock_db):
    """Multiturn rows persist the prior transcript; replaying it keeps
    reference-resolution turns judgeable instead of recording phantom drops."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=1,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-f-01_t2",
        source="judge",
        question_text="Which of those have A100s?",
        answer_text="Delta and DeltaAI.",
        # Stored as list-of-lists — JSON has no tuples.
        context={
            "rag_context": None,
            "tool_results": None,
            "node_trace": None,
            "conversation_history": [["What GPU resources exist?", "Delta, DeltaAI, Anvil."]],
        },
        composite_score=0.5,
    )

    judge = MagicMock()
    judge.score = AsyncMock(return_value=None)  # judge_error path; we only assert the call
    with patch("src.eval.rejudge.Judge", return_value=judge):
        await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    history = judge.score.call_args.kwargs["conversation_history"]
    assert history == [("What GPU resources exist?", "Delta, DeltaAI, Anvil.")]


async def test_rejudge_multiturn_rebuilds_summary_under_macro_semantics(mock_db):
    """Identical verdicts must report ~zero drift. The stored composite is a macro
    mean of Fair-only thread composites; scoring the rejudge as an all-rows micro
    mean would invent a delta out of thread-length imbalance alone."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=3,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    # Deliberately uneven: thread A has 2 turns, thread B has 1. Every turn scores
    # composite 1.0 under the mocked judge, so macro == micro == 1.0 only if the
    # rebuild groups by thread; the point is the CONTRACT shape plus a zero delta.
    for thread_id, turn_ids in (("mt-a", ("t1", "t2")), ("mt-b", ("t1",))):
        for turn_id in turn_ids:
            db.add_score(
                run_id=original.id,
                question_id=f"{thread_id}_{turn_id}",
                source="judge",
                question_text="Which of those have A100s?",
                answer_text="Delta and DeltaAI.",
                context={
                    "rag_context": None,
                    "tool_results": None,
                    "node_trace": None,
                    "thread_id": thread_id,
                },
                composite_score=1.0,
            )
    db.update_run_summary(
        str(original.id),
        {"per_dimension": {}, "thread_composites": {"mt-a": 1.0, "mt-b": 1.0}},
        1.0,
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    # Identical verdicts, so no phantom judge drift.
    assert summary["delta"] == 0.0
    assert summary["new_composite"] == 1.0

    stored = db.get_run(str(summary["new_run_id"])).scores_summary
    # Rebuilt under the full multiturn contract, not a flat dimension map.
    assert set(stored) == {
        "per_dimension",
        "thread_composites",
        "unscored_threads",
        "screened_turns",
        "failed_turns",
    }
    assert stored["thread_composites"] == {"mt-a": 1.0, "mt-b": 1.0}
    assert stored["per_dimension"]["correctness"] == 2.0


async def test_rejudge_multiturn_macro_differs_from_micro(mock_db):
    """The rebuilt composite is the mean of THREAD means, so a long thread cannot
    dominate — the same rule the live runner applies."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=4,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    # 3-turn thread scores 0.0 per turn, 1-turn thread scores 1.0.
    # Macro = (0.0 + 1.0)/2 = 0.5; a micro mean over 4 rows would give 0.25.
    for thread_id, turn_ids in (("mt-long", ("t1", "t2", "t3")), ("mt-short", ("t1",))):
        for turn_id in turn_ids:
            db.add_score(
                run_id=original.id,
                question_id=f"{thread_id}_{turn_id}",
                source="judge",
                question_text="q",
                answer_text="a",
                context={"thread_id": thread_id},
                composite_score=0.5,
            )

    worst = json.dumps(
        {
            "answerable": "Fair",
            "correctness": {"value": "Incorrect", "justification": "wrong"},
            "specificity": {"value": "Generic", "justification": "vague"},
            "relevance": {"value": "Off", "justification": "off"},
            "citation_quality": {"value": "Poor", "justification": "none"},
            "hedging": {"value": "Miscalibrated", "justification": "hedged"},
        }
    )
    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        # get_scores_for_run orders by created_at; the 3 mt-long rows were inserted first.
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[_completion(worst)] * 3 + [_completion(MOCK_JUDGE_BEST)]
        )
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    stored = db.get_run(str(summary["new_run_id"])).scores_summary
    assert stored["thread_composites"]["mt-long"] == pytest.approx(0.0)
    assert stored["thread_composites"]["mt-short"] == pytest.approx(1.0)
    assert summary["new_composite"] == pytest.approx(0.5)  # micro would be 0.25


async def test_rejudge_single_turn_summary_stays_flat(mock_db):
    """Single-turn rejudge behavior is unchanged: a flat dimension map, micro mean."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="tiny",
        question_count=1,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="A program.",
        context={"rag_context": None, "tool_results": None, "node_trace": None},
        composite_score=0.5,
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    stored = db.get_run(str(summary["new_run_id"])).scores_summary
    assert "thread_composites" not in stored
    assert stored["correctness"] == 2.0


async def test_rejudge_row_carries_new_judge_fact_verdicts(mock_db):
    """fact_verdicts are a judge OUTPUT. The replayed transcript stays the source
    row's, but the verdicts on the rejudged row must be the NEW judge's — otherwise
    a rejudge reports the old judge's per-fact calls next to new dimension scores."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="tiny",
        question_count=1,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="A program.",
        context={
            "rag_context": "docs",
            "required_facts": [{"fact_id": 7, "fact_text": "ACCESS allocates HPC resources"}],
            "conversation_history": [["earlier q", "earlier a"]],
            # The OLD judge said the fact was missing.
            "fact_verdicts": [{"id": "7", "verdict": "missing", "justification": "old judge"}],
        },
        composite_score=0.5,
    )

    new_verdicts = [{"id": "7", "verdict": "supported", "justification": "new judge"}]
    judge_result = MagicMock()
    judge_result.scores = dict.fromkeys(
        ("correctness", "specificity", "relevance", "citation_quality", "hedging"), 1
    )
    judge_result.specificity_na = False
    judge_result.answerable = True
    judge_result.composite = 0.5
    judge_result.justifications = {}
    judge_result.fact_verdicts = new_verdicts

    judge = MagicMock()
    judge.score = AsyncMock(return_value=judge_result)
    with patch("src.eval.rejudge.Judge", return_value=judge):
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    rescored = db.get_scores_for_run(str(summary["new_run_id"]))[0]
    assert rescored.context["fact_verdicts"] == new_verdicts
    # Only fact_verdicts is replaced — the replayed transcript stays the original's.
    assert rescored.context["rag_context"] == "docs"
    assert rescored.context["conversation_history"] == [["earlier q", "earlier a"]]
    assert rescored.context["required_facts"][0]["fact_text"] == "ACCESS allocates HPC resources"
    # No regression on the rubric stamp (fix 4).
    assert rescored.rubric_version == scoring.RUBRIC_VERSION

    # The SOURCE row is untouched — rejudge never mutates the run it replays.
    assert db.get_scores_for_run(str(original.id))[0].context["fact_verdicts"] == [
        {"id": "7", "verdict": "missing", "justification": "old judge"}
    ]


async def test_rejudge_materializes_skipped_rows_so_failed_turns_survive(mock_db):
    """A rejudge-of-a-rejudge must report the same failed_turns. skipped rows are not
    judgeable, so they are copied forward; without that the failure record vanishes
    from the new run and the next generation reports a clean thread."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=2,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t1",
        source="judge",
        question_text="q1",
        answer_text="a1",
        context={"thread_id": "mt-a", "turn_index": 1},
        composite_score=1.0,
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t2",
        source="skipped",
        question_text="q2",
        answer_text="boom",
        context={"thread_id": "mt-a", "turn_index": 2},
        justifications={"error": "boom"},
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        gen1 = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    gen1_id = str(gen1["new_run_id"])
    assert db.get_run(gen1_id).scores_summary["failed_turns"] == {"mt-a": 1}

    # The skipped row was materialized into the rejudged run, with its text and context.
    carried = [s for s in db.get_scores_for_run(gen1_id) if s.source == "skipped"]
    assert len(carried) == 1
    assert carried[0].question_id == "mt-a_t2"
    assert carried[0].question_text == "q2"
    assert carried[0].answer_text == "boom"
    assert carried[0].context["thread_id"] == "mt-a"

    # Rejudging the rejudge reports the SAME failed_turns.
    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        gen2 = await rejudge_run(original_run_id=gen1_id, database_url=mock_db)

    assert db.get_run(str(gen2["new_run_id"])).scores_summary["failed_turns"] == {"mt-a": 1}


async def test_rejudge_excludes_human_rows_from_multiturn_rebuild(mock_db):
    """Human review rows describe a question a judge row already covers. Ingesting
    them would invent a phantom thread with no Fair turn, reporting the run as
    partly unscored when every turn actually scored."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=1,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t1",
        source="judge",
        question_text="q1",
        answer_text="a1",
        context={"thread_id": "mt-a", "turn_index": 1},
        composite_score=1.0,
    )
    # A human reviewer scored the same question later. No thread_id, source="human".
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t1",
        source="human",
        reviewer_id="drew",
        question_text="q1",
        answer_text="a1",
        context={},
        composite_score=0.75,
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    stored = db.get_run(str(summary["new_run_id"])).scores_summary
    # No "(no thread)" bucket, and the human row did not become an unscored thread.
    assert list(stored["thread_composites"]) == ["mt-a"]
    assert stored["unscored_threads"] == []
    assert stored["failed_turns"] == {}


async def test_rejudge_single_turn_passes_no_history(mock_db):
    """Single-turn rows have no stored history — the judge call must stay unchanged."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="tiny",
        question_count=1,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="A program.",
        context={"rag_context": None, "tool_results": None, "node_trace": None},
        composite_score=0.5,
    )

    judge = MagicMock()
    judge.score = AsyncMock(return_value=None)
    with patch("src.eval.rejudge.Judge", return_value=judge):
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    assert judge.score.call_args.kwargs["conversation_history"] is None
    # No mode on the source run means no mode key invented on the rejudge run.
    assert "mode" not in (db.get_run(str(summary["new_run_id"])).metadata_ or {})


async def test_rejudge_materializes_judge_error_rows_so_failed_turns_survive(mock_db):
    """judge_error originals are as un-rejudgeable as skipped ones — there is no
    answer the new judge could score differently. Leaving them out would decay
    failed_turns to zero across generations exactly like a dropped skipped row."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=2,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t1",
        source="judge",
        question_text="q1",
        answer_text="a1",
        context={"thread_id": "mt-a", "turn_index": 1},
        composite_score=1.0,
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t2",
        source="judge_error",
        question_text="q2",
        answer_text="a2",
        context={"thread_id": "mt-a", "turn_index": 2, "node_trace": "trace"},
        justifications={"error": "judge returned nothing"},
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        gen1 = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    gen1_id = str(gen1["new_run_id"])
    assert db.get_run(gen1_id).scores_summary["failed_turns"] == {"mt-a": 1}

    carried = [s for s in db.get_scores_for_run(gen1_id) if s.source == "judge_error"]
    assert len(carried) == 1
    assert carried[0].question_id == "mt-a_t2"
    assert carried[0].answer_text == "a2"
    assert carried[0].context["thread_id"] == "mt-a"
    assert carried[0].context["node_trace"] == "trace"

    # The next generation reports the SAME failed_turns — self-contained, no decay.
    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        gen2 = await rejudge_run(original_run_id=gen1_id, database_url=mock_db)

    assert db.get_run(str(gen2["new_run_id"])).scores_summary["failed_turns"] == {"mt-a": 1}


async def test_rejudge_error_row_drops_the_source_fact_verdicts(mock_db):
    """A row that FAILS under the new judge writes a fresh judge_error row. The new
    judge produced no verdicts, so the row carries none — keeping the source row's
    would present the old judge's per-fact calls as this generation's."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="tiny",
        question_count=1,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="A program.",
        context={
            "rag_context": "docs",
            "required_facts": [{"fact_id": "f1", "fact_text": "ACCESS allocates HPC"}],
            "conversation_history": [["earlier q", "earlier a"]],
            "fact_verdicts": [{"id": "f1", "verdict": "yes", "justification": "old judge"}],
        },
        composite_score=0.5,
    )

    judge = MagicMock()
    judge.score = AsyncMock(return_value=None)  # the new judge fails on this row
    with patch("src.eval.rejudge.Judge", return_value=judge):
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    row = db.get_scores_for_run(str(summary["new_run_id"]))[0]
    assert row.source == "judge_error"
    assert "fact_verdicts" not in row.context
    # Everything else replayed stays, so the row is manually re-scorable.
    assert row.context["rag_context"] == "docs"
    assert row.context["conversation_history"] == [["earlier q", "earlier a"]]
    assert row.context["required_facts"][0]["fact_id"] == "f1"


async def test_rejudge_warns_about_turn_rows_without_a_thread_id(mock_db, caplog):
    """A turn row with no thread_id cannot reach any composite. It must not vanish
    silently, and it must not invent a phantom thread either."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=2,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-a_t1",
        source="judge",
        question_text="q1",
        answer_text="a1",
        context={"thread_id": "mt-a", "turn_index": 1},
        composite_score=1.0,
    )
    db.add_score(
        run_id=original.id,
        question_id="orphan_t1",
        source="judge",
        question_text="q2",
        answer_text="a2",
        context={"turn_index": 1},  # no thread_id
        composite_score=1.0,
    )

    with (
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
        caplog.at_level(logging.WARNING, logger="src.eval.rejudge"),
    ):
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    assert "orphan_t1" in caplog.text
    assert "no context.thread_id" in caplog.text

    stored = db.get_run(str(summary["new_run_id"])).scores_summary
    assert list(stored["thread_composites"]) == ["mt-a"]  # no phantom thread
    assert stored["unscored_threads"] == []
    # The row itself is still persisted in the new run — excluded from the summary,
    # not from the data.
    assert any(
        s.question_id == "orphan_t1" for s in db.get_scores_for_run(str(summary["new_run_id"]))
    )
