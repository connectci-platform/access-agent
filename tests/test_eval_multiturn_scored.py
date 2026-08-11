"""Scored multiturn path: per-turn judging, persistence, history propagation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.db import EvalDB
from src.eval.judge import JudgeResult

GOOD = JudgeResult(
    scores={
        "correctness": 2,
        "specificity": 2,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
    },
    justifications={},
    composite=0.9,
    answerable=True,
)

THREAD = {
    "thread_id": "mt-f-01",
    "description": "d",
    "questions": [
        {"turn_id": "t1", "question": "What GPU resources exist?"},
        {
            "turn_id": "t2",
            "question": "Which have A100s?",
            "ground_truth_stability": "time_bound",
        },
        {"turn_id": "t3", "question": "Walltime on that one?"},
    ],
}

STATE = {"final_answer": "An answer.", "tools_used": ["search_resources"], "messages": [1, 2]}


def _scoring(tmp_path):
    from src.eval.multiturn import ScoringContext

    db = EvalDB(f"sqlite:///{tmp_path}/eval.db")
    run = db.create_run(run_type="pre_production")
    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    return (
        ScoringContext(db=db, judge=judge, run_id=str(run.id), battery_id="test_battery"),
        db,
        judge,
    )


@pytest.mark.asyncio
async def test_scored_thread_persists_one_row_per_turn(tmp_path):
    from src.eval.multiturn import run_thread

    scoring, db, judge = _scoring(tmp_path)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        result = await run_thread(
            THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring
        )

    rows = db.get_scores_for_run(scoring.run_id)
    # Sort in the assertion — get_scores_for_run orders by created_at, which can tie
    # across three fast mocked inserts on SQLite (review finding: don't rely on it).
    assert sorted(r.question_id for r in rows) == ["mt-f-01_t1", "mt-f-01_t2", "mt-f-01_t3"]
    assert all(r.source == "judge" for r in rows)
    assert all(t.composite == 0.9 for t in result.turns)
    # Turn 3's judge call saw turns 1-2 as history
    hist = judge.score.call_args_list[2].kwargs["conversation_history"]
    assert [q for q, _ in hist] == ["What GPU resources exist?", "Which have A100s?"]

    # ground_truth_stability from the battery turn lands in the row's context (t2 has it set,
    # t1/t3 don't and should carry None rather than omit the key).
    by_qid = {r.question_id: r for r in rows}
    assert by_qid["mt-f-01_t2"].context["ground_truth_stability"] == "time_bound"
    assert by_qid["mt-f-01_t1"].context["ground_truth_stability"] is None


@pytest.mark.asyncio
async def test_failed_turn_marks_history_and_writes_skipped(tmp_path):
    from src.eval.multiturn import run_thread

    scoring, db, judge = _scoring(tmp_path)
    calls = {"n": 0}

    async def flaky_agent(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return STATE

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=flaky_agent)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    rows = {r.question_id: r for r in db.get_scores_for_run(scoring.run_id)}
    assert rows["mt-f-01_t2"].source == "skipped"
    hist = judge.score.call_args_list[-1].kwargs["conversation_history"]
    assert hist[1][1] == "(no answer — turn failed)"


@pytest.mark.asyncio
async def test_turn_reports_carry_real_turn_index(tmp_path):
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    reporter = AsyncMock()
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=reporter),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    indexes = [c.kwargs["turn_index"] for c in reporter.call_args_list]
    assert indexes == [1, 2, 3]


@pytest.mark.asyncio
async def test_scored_battery_creates_run_and_composites(tmp_path):
    from src.eval import multiturn

    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD, {**THREAD, "thread_id": "mt-f-02"}]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    assert summary is not None
    assert summary["run_id"].startswith("mtloop-")
    assert summary["composite_score"] == pytest.approx(0.9)
    assert summary["thread_composites"] == {
        "mt-f-01": pytest.approx(0.9),
        "mt-f-02": pytest.approx(0.9),
    }
    db = EvalDB(db_url)
    run = db.get_run(summary["run_id"])
    assert run.metadata_["mode"] == "multiturn"
    assert run.question_count == 6


@pytest.mark.asyncio
async def test_all_failed_thread_excluded_from_run_composite(tmp_path):
    from src.eval import multiturn

    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD, {**THREAD, "thread_id": "mt-dead"}]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    async def agent_by_thread(**kwargs):
        if "mt-dead" in kwargs["session_id"]:
            raise RuntimeError("dead thread")
        return STATE

    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=agent_by_thread)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    assert summary["unscored_threads"] == ["mt-dead"]
    assert summary["composite_score"] == pytest.approx(0.9)  # dead thread excluded, not 0.45


@pytest.mark.asyncio
async def test_run_composite_is_mean_of_thread_means_not_turns(tmp_path):
    """Long-thread non-dominance: a 3-turn thread and a 1-turn thread weigh equally."""
    from src.eval import multiturn
    from src.eval.judge import JudgeResult

    short_thread = {
        "thread_id": "mt-short",
        "questions": [{"turn_id": "t1", "question": "One?"}],
    }
    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD, short_thread]))  # 3 turns + 1 turn
    db_url = f"sqlite:///{tmp_path}/eval.db"

    def _result(composite):
        return JudgeResult(
            scores={
                "correctness": 2,
                "specificity": 2,
                "relevance": 2,
                "citation_quality": 2,
                "hedging": 1,
            },
            justifications={},
            composite=composite,
            answerable=True,
        )

    # 3-turn thread scores 0.3 per turn; 1-turn thread scores 0.9.
    judge = MagicMock()
    judge.score = AsyncMock(side_effect=[_result(0.3)] * 3 + [_result(0.9)])
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    # Mean of thread means = (0.3 + 0.9) / 2 = 0.6; mean of turns would be 0.45.
    assert summary["composite_score"] == pytest.approx(0.6)


def _judge_result(composite=0.9, answerable=True, correctness=2):
    return JudgeResult(
        scores={
            "correctness": correctness,
            "specificity": 2,
            "relevance": 2,
            "citation_quality": 2,
            "hedging": 1,
        },
        justifications={},
        composite=composite,
        answerable=answerable,
    )


def _tool_result(turn):
    """A rebuilt tool_results entry, keyed by its originating tool_call id (step_id)."""
    return {
        "step_id": f"call-turn{turn}",
        "tool_name": f"tool_turn{turn}",
        "data": [f"payload-turn{turn}"],
        "success": True,
    }


@pytest.mark.asyncio
async def test_judge_context_is_per_turn_delta_not_cumulative(tmp_path):
    """Under checkpointing the state is thread-cumulative; the judge must see only
    the current turn's tool results and trace entries (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    # Distinguishable, growing state: turn N's state holds turns 1..N's entries.
    states = [
        {
            **STATE,
            "tool_results": [_tool_result(n) for n in range(1, turn + 1)],
            "node_trace": [{"node": f"trace_turn{n}"} for n in range(1, turn + 1)],
        }
        for turn in (1, 2, 3)
    ]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    turn2 = judge.score.call_args_list[1].kwargs
    assert "tool_turn2" in turn2["tool_results"]
    assert "payload-turn2" in turn2["tool_results"]
    # Turn 1's cumulative entries must NOT appear as support for turn 2's answer.
    assert "tool_turn1" not in turn2["tool_results"]
    assert "trace_turn2" in turn2["node_trace"]
    assert "trace_turn1" not in turn2["node_trace"]

    turn3 = judge.score.call_args_list[2].kwargs
    assert "tool_turn3" in turn3["tool_results"]
    assert "tool_turn1" not in turn3["tool_results"]
    assert "tool_turn2" not in turn3["tool_results"]


@pytest.mark.asyncio
async def test_delta_survives_compaction_shrinking_tool_results(tmp_path):
    """SummarizationMiddleware compaction shrinks the REBUILT tool_results mid-thread,
    so turn 3's list can be shorter than turn 2's count. A count-slice would drop
    turn 3's own calls entirely; the identity diff still surfaces them (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    states = [
        # t1: one call. t2: two cumulative calls (count now 2).
        {**STATE, "tool_results": [_tool_result(1)]},
        {**STATE, "tool_results": [_tool_result(1), _tool_result(2)]},
        # t3: compaction dropped turns 1-2 from the thread, so the rebuild holds ONE
        # entry — this turn's — and len(1) < the prior count of 2.
        {**STATE, "tool_results": [_tool_result(3)]},
    ]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    turn3 = judge.score.call_args_list[2].kwargs
    assert "tool_turn3" in turn3["tool_results"]
    assert "payload-turn3" in turn3["tool_results"]


@pytest.mark.asyncio
async def test_empty_answer_turn_advances_the_delta_baseline(tmp_path):
    """A turn that returns state with no answer still grew the checkpoint — its calls
    must not resurface as the NEXT turn's evidence (D1 baseline rule)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    states = [
        # t1 ran a tool but produced no answer (success=False, no exception).
        {**STATE, "final_answer": "", "tool_results": [_tool_result(1)]},
        {**STATE, "tool_results": [_tool_result(1), _tool_result(2)]},
        {**STATE, "tool_results": [_tool_result(1), _tool_result(2), _tool_result(3)]},
    ]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    # t1 wasn't judged (no answer -> skipped row), so t2 is the first judge call.
    turn2 = judge.score.call_args_list[0].kwargs
    assert "tool_turn2" in turn2["tool_results"]
    assert "tool_turn1" not in turn2["tool_results"]


@pytest.mark.asyncio
async def test_turn_reports_carry_only_that_turns_tool_calls(tmp_path):
    """report_battery_turn gets the same delta view the judge does, so turn 2's report
    can't be credited with turn 1's calls (invoked_write / capability inference)."""
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    reporter = AsyncMock()
    states = [
        {**STATE, "tool_results": [_tool_result(n) for n in range(1, turn + 1)]}
        for turn in (1, 2, 3)
    ]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=reporter),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    reported = [
        [r["tool_name"] for r in c.kwargs["state"]["tool_results"]] for c in reporter.call_args_list
    ]
    assert reported == [["tool_turn1"], ["tool_turn2"], ["tool_turn3"]]
    # The rest of the state still rides along — the report needs final_answer/tools_used.
    assert reporter.call_args_list[1].kwargs["state"]["final_answer"] == "An answer."


@pytest.mark.asyncio
async def test_ran_but_failed_turn_reports_its_real_delta_state(tmp_path):
    """A turn that ran tools but produced no answer reports its real state, not {} —
    only the exception path (no state at all) reports empty."""
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    reporter = AsyncMock()

    async def agent(**kwargs):
        if kwargs["question_id"].endswith("_t2"):
            raise RuntimeError("boom")
        return {**STATE, "final_answer": "", "tool_results": [_tool_result(1)]}

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=agent)),
        patch("src.eval.multiturn.report_battery_turn", new=reporter),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    by_qid = {c.kwargs["question_id"]: c.kwargs["state"] for c in reporter.call_args_list}
    # Ran-but-empty: real state with this turn's call.
    assert [r["tool_name"] for r in by_qid["mt-f-01_t1"]["tool_results"]] == ["tool_turn1"]
    # Exception: no state exists.
    assert by_qid["mt-f-01_t2"] == {}


@pytest.mark.asyncio
async def test_turn_capture_reset_per_turn(tmp_path):
    """turn_reports carry per-turn summarized/timings/chunks only if the capture is
    reset before each turn (mirrors the single-turn runner)."""
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    resets = MagicMock()
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.reset_turn_capture", new=resets),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    assert resets.call_count == 3  # one per turn, not once per thread


@pytest.mark.asyncio
async def test_unfair_turns_excluded_from_composites_and_counted(tmp_path):
    """Judge-screened (answerable=False) turns leave composites alone but are counted."""
    from src.eval import multiturn

    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    judge = MagicMock()
    # t1 Fair 0.8, t2 Unfair 0.1 (must not drag the mean), t3 Fair 0.6.
    judge.score = AsyncMock(
        side_effect=[
            _judge_result(0.8),
            _judge_result(0.1, answerable=False),
            _judge_result(0.6),
        ]
    )
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    # Fair-only mean = (0.8 + 0.6) / 2 = 0.7; including the Unfair turn would give 0.5.
    assert summary["thread_composites"]["mt-f-01"] == pytest.approx(0.7)
    assert summary["composite_score"] == pytest.approx(0.7)
    assert summary["screened_turns"] == 1
    # The screened row still persists so the screen is auditable.
    rows = EvalDB(db_url).get_scores_for_run(summary["run_id"])
    assert {r.question_id: r.source for r in rows}["mt-f-01_t2"] == "judge"


@pytest.mark.asyncio
async def test_scores_summary_carries_full_consumer_contract(tmp_path):
    """per_dimension (Fair turns only) + screened/failed counts land in scores_summary."""
    from src.eval import multiturn

    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    calls = {"n": 0}

    async def flaky_agent(**kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return STATE

    judge = MagicMock()
    # Two judged turns: correctness 2 and 0 -> per_dimension mean 1.0.
    judge.score = AsyncMock(
        side_effect=[_judge_result(correctness=2), _judge_result(correctness=0)]
    )
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=flaky_agent)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    stored = EvalDB(db_url).get_run(summary["run_id"]).scores_summary
    assert set(stored) == {
        "per_dimension",
        "thread_composites",
        "unscored_threads",
        "screened_turns",
        "failed_turns",
    }
    assert stored["per_dimension"]["correctness"] == pytest.approx(1.0)
    assert stored["per_dimension"]["hedging"] == pytest.approx(1.0)
    assert stored["failed_turns"] == {"mt-f-01": 1}
    assert stored["screened_turns"] == 0


@pytest.mark.asyncio
async def test_scored_run_refuses_draft_author_facts(tmp_path):
    """A resolved fact still carrying AUTHOR: is an authoring instruction, not ground
    truth — refuse before the first agent call, listing the offending questions."""
    from src.eval import multiturn

    thread = {
        "thread_id": "mt-draft",
        "questions": [
            {
                "turn_id": "t1",
                "question": "Q1?",
                "required_facts": ["Answer names X (AUTHOR: TBD)"],
            },
            {"turn_id": "t2", "question": "Q2?", "required_facts": ["Answer names Y."]},
        ],
    }
    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([thread]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    agent = AsyncMock(return_value=STATE)
    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    with (
        patch("src.eval.multiturn.run_agent", new=agent),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        with pytest.raises(ValueError, match="mt-draft_t1"):
            await multiturn.run_battery(battery_path=str(battery), score=True, database_url=db_url)
        assert agent.call_count == 0  # true pre-flight: refused before any agent call

        # The override runs the battery.
        _, summary = await multiturn.run_battery(
            battery_path=str(battery),
            score=True,
            database_url=db_url,
            allow_draft_facts=True,
        )
    assert agent.call_count == 2
    assert summary["thread_composites"]["mt-draft"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_draft_guard_ignores_authoring_notes(tmp_path):
    """authoring_notes legitimately keep AUTHOR markers after authoring — the scan
    targets fact text only."""
    from src.eval import multiturn

    thread = {
        "thread_id": "mt-notes",
        "questions": [
            {
                "turn_id": "t1",
                "question": "Q1?",
                "required_facts": ["Answer names X."],
                "authoring_notes": ["AUTHOR: verify the list live before the run"],
            }
        ],
    }
    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([thread]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    assert summary["thread_composites"]["mt-notes"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_facts_resolved_once_and_reused_per_turn(tmp_path):
    """Pre-resolution is reused during the run: one resolve per turn, not two."""
    from src.eval import multiturn

    thread = {
        "thread_id": "mt-reuse",
        "questions": [
            {"turn_id": "t1", "question": "Q1?", "required_facts": ["Fact A."]},
            {"turn_id": "t2", "question": "Q2?", "required_facts": ["Fact B."]},
        ],
    }
    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([thread]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    resolver = MagicMock(side_effect=lambda db, qid, yaml_facts: yaml_facts)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
        patch("src.eval.multiturn.resolve_required_facts", new=resolver),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        await multiturn.run_battery(battery_path=str(battery), score=True, database_url=db_url)

    assert resolver.call_count == 2  # pre-flight only; the run reuses the map
    assert judge.score.call_args_list[0].kwargs["required_facts"] == ["Fact A."]
    assert judge.score.call_args_list[1].kwargs["required_facts"] == ["Fact B."]


def test_compare_renders_per_dimension_for_both_summary_shapes(tmp_path, capsys):
    """`eval compare` between a single-turn and a multiturn run must show real
    per-dimension values: single-turn stores the dims AS scores_summary, multiturn
    nests them under per_dimension."""
    from src.eval.__main__ import _handle_compare

    db_url = f"sqlite:///{tmp_path}/eval.db"
    db = EvalDB(db_url)
    single = db.create_run(run_type="pre_production", metadata_={"system": "agent_full"})
    db.update_run_summary(
        str(single.id),
        {
            "correctness": 1.5,
            "specificity": 1.0,
            "relevance": 2.0,
            "citation_quality": 1.0,
            "hedging": 1.0,
        },
        0.62,
    )
    multi = db.create_run(run_type="pre_production", metadata_={"mode": "multiturn"})
    db.update_run_summary(
        str(multi.id),
        {
            "per_dimension": {
                "correctness": 2.0,
                "specificity": 1.0,
                "relevance": 2.0,
                "citation_quality": 1.0,
                "hedging": 1.0,
            },
            "thread_composites": {"mt-f-01": 0.8},
            "unscored_threads": [],
            "screened_turns": 1,
            "failed_turns": {},
        },
        0.8,
    )

    args = MagicMock(run_a=str(single.id), run_b=str(multi.id))
    with patch("src.config.settings.DATABASE_URL", db_url):
        _handle_compare(args)
    out = capsys.readouterr().out

    # Both columns render real values — a raw scores_summary read would give the
    # multiturn side 0.00 for every dimension.
    rows = {
        line.split()[0]: [float(v) for v in line.split()[1:3]]
        for line in out.splitlines()
        if line.strip().startswith(("correctness", "relevance", "hedging"))
    }
    assert rows["correctness"] == [1.5, 2.0]
    assert rows["relevance"] == [2.0, 2.0]
    assert rows["hedging"] == [1.0, 1.0]
