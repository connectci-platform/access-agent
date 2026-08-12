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
async def test_judge_failure_marks_score_source_judge_error(tmp_path):
    """When the judge itself returns None (a transient judge outage),
    score_and_persist_turn persists a judge_error row and the turn's
    score_source records that — distinct from a agent-side "skipped" turn."""
    from src.eval.multiturn import run_thread

    scoring, db, judge = _scoring(tmp_path)
    judge.score = AsyncMock(return_value=None)
    thread = {**THREAD, "questions": THREAD["questions"][:1]}

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        result = await run_thread(
            thread, tool_catalog=None, session_namespace="ns", scoring=scoring
        )

    assert result.turns[0].score_source == "judge_error"
    assert result.turns[0].composite is None
    row = db.get_scores_for_run(scoring.run_id)[0]
    assert row.source == "judge_error"


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


def _tool_result(turn, message_id, step_id=None):
    """A rebuilt tool_results entry as _build_tool_results emits it.

    ``message_id`` is the source ToolMessage's LangChain id — the field the
    boundary delta slices on. ``step_id`` defaults to a per-turn unique id
    but is overridable so tests can reproduce the production vLLM pattern of
    ids that REPEAT across turns.
    """
    return {
        "step_id": step_id or f"call-turn{turn}",
        "tool_name": f"tool_turn{turn}",
        "data": [f"payload-turn{turn}"],
        "success": True,
        "message_id": message_id,
    }


class _Msg:
    """Minimal stand-in for a LangChain message (the delta reads ``.type``/``.id``)."""

    def __init__(self, type_, id_=None):
        self.type = type_
        self.id = id_


def _Human(id_=None):
    return _Msg("human", id_)


def _Other(id_=None):
    return _Msg("ai", id_)


def _tool_msg_id(turn):
    """The id of turn N's ToolMessage, matching what _thread_state builds."""
    return f"msg-turn{turn}-tool"


def _thread_state(turns, *, tool_results_override=None, **overrides):
    """A cumulative checkpointed state after ``turns`` turns.

    Each turn contributes Human, AI, Tool, AI messages, all id-stamped the way
    ``add_messages`` guarantees. node_trace grows by exactly one entry per turn
    (the loop appends one).
    """
    messages = []
    for n in range(1, turns + 1):
        messages += [
            _Human(f"msg-turn{n}-human"),
            _Other(f"msg-turn{n}-ai"),
            _Other(_tool_msg_id(n)),
            _Other(f"msg-turn{n}-final"),
        ]
    return {
        **STATE,
        "messages": messages,
        "tool_results": (
            tool_results_override
            if tool_results_override is not None
            else [_tool_result(n, _tool_msg_id(n)) for n in range(1, turns + 1)]
        ),
        "node_trace": [{"node": f"trace_turn{n}"} for n in range(1, turns + 1)],
        **overrides,
    }


@pytest.mark.asyncio
async def test_judge_context_is_per_turn_delta_not_cumulative(tmp_path):
    """Under checkpointing the state is thread-cumulative; the judge must see only
    the current turn's tool results and trace entries (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    # Distinguishable, growing state: turn N's state holds turns 1..N's entries.
    states = [_thread_state(turn) for turn in (1, 2, 3)]

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
    # node_trace delta is the FINAL entry (the loop appends exactly one per turn).
    assert "trace_turn2" in turn2["node_trace"]
    assert "trace_turn1" not in turn2["node_trace"]

    turn3 = judge.score.call_args_list[2].kwargs
    assert "tool_turn3" in turn3["tool_results"]
    assert "tool_turn1" not in turn3["tool_results"]
    assert "tool_turn2" not in turn3["tool_results"]
    assert "trace_turn3" in turn3["node_trace"]
    assert "trace_turn2" not in turn3["node_trace"]


@pytest.mark.asyncio
async def test_delta_isolates_turns_when_tool_call_ids_repeat(tmp_path):
    """The production vLLM path emits deterministic tool_call ids (chatcmpl-tool-0)
    that REPEAT every turn. An identity diff would silently empty turn 2's delta;
    the boundary mechanism doesn't consult ids at all (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    # Both turns' calls carry the SAME provider id; only message_id separates them.
    states = [
        _thread_state(
            1,
            tool_results_override=[
                _tool_result(n, _tool_msg_id(n), step_id="chatcmpl-tool-0") for n in (1,)
            ],
        ),
        _thread_state(
            2,
            tool_results_override=[
                _tool_result(n, _tool_msg_id(n), step_id="chatcmpl-tool-0") for n in (1, 2)
            ],
        ),
        _thread_state(
            3,
            tool_results_override=[
                _tool_result(n, _tool_msg_id(n), step_id="chatcmpl-tool-0") for n in (1, 2, 3)
            ],
        ),
    ]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    turn2 = judge.score.call_args_list[1].kwargs
    assert "tool_turn2" in turn2["tool_results"]
    assert "tool_turn1" not in turn2["tool_results"]

    turn3 = judge.score.call_args_list[2].kwargs
    assert "tool_turn3" in turn3["tool_results"]
    assert "tool_turn2" not in turn3["tool_results"]


@pytest.mark.asyncio
async def test_delta_keeps_unpositioned_entry_when_no_boundary_exists():
    """An entry with no message_id in a thread with no HumanMessage is KEPT —
    unpositioned + no boundary means "show the evidence", never hide it (D1)."""
    from src.eval.multiturn import _turn_delta_state

    view = _turn_delta_state(
        {
            "messages": [_Other("a"), _Other("b")],
            "tool_results": [{"tool_name": "alpha", "message_id": ""}],
        }
    )
    assert [r["tool_name"] for r in view["tool_results"]] == ["alpha"]


@pytest.mark.asyncio
async def test_empty_answer_turns_calls_do_not_leak_into_the_next_turn(tmp_path):
    """A turn that returns state with no answer still grew the checkpoint — its calls
    must not resurface as the NEXT turn's evidence (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    states = [
        # t1 ran a tool but produced no answer (success=False, no exception).
        _thread_state(1, final_answer=""),
        _thread_state(2),
        _thread_state(3),
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
async def test_post_checkpoint_exception_does_not_leak_into_next_turn(tmp_path):
    """Turn N's calls can be durably checkpointed BEFORE turn N raises (the graph
    writes state before tail work that can fail), so they show up in turn N+1's
    cumulative state. Any advancing baseline would miss them because the raising
    turn never advanced it; the boundary derivation never had one (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, judge = _scoring(tmp_path)

    async def agent(**kwargs):
        if kwargs["question_id"].endswith("_t2"):
            # t2's calls WERE committed to the checkpoint, then the turn raised.
            raise RuntimeError("boom after checkpoint write")
        if kwargs["question_id"].endswith("_t1"):
            return _thread_state(1)
        # t3's cumulative state carries t2's committed messages AND results.
        return _thread_state(3)

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=agent)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    # t2 raised (persisted skipped, not judged), so t3 is the second judge call.
    turn3 = judge.score.call_args_list[1].kwargs
    assert "tool_turn3" in turn3["tool_results"]
    assert "tool_turn2" not in turn3["tool_results"]
    assert "tool_turn1" not in turn3["tool_results"]


@pytest.mark.asyncio
async def test_turn_reports_carry_only_that_turns_tool_calls(tmp_path):
    """report_battery_turn gets the same delta view the judge does, so turn 2's report
    can't be credited with turn 1's calls (invoked_write / capability inference)."""
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    reporter = AsyncMock()
    states = [_thread_state(turn) for turn in (1, 2, 3)]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=reporter),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    reported = [
        [r["tool_name"] for r in c.kwargs["state"]["tool_results"]] for c in reporter.call_args_list
    ]
    assert reported == [["tool_turn1"], ["tool_turn2"], ["tool_turn3"]]
    # The rest of the projection still rides along — the report needs final_answer.
    assert reporter.call_args_list[1].kwargs["state"]["final_answer"] == "An answer."


@pytest.mark.asyncio
async def test_view_tools_used_is_turn_scoped_not_the_cumulative_rescan(tmp_path):
    """state["tools_used"] is a full-thread rescan; the per-turn view derives its own
    from the delta tool_results, order-preserving and deduped (D1)."""
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    reporter = AsyncMock()
    # The cumulative rescan names every tool the thread ever called.
    states = [
        _thread_state(turn, tools_used=[f"tool_turn{n}" for n in range(1, turn + 1)])
        for turn in (1, 2, 3)
    ]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=reporter),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    reported = [c.kwargs["state"]["tools_used"] for c in reporter.call_args_list]
    assert reported == [["tool_turn1"], ["tool_turn2"], ["tool_turn3"]]


def test_view_tools_used_dedupes_preserving_call_order():
    """Two calls to the same tool in one turn collapse to one name, in call order."""
    from src.eval.multiturn import _turn_delta_state

    state = {
        "messages": [_Human("h"), _Other("m1"), _Other("m2"), _Other("m3")],
        "tool_results": [
            {"tool_name": "beta", "message_id": "m1"},
            {"tool_name": "alpha", "message_id": "m2"},
            {"tool_name": "beta", "message_id": "m3"},
        ],
        "tools_used": ["everything", "else"],
    }
    assert _turn_delta_state(state)["tools_used"] == ["beta", "alpha"]


def test_projection_does_not_alias_the_checkpointed_state():
    """The view is an explicit projection whose per-turn slices are deep-copied —
    mutating it, INCLUDING the nested dicts inside an entry or a trace entry, must
    not corrupt the checkpointed state the next turn builds on (D1)."""
    from src.eval.multiturn import _turn_delta_state

    entry = {"tool_name": "alpha", "message_id": "m1", "data": {"rows": ["payload"]}}
    state = {
        "messages": [_Human("h"), _Other("m1"), _Other("m2")],
        "tool_results": [entry],
        "tools_used": ["alpha"],
        "node_trace": [{"node": "t1"}, {"node": "t2", "tools_called": ["alpha"]}],
        "rag_matches": [],
        "model_calls": [{"model": "m"}],
        "final_answer": "An answer.",
    }
    view = _turn_delta_state(state)

    view["tool_results"].append({"tool_name": "injected", "message_id": "m9"})
    view["tools_used"].append("injected")
    view["node_trace"].append({"node": "injected"})
    view["model_calls"].append({"model": "injected"})
    view["rag_matches"].append("injected")
    view["final_answer"] = "clobbered"

    # Nested mutation of the surviving delta entry and the trace entry.
    view["tool_results"][0]["tool_name"] = "clobbered"
    view["tool_results"][0]["data"]["rows"].append("injected")
    view["node_trace"][0]["node"] = "clobbered"
    view["node_trace"][0]["tools_called"].append("injected")

    assert [r["tool_name"] for r in state["tool_results"]] == ["alpha"]
    assert state["tool_results"][0]["data"] == {"rows": ["payload"]}
    assert state["tools_used"] == ["alpha"]
    assert [t["node"] for t in state["node_trace"]] == ["t1", "t2"]
    assert state["node_trace"][1]["tools_called"] == ["alpha"]
    assert state["model_calls"] == [{"model": "m"}]
    assert state["rag_matches"] == []
    assert state["final_answer"] == "An answer."


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
        return _thread_state(1, final_answer="")

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
async def test_ran_but_failed_skipped_row_carries_its_tool_transcript(tmp_path):
    """A turn that ran tools and came up empty is the interesting failure mode. Its
    skipped row must carry the same formatted delta transcript the judge path uses —
    nothing else stores it. Exception-path rows have no state and stay identity-only."""
    from src.eval.multiturn import run_thread

    scoring, db, _ = _scoring(tmp_path)

    async def agent(**kwargs):
        if kwargs["question_id"].endswith("_t2"):
            raise RuntimeError("boom")
        # t1 and t3 ran a tool but produced no answer.
        return _thread_state(1, final_answer="")

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=agent)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    rows = {r.question_id: r for r in db.get_scores_for_run(scoring.run_id)}
    assert all(r.source == "skipped" for r in rows.values())

    ran = rows["mt-f-01_t1"].context
    assert ran["thread_id"] == "mt-f-01"
    assert ran["turn_index"] == 1
    # Formatted the same way the judge sees it: the tool name and its payload.
    assert "tool_turn1" in ran["tool_results"]
    assert "payload-turn1" in ran["tool_results"]
    assert "trace_turn1" in ran["node_trace"]

    # Exception path: no state existed, so there is nothing to format.
    crashed = rows["mt-f-01_t2"].context
    assert crashed["thread_id"] == "mt-f-01"
    assert crashed["turn_index"] == 2
    assert crashed["tool_results"] is None
    assert crashed["node_trace"] is None


@pytest.mark.asyncio
async def test_no_tool_turn_midthread_reports_zero_tools(tmp_path):
    """TurnResult.tools_used is the per-turn delta, not state["tools_used"]. That
    field is a full-thread rescan under checkpointing, so a turn that called nothing
    would otherwise inherit every earlier turn's tools and print a nonzero count."""
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)

    # Turn 2 adds no tool_results of its own, but the cumulative rescan still names
    # turn 1's tool — and turn 2's messages carry no ToolMessage past the boundary.
    def _state(turn, tool_turns):
        messages = []
        for n in range(1, turn + 1):
            messages += [_Human(f"msg-turn{n}-human"), _Other(f"msg-turn{n}-final")]
            if n in tool_turns:
                messages.insert(-1, _Other(_tool_msg_id(n)))
        return {
            **STATE,
            "messages": messages,
            "tool_results": [_tool_result(n, _tool_msg_id(n)) for n in tool_turns],
            "tools_used": [f"tool_turn{n}" for n in tool_turns],
            "node_trace": [{"node": f"trace_turn{n}"} for n in range(1, turn + 1)],
        }

    states = [_state(1, [1]), _state(2, [1]), _state(3, [1, 3])]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        result = await run_thread(
            THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring
        )

    assert [t.tools_used for t in result.turns] == [["tool_turn1"], [], ["tool_turn3"]]
    # print_summary prints len(tools_used); the middle turn must read 0.
    assert [len(t.tools_used) for t in result.turns] == [1, 0, 1]


@pytest.mark.asyncio
async def test_unscored_run_skips_the_deepcopy_projection(tmp_path):
    """The full projection deep-copies every tool payload and exists only for the
    judge context and the turn report, both of which live under scoring. An unscored
    run must not pay for it — but must still report per-turn tools."""
    from src.eval import multiturn
    from src.eval.multiturn import run_thread

    states = [_thread_state(turn) for turn in (1, 2, 3)]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch.object(
            multiturn, "_turn_delta_state", wraps=multiturn._turn_delta_state
        ) as projection,
    ):
        result = await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=None)

    projection.assert_not_called()
    # The cheap per-turn names path still ran.
    assert [t.tools_used for t in result.turns] == [
        ["tool_turn1"],
        ["tool_turn2"],
        ["tool_turn3"],
    ]


@pytest.mark.asyncio
async def test_scored_run_still_builds_the_projection(tmp_path):
    """The gate is on scoring, not on removal — the scored path still projects."""
    from src.eval import multiturn
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    states = [_thread_state(turn) for turn in (1, 2, 3)]

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=states)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch.object(
            multiturn, "_turn_delta_state", wraps=multiturn._turn_delta_state
        ) as projection,
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    assert projection.call_count == 3


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
                "required_facts": [
                    {"fact_id": "mt-draft-t1-x", "fact_text": "Answer names X (AUTHOR: TBD)"}
                ],
            },
            {
                "turn_id": "t2",
                "question": "Q2?",
                "required_facts": [{"fact_id": "mt-draft-t2-y", "fact_text": "Answer names Y."}],
            },
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
                "required_facts": [{"fact_id": "mt-notes-t1-x", "fact_text": "Answer names X."}],
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
            {
                "turn_id": "t1",
                "question": "Q1?",
                "required_facts": [{"fact_id": "mt-reuse-t1-a", "fact_text": "Fact A."}],
            },
            {
                "turn_id": "t2",
                "question": "Q2?",
                "required_facts": [{"fact_id": "mt-reuse-t2-b", "fact_text": "Fact B."}],
            },
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
    assert judge.score.call_args_list[0].kwargs["required_facts"] == [
        {"fact_id": "mt-reuse-t1-a", "fact_text": "Fact A."}
    ]
    assert judge.score.call_args_list[1].kwargs["required_facts"] == [
        {"fact_id": "mt-reuse-t2-b", "fact_text": "Fact B."}
    ]


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

    # The composite row is annotated, not delta'd: multiturn's macro Fair-only mean
    # and single-turn's micro average are not the same statistic.
    composite_line = next(line for line in out.splitlines() if "COMPOSITE" in line)
    assert "n/a" in composite_line
    assert "macro, Fair-only" in out


def test_compare_between_two_single_turn_runs_keeps_composite_delta(tmp_path, capsys):
    """Same semantics on both sides — the composite delta is still real and shown."""
    from src.eval.__main__ import _handle_compare

    db_url = f"sqlite:///{tmp_path}/eval.db"
    db = EvalDB(db_url)
    dims = {
        "correctness": 1.0,
        "specificity": 1.0,
        "relevance": 1.0,
        "citation_quality": 1.0,
        "hedging": 1.0,
    }
    a = db.create_run(run_type="pre_production", metadata_={"system": "agent_full"})
    db.update_run_summary(str(a.id), dims, 0.50)
    b = db.create_run(run_type="pre_production", metadata_={"system": "agent_full"})
    db.update_run_summary(str(b.id), {**dims, "correctness": 2.0}, 0.75)

    args = MagicMock(run_a=str(a.id), run_b=str(b.id))
    with patch("src.config.settings.DATABASE_URL", db_url):
        _handle_compare(args)
    out = capsys.readouterr().out

    composite_line = next(line for line in out.splitlines() if "COMPOSITE" in line)
    assert composite_line.split()[1:] == ["0.50", "0.75", "+", "0.25"]
    assert "macro, Fair-only" not in out


def test_compare_between_two_multiturn_runs_keeps_composite_delta(tmp_path, capsys):
    """Two multiturn runs share macro semantics, so their composites DO compare —
    this is the incumbent-vs-nothink A/B the battery exists for."""
    from src.eval.__main__ import _handle_compare

    db_url = f"sqlite:///{tmp_path}/eval.db"
    db = EvalDB(db_url)

    def _summary(correctness):
        return {
            "per_dimension": {
                "correctness": correctness,
                "specificity": 1.0,
                "relevance": 1.0,
                "citation_quality": 1.0,
                "hedging": 1.0,
            },
            "thread_composites": {"mt-f-01": 0.8},
            "unscored_threads": [],
            "screened_turns": 0,
            "failed_turns": {},
        }

    a = db.create_run(run_type="pre_production", metadata_={"mode": "multiturn"})
    db.update_run_summary(str(a.id), _summary(1.0), 0.60)
    b = db.create_run(run_type="pre_production", metadata_={"mode": "multiturn"})
    db.update_run_summary(str(b.id), _summary(2.0), 0.70)

    args = MagicMock(run_a=str(a.id), run_b=str(b.id))
    with patch("src.config.settings.DATABASE_URL", db_url):
        _handle_compare(args)
    out = capsys.readouterr().out

    composite_line = next(line for line in out.splitlines() if "COMPOSITE" in line)
    assert composite_line.split()[1:] == ["0.60", "0.70", "+", "0.10"]
    assert "macro, Fair-only" not in out


def test_compare_non_dict_scores_summary_is_not_macro_and_has_no_dimensions(tmp_path, capsys):
    """A run whose scores_summary was never populated (still None, the column
    default) must not blow up `eval compare` — both helpers feature-detect on
    dict shape and fall back cleanly for anything else."""
    from src.eval.__main__ import _handle_compare

    db_url = f"sqlite:///{tmp_path}/eval.db"
    db = EvalDB(db_url)
    never_scored = db.create_run(run_type="pre_production", metadata_={"system": "agent_full"})
    scored = db.create_run(run_type="pre_production", metadata_={"system": "agent_full"})
    db.update_run_summary(
        str(scored.id),
        {
            "correctness": 1.0,
            "specificity": 1.0,
            "relevance": 1.0,
            "citation_quality": 1.0,
            "hedging": 1.0,
        },
        0.5,
    )

    args = MagicMock(run_a=str(never_scored.id), run_b=str(scored.id))
    with patch("src.config.settings.DATABASE_URL", db_url):
        _handle_compare(args)
    out = capsys.readouterr().out

    # Both runs report the same (non-macro) semantics, so no annotation is printed —
    # exercises _is_macro_composite's non-dict False branch on both sides.
    assert "macro, Fair-only" not in out
    # _per_dimension's non-dict branch returns {} rather than raising, so the
    # never-scored side renders as 0.00 for every dimension instead of crashing.
    rows = {
        line.split()[0]: [float(v) for v in line.split()[1:3]]
        for line in out.splitlines()
        if line.strip().startswith(("correctness", "relevance", "hedging"))
    }
    assert rows["correctness"] == [0.0, 1.0]


def test_print_summary_renders_scored_and_unscored_threads(capsys):
    """print_summary's run-summary block: composite line, per-thread composite
    lines (with a failed-turn suffix), the unscored-thread line, screened-turn
    note, failed-turns-in-unscored-threads line, and the run composite."""
    from src.eval.multiturn import ThreadResult, TurnResult, print_summary

    scored_thread = ThreadResult(
        thread_id="mt-scored",
        description="a scored thread",
        turns=[
            TurnResult(
                turn_id="t1",
                question="Q1?",
                answer="A1",
                tools_used=["search_resources"],
                message_count=2,
                duration_ms=123.4,
                success=True,
                composite=0.8,
            ),
            TurnResult(
                turn_id="t2",
                question="Q2?",
                answer="",
                success=False,
                error="boom",
            ),
            TurnResult(
                turn_id="t3",
                question="Q3?",
                answer="",
                success=True,
                answerable=False,
            ),
        ],
    )
    unscored_thread = ThreadResult(
        thread_id="mt-unscored",
        description="",
        turns=[
            TurnResult(
                turn_id="t1",
                question="Q1?",
                answer="",
                success=False,
                error="crashed",
            ),
        ],
    )

    summary = {
        "run_id": "run-123",
        "thread_composites": {"mt-scored": 0.8},
        "unscored_threads": ["mt-unscored"],
        "screened_turns": 1,
        "failed_turns": {"mt-scored": 1, "mt-unscored": 1},
        "composite_score": 0.8,
    }

    print_summary([scored_thread, unscored_thread], summary)
    out = capsys.readouterr().out

    assert "=== Thread mt-scored ===" in out
    assert "a scored thread" in out
    assert "=== Thread mt-unscored ===" in out
    assert "FAIL: boom" in out
    assert "screened (unfair)" in out
    assert "=== Run run-123 ===" in out
    # Scored thread's composite line carries the failed-turn suffix.
    assert "mt-scored: 0.80  (1 failed turn(s))" in out
    # Unscored thread is listed separately, not as a composite line.
    assert "unscored: mt-unscored" in out
    assert "screened (unfair) turns excluded from composites: 1" in out
    # Failed turns belonging to an unscored thread get their own line.
    assert "failed turns in unscored threads: mt-unscored=1" in out
    assert "run composite: 0.80" in out


def test_print_summary_omits_optional_lines_when_nothing_to_report():
    """A clean run (no failures, nothing screened, nothing unscored) must not
    print the failed-turn suffix, the unscored line, or the screened line."""
    from src.eval.multiturn import ThreadResult, TurnResult, print_summary

    thread = ThreadResult(
        thread_id="mt-clean",
        description="",
        turns=[
            TurnResult(turn_id="t1", question="Q?", answer="A", success=True, composite=1.0),
        ],
    )
    summary = {
        "run_id": "run-clean",
        "thread_composites": {"mt-clean": 1.0},
        "unscored_threads": [],
        "screened_turns": 0,
        "failed_turns": {},
        "composite_score": 1.0,
    }

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary([thread], summary)
    out = buf.getvalue()

    assert "mt-clean: 1.00" in out
    assert "failed turn(s)" not in out
    assert "unscored:" not in out
    assert "screened (unfair)" not in out
    assert "failed turns in unscored threads" not in out
    assert "run composite: 1.00" in out


def test_print_summary_without_summary_arg_prints_only_threads(capsys):
    """summary=None (the default) skips the whole run-summary block."""
    from src.eval.multiturn import ThreadResult, TurnResult, print_summary

    thread = ThreadResult(
        thread_id="mt-only",
        description="",
        turns=[TurnResult(turn_id="t1", question="Q?", answer="A", success=True)],
    )
    print_summary([thread])
    out = capsys.readouterr().out

    assert "=== Thread mt-only ===" in out
    assert "=== Run" not in out


def test_last_human_index_dict_message_reads_type_or_role():
    """Dict-shaped messages (fixture form) recognize the human turn boundary via
    either 'type' or 'role', matching the LangChain-object form's `.type`."""
    from src.eval.multiturn import _last_human_index

    messages = [
        {"type": "human", "id": "m1"},
        {"role": "user", "id": "m2"},
        {"type": "ai", "id": "m3"},
    ]
    # The LAST human/user message is the boundary — here index 1 (role=user).
    assert _last_human_index(messages) == 1


def test_build_judge_uses_settings_and_override():
    """_build_judge wires settings.EVAL_JUDGE_* into the Judge, and an explicit
    judge_model argument overrides settings.EVAL_JUDGE_MODEL."""
    from src.eval.multiturn import _build_judge

    judge = _build_judge("explicit-model")
    assert judge.model == "explicit-model"

    judge_default = _build_judge(None)
    from src.config import settings

    assert judge_default.model == settings.EVAL_JUDGE_MODEL


def test_fact_text_bare_string_fact_returned_as_is():
    """`_fact_text` on a plain (non-dict) fact — the YAML-fallback shape — just
    stringifies it directly rather than looking for `fact_text`."""
    from src.eval.multiturn import _fact_text

    assert _fact_text("Answer names X.") == "Answer names X."
