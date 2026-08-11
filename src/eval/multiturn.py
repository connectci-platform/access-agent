"""Multi-turn eval harness.

Runs a thread of questions through the agent in ONE session so accumulated
message history can grow across turns. Designed to exercise context-management
behavior (SummarizationMiddleware) which never fires in single-turn eval.

Battery format (YAML or JSON):

    [
      {
        "thread_id": "compaction-thread-01",
        "description": "Tool-heavy thread to exercise SummarizationMiddleware",
        "questions": [
          {"turn_id": "t1", "question": "..."},
          {"turn_id": "t2", "question": "...", "required_facts": [...]},
          ...
        ],
        "scenario": "Optional scenario context (applies to all turns)",
        "acting_user_required": true
      }
    ]

Optional per-thread fields: ``scenario`` (applies to all turns in the thread)
and ``acting_user_required`` (enforce acting_user parameter). Optional
per-turn fields: ``required_facts`` (YAML list of fact objects or strings).

The harness loads each thread, then issues each turn's question via
``run_agent(use_checkpointing=True, session_id=thread_id)``. The LangGraph
checkpointer loads the prior message list before each turn, so the agent sees
the full conversation context (subject to the middleware's compaction).

Per-turn output (printed + collected) includes message count growth,
duration, tools used, and answer length — enough to eyeball whether
compaction fired and whether answers degraded.

With ``--score``, the harness runs a per-turn judge with access to the prior
turns' visible transcript via ScoringContext. Each successful turn is scored and
persisted to eval_runs/eval_scores.

Judge context is per-turn, not cumulative. Under checkpointing the agent state
grows across the thread (``tool_results`` is rebuilt each turn from the full
ordered message thread; ``node_trace`` appends via its reducer), so the runner
formats only this turn's new entries: ``tool_results`` diffed by tool-call
identity (compaction can SHRINK the rebuilt list, so a count-slice would hide
the current turn's calls) and ``node_trace`` by count (reducer-appended, never
rebuilt). The same delta view feeds the turn report, so a turn's
``invoked_write``/capabilities are never inherited from an earlier turn. Prior
turns reach the judge only through the conversation history — cumulative tool
payloads would present old calls as support for the current answer.

Composites are macro and Fair-only. A thread composite is the mean over its
turns that were judged AND not screened ``answerable=False`` ("Unfair"); the run
composite is the mean of thread composites, so a long thread cannot dominate a
short one. Turns excluded from composites are still counted and surfaced:
``scores_summary`` carries
``{"per_dimension", "thread_composites", "unscored_threads", "screened_turns",
"failed_turns"}``, where ``per_dimension`` holds DIMENSION_NAMES-keyed means over
Fair judged turns (the shape ``eval compare`` consumes) and ``failed_turns`` maps
thread_id to its skipped/judge_error count. A thread with no Fair-judged turn is
reported ``unscored`` and left out of the run composite rather than scoring zero.

Scored runs pre-resolve every turn's required facts before the first agent call
and refuse to start when a resolved fact still contains an ``AUTHOR:``
placeholder (``--allow-draft-facts`` overrides for smoke tests).

The harness writes to ``stdout`` and returns a structured result list and
(when scoring) a summary of run/thread/turn composites.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.agent.graph import run_agent
from src.agent.turn_capture import reset_turn_capture
from src.config import settings
from src.tools import ToolRegistry, get_catalog_aggregator

from .db import EvalDB
from .formatting import format_node_trace, format_rag_matches, format_tool_results
from .judge import Judge
from .question_facts import resolve_required_facts
from .rubric import DIMENSION_NAMES
from .runner import report_battery_turn
from .scoring import persist_skipped_turn, score_and_persist_turn

if TYPE_CHECKING:
    from src.agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_QUESTION_ID_LEN = 64  # eval_scores.question_id is String(64)
FAILED_TURN_MARKER = "(no answer — turn failed)"
DRAFT_FACT_MARKER = "AUTHOR:"


def _tool_call_identity(entry: Any) -> str | None:
    """The tool-call id a rebuilt tool_results entry carries.

    ``_build_tool_results`` (src/agent/nodes/tool_calling_loop.py) anchors every
    entry to its originating AIMessage tool_call and stores that id as
    ``step_id`` — orphan ToolMessages are dropped rather than given a synthetic
    id, so every surviving entry has one. Entries arrive as ToolResult models
    from the agent and as plain dicts from tests/fixtures.
    """
    value = entry.get("step_id") if isinstance(entry, dict) else getattr(entry, "step_id", None)
    return str(value) if value else None


def _turn_delta_state(
    state: Any, seen_tool_call_ids: set[str], prior_node_trace: int
) -> dict[str, Any]:
    """Project the cumulative agent state down to THIS turn's new activity (D1).

    Under checkpointing the state the agent returns is thread-cumulative:
    ``tool_results`` is *rebuilt* each turn from the full ordered message thread
    and ``node_trace`` appends via its ``operator.add`` reducer.

    ``tool_results`` is diffed by tool-call **identity**, not by count.
    SummarizationMiddleware compaction shrinks the rebuilt thread mid-run, so a
    prior turn's count can exceed the post-compaction rebuild and a count-slice
    would drop the current turn's own calls — on exactly the post-compaction
    turn the compaction battery exists to measure. ``node_trace`` is never
    rebuilt from messages (it is reducer-appended and compaction-immune), so
    count-slicing stays valid there.

    An entry with no resolvable identity is treated as new: presenting an
    unattributable call to the judge is the safer failure than hiding this
    turn's evidence.

    The result is a shallow copy of the full state with those two lists
    replaced, so downstream consumers (judge formatting AND the turn report)
    still see ``final_answer``/``tools_used``/``total_tokens`` while seeing only
    this turn's tool activity. ``rag_matches`` is NOT sliced: no node on the
    current path writes it (see src/agent/state.py), so it is always the empty
    per-turn default. Should a node start populating it cumulatively, it needs
    the same treatment.
    """
    delta_tool_results = [
        entry
        for entry in state.get("tool_results", []) or []
        if (identity := _tool_call_identity(entry)) is None or identity not in seen_tool_call_ids
    ]
    return {
        **dict(state),
        "tool_results": delta_tool_results,
        "node_trace": list(state.get("node_trace", []) or [])[prior_node_trace:],
    }


@dataclass
class ScoringContext:
    """Injected by run_battery; never constructed inside run_thread (testability seam)."""

    db: EvalDB
    judge: Judge
    run_id: str
    battery_id: str


@dataclass
class TurnResult:
    """Outcome of a single turn within a multi-turn thread."""

    turn_id: str
    question: str
    answer: str
    tools_used: list[str] = field(default_factory=list)
    message_count: int = 0
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    composite: float | None = None
    score_source: str | None = None
    # Judge screen: False means "Unfair" — excluded from thread/run composites (D2).
    answerable: bool | None = None
    # Per-dimension raw scores from the JudgeResult, for the run's per_dimension means.
    scores: dict[str, int | None] | None = None


@dataclass
class ThreadResult:
    """Outcome of one multi-turn thread."""

    thread_id: str
    description: str
    turns: list[TurnResult] = field(default_factory=list)


def load_thread_battery(path: str) -> list[dict[str, Any]]:
    """Load a multi-turn battery (YAML or JSON by extension) and fail fast on bad shape."""
    file_path = Path(path)
    with file_path.open() as f:
        data: Any = yaml.safe_load(f) if file_path.suffix in (".yaml", ".yml") else json.load(f)
    if not isinstance(data, list):
        # ValueError (not TypeError) to match the other fail-fast validation in this
        # function — all battery-shape errors raise the same exception type.
        raise ValueError("battery must be a top-level list of threads")  # noqa: TRY004
    threads: list[dict[str, Any]] = data

    seen_threads: set[str] = set()
    for thread in threads:
        tid = thread["thread_id"]
        if tid in seen_threads:
            raise ValueError(f"duplicate thread_id: {tid!r}")
        seen_threads.add(tid)
        seen_turns: set[str] = set()
        for q in thread["questions"]:
            turn_id = q["turn_id"]
            if turn_id in seen_turns:
                raise ValueError(f"duplicate turn_id {turn_id!r} in thread {tid!r}")
            seen_turns.add(turn_id)
            question_id = f"{tid}_{turn_id}"
            if len(question_id) > MAX_QUESTION_ID_LEN:
                raise ValueError(f"question_id {question_id!r} exceeds {MAX_QUESTION_ID_LEN} chars")
            if not str(q.get("question", "")).strip():
                raise ValueError(f"empty question in thread {tid!r} turn {turn_id!r}")
    return threads


async def run_thread(
    thread_spec: dict[str, Any],
    tool_catalog: Any,
    *,
    session_namespace: str,
    acting_user: str | None = None,
    resource_context: str | None = None,
    scoring: ScoringContext | None = None,
    resolved_facts: dict[str, list[Any] | None] | None = None,
) -> ThreadResult:
    """Run all questions in a thread sequentially in one checkpointed session.

    Each turn calls ``run_agent(use_checkpointing=True)`` with a stable
    ``session_id`` formed from the session namespace and thread_id. The LangGraph
    checkpointer loads prior messages before each turn so the agent sees the
    full conversation context.

    Args:
        thread_spec: Dict with ``thread_id``, ``description``, and ``questions``.
        tool_catalog: MCP tool catalog (passed to run_agent each turn).
        session_namespace: Namespace for this battery run, ensuring disjoint sessions across runs.
        acting_user: Optional ACCESS ID for personalized tool calls.
        resource_context: Optional RP slug; constant across the thread.
        scoring: When provided, each successful turn is judged and persisted
            (with prior turns as conversation history) and a battery turn
            report is written; failed turns are persisted as skipped.
        resolved_facts: Optional question_id -> required facts map pre-resolved by
            ``run_battery`` (draft-placeholder pre-flight, D3). When a turn's
            question_id is absent from the map its facts are resolved inline.

    Returns:
        ThreadResult with one TurnResult per question.
    """
    thread_id = thread_spec["thread_id"]
    description = thread_spec.get("description", "")
    questions = thread_spec["questions"]

    result = ThreadResult(thread_id=thread_id, description=description)
    session_id = f"eval_{session_namespace}_{thread_id}"
    history: list[tuple[str, str]] = []
    # Delta baseline over the thread-cumulative agent state (D1): tool_results is
    # diffed by tool-call identity (compaction-safe), node_trace by count.
    seen_tool_call_ids: set[str] = set()
    prior_node_trace = 0

    logger.info(f"=== Thread {thread_id}: {description} ===")
    logger.info(f"Turns: {len(questions)}")

    for i, q in enumerate(questions, 1):
        turn_id = q["turn_id"]
        question_text = q["question"]
        question_id = f"{thread_id}_{turn_id}"

        logger.info(f"[turn {i}/{len(questions)}] {turn_id}: {question_text[:80]}")

        start = time.monotonic()
        state: dict[str, Any] | AgentState = {}
        returned_state = False
        # Per-turn capture (summarized flag, tool timings, retrieved chunks) —
        # mirrors the single-turn runner so turn_reports carry this turn's data only.
        reset_turn_capture()
        try:
            state = await run_agent(
                query=question_text,
                session_id=session_id,
                question_id=question_id,
                tool_catalog=tool_catalog,
                acting_user=acting_user,
                resource_context=resource_context,
                use_checkpointing=True,
                db_uri=settings.DATABASE_URL,
            )
            returned_state = True
            duration_ms = (time.monotonic() - start) * 1000
            messages = state.get("messages", [])
            tools_used = state.get("tools_used", [])
            answer = state.get("final_answer", "") or ""
            turn_result = TurnResult(
                turn_id=turn_id,
                question=question_text,
                answer=answer,
                tools_used=tools_used,
                message_count=len(messages),
                duration_ms=duration_ms,
                success=bool(answer),
            )
            logger.info(
                f"  -> answer_len={len(answer)} tools={tools_used} "
                f"messages={len(messages)} duration_ms={int(duration_ms)}"
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(f"  -> error: {e}")
            turn_result = TurnResult(
                turn_id=turn_id,
                question=question_text,
                answer="",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            )

        # This turn's activity only — the same view the judge and the turn report see,
        # so a turn's report can't be credited with an earlier turn's tool calls.
        # On the exception path there is no state, so the delta is empty by construction.
        turn_state = (
            _turn_delta_state(state, seen_tool_call_ids, prior_node_trace) if returned_state else {}
        )

        if scoring is not None:
            await _score_turn(
                scoring,
                turn_result,
                turn_spec=q,
                turn_state=turn_state,
                thread_id=thread_id,
                session_id=session_id,
                question_id=question_id,
                turn_index=i,
                history=history,
                resolved_facts=resolved_facts,
            )

        # Advance the delta baseline whenever run_agent RETURNED state — an
        # empty-answer turn still grew the checkpoint, so its calls must not resurface
        # as the next turn's context. Only the exception path (no state) leaves it put.
        if returned_state:
            for entry in state.get("tool_results", []) or []:
                if (identity := _tool_call_identity(entry)) is not None:
                    seen_tool_call_ids.add(identity)
            prior_node_trace = len(state.get("node_trace", []) or [])

        history.append(
            (question_text, turn_result.answer if turn_result.success else FAILED_TURN_MARKER)
        )

        result.turns.append(turn_result)

    return result


async def _score_turn(
    scoring: ScoringContext,
    turn_result: TurnResult,
    *,
    turn_spec: dict[str, Any],
    turn_state: dict[str, Any],
    thread_id: str,
    session_id: str,
    question_id: str,
    turn_index: int,
    history: list[tuple[str, str]],
    resolved_facts: dict[str, list[Any] | None] | None,
) -> None:
    """Judge + persist one turn, then write its battery turn report.

    ``turn_state`` is the per-turn delta view (D1) — this turn's tool results and
    trace entries over the full state — and feeds BOTH the judge context and the
    turn report, so ``invoked_write`` and capability inference are never
    misattributed from an earlier turn. A turn that ran but produced no answer
    still passes its real delta state; only the exception path is empty.

    Mutates ``turn_result`` with the judge verdict (composite, answerable, raw
    dimension scores, score source). A failed turn is persisted as ``skipped``
    and the thread continues — a mid-thread failure degrading later turns is
    itself multi-turn robustness signal.
    """
    if turn_result.success:
        if resolved_facts is not None and question_id in resolved_facts:
            required_facts = resolved_facts[question_id]
        else:
            required_facts = resolve_required_facts(
                scoring.db, question_id, turn_spec.get("required_facts")
            )
        judge_result = await score_and_persist_turn(
            scoring.db,
            scoring.judge,
            run_id=scoring.run_id,
            question_id=question_id,
            question_text=turn_result.question,
            answer=turn_result.answer,
            rag_context=format_rag_matches(turn_state),
            tool_results=format_tool_results(turn_state),
            node_trace=format_node_trace(turn_state),
            required_facts=required_facts,
            conversation_history=list(history) or None,
            extra_context={
                "thread_id": thread_id,
                "turn_index": turn_index,
                "ground_truth_stability": turn_spec.get("ground_truth_stability"),
            },
            duration_ms=turn_result.duration_ms,
        )
        if judge_result is not None:
            turn_result.composite = judge_result.composite
            turn_result.answerable = judge_result.answerable
            turn_result.scores = judge_result.scores
            turn_result.score_source = "judge"
        else:
            turn_result.score_source = "judge_error"
    else:
        persist_skipped_turn(
            scoring.db,
            run_id=scoring.run_id,
            question_id=question_id,
            question_text=turn_result.question,
            error=turn_result.error,
            duration_ms=turn_result.duration_ms,
            extra_context={"thread_id": thread_id, "turn_index": turn_index},
        )
        turn_result.score_source = "skipped"

    await report_battery_turn(
        state=turn_state,
        session_id=session_id,
        question_id=question_id,
        query_text=turn_result.question,
        duration_ms=turn_result.duration_ms,
        battery_id=scoring.battery_id,
        battery_run_id=scoring.run_id,
        success=turn_result.success,
        turn_index=turn_index,
    )


def _fact_text(fact: Any) -> str:
    """The judge-visible text of a required fact — never authoring metadata.

    Facts arrive either as plain strings (YAML fallback) or as dicts carrying
    ``fact_text`` plus authoring fields. Only the fact text is scanned for the
    draft marker: ``authoring_notes`` legitimately keep their AUTHOR markers.
    """
    if isinstance(fact, dict):
        return str(fact.get("fact_text", ""))
    return str(fact)


def _preresolve_facts(
    db: EvalDB, threads: list[dict[str, Any]], *, allow_draft_facts: bool
) -> dict[str, list[Any] | None]:
    """Resolve every turn's facts up front and refuse unreviewed drafts (D3).

    Runs before the first agent call so an ``AUTHOR:`` placeholder is a true
    pre-flight refusal rather than a mid-run crash, and returns the map so the
    run reuses one resolution per turn (consistency + fewer DB round trips).
    """
    resolved: dict[str, list[Any] | None] = {}
    offending: list[str] = []
    for thread in threads:
        for q in thread["questions"]:
            question_id = f"{thread['thread_id']}_{q['turn_id']}"
            facts = resolve_required_facts(db, question_id, q.get("required_facts"))
            resolved[question_id] = facts
            if facts and any(DRAFT_FACT_MARKER in _fact_text(f) for f in facts):
                offending.append(question_id)

    if offending and not allow_draft_facts:
        raise ValueError(
            f"required facts still contain {DRAFT_FACT_MARKER!r} placeholders for: "
            f"{', '.join(offending)} — confirm the facts or pass --allow-draft-facts"
        )
    if offending:
        logger.warning(
            f"running with {len(offending)} draft-placeholder fact set(s): {', '.join(offending)}"
        )
    return resolved


def _build_judge(judge_model: str | None) -> Judge:
    """Judge construction isolated for test injection."""
    return Judge(
        base_url=settings.EVAL_JUDGE_BASE_URL or None,
        api_key=settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY,
        model=judge_model or settings.EVAL_JUDGE_MODEL,
        thinking=settings.EVAL_JUDGE_THINKING,
    )


async def run_battery(
    battery_path: str,
    acting_user: str | None = None,
    resource_context: str | None = None,
    *,
    score: bool = False,
    judge_model: str | None = None,
    database_url: str | None = None,
    allow_draft_facts: bool = False,
) -> tuple[list[ThreadResult], dict[str, Any] | None]:
    """Load a multi-turn battery (YAML or JSON) file and run every thread in it.

    When ``score`` is set, each turn is judged and persisted under a new
    eval_runs row, and a run-level summary (thread + run composites) is
    returned alongside the per-thread results. Scored runs pre-resolve every
    turn's required facts and refuse to start if any resolved fact text still
    carries an ``AUTHOR:`` placeholder, unless ``allow_draft_facts`` is set.
    """
    threads = load_thread_battery(battery_path)

    if acting_user is None:
        missing_acting_user = [t["thread_id"] for t in threads if t.get("acting_user_required")]
        if missing_acting_user:
            raise ValueError(
                f"acting_user_required threads need --acting-user: {', '.join(missing_acting_user)}"
            )

    # Pre-flight before the catalog fetch and any agent call (and before the
    # eval_runs row exists, so a refusal leaves no orphan run): resolve every
    # turn's facts once and refuse unreviewed draft placeholders (D3).
    db: EvalDB | None = None
    resolved_facts: dict[str, list[Any] | None] | None = None
    if score:
        db = EvalDB(database_url or settings.DATABASE_URL)
        resolved_facts = _preresolve_facts(db, threads, allow_draft_facts=allow_draft_facts)

    aggregator = get_catalog_aggregator()
    catalog = await aggregator.fetch_catalog()
    registry = ToolRegistry(catalog=catalog)
    tool_catalog = registry.catalog

    scoring: ScoringContext | None = None
    if db is not None:
        from src.llm.providers import active_model_name

        from .runner import gen_semantic_run_id, get_git_info

        git_info = get_git_info()
        run = db.create_run(
            id=gen_semantic_run_id("mtloop"),
            run_type="pre_production",
            agent_commit=git_info.get("commit"),
            agent_branch=git_info.get("branch"),
            tool_catalog={
                "tool_count": registry.tool_count,
                "tools": [
                    t.get("name", "unknown") for t in (registry.catalog or {}).get("tools", [])
                ],
            },
            llm_model=active_model_name(),
            judge_model=judge_model or settings.EVAL_JUDGE_MODEL,
            question_set=battery_path,
            question_count=sum(len(t["questions"]) for t in threads),
            metadata_={"system": "agent_full", "mode": "multiturn"},
        )
        scoring = ScoringContext(
            db=db,
            judge=_build_judge(judge_model),
            run_id=str(run.id),
            battery_id=Path(battery_path).stem,
        )
        session_namespace = str(run.id)
    else:
        session_namespace = secrets.token_hex(4)

    results: list[ThreadResult] = []
    for thread_spec in threads:
        results.append(
            await run_thread(
                thread_spec,
                tool_catalog=tool_catalog,
                acting_user=acting_user,
                resource_context=resource_context,
                session_namespace=session_namespace,
                scoring=scoring,
                resolved_facts=resolved_facts,
            )
        )

    summary: dict[str, Any] | None = None
    if scoring is not None:
        scores_summary, run_composite = _build_scores_summary(results)
        scoring.db.update_run_summary(scoring.run_id, scores_summary, run_composite)
        summary = {"run_id": scoring.run_id, "composite_score": run_composite, **scores_summary}

    return results, summary


def _is_fair_scored(turn: TurnResult) -> bool:
    """A turn counts toward composites only when judged AND not screened Unfair (D2)."""
    return turn.composite is not None and turn.answerable is not False


def _build_scores_summary(results: list[ThreadResult]) -> tuple[dict[str, Any], float]:
    """Aggregate thread/run composites and the full scores_summary consumer contract.

    Thread composite = mean over its Fair-judged turns; run composite = mean of
    thread composites (macro, so a long thread cannot dominate). Unfair-screened
    turns and failed/judge_error turns are excluded from composites but counted,
    so a brittle variant cannot look good by scoring only what it survived.
    """
    thread_composites: dict[str, float] = {}
    unscored_threads: list[str] = []
    screened_turns = 0
    failed_turns: dict[str, int] = {}
    fair_scores: list[dict[str, int | None]] = []

    for thread in results:
        fair = [t for t in thread.turns if _is_fair_scored(t)]
        screened_turns += sum(1 for t in thread.turns if t.answerable is False)
        failed = sum(1 for t in thread.turns if t.score_source in ("skipped", "judge_error"))
        if failed:
            failed_turns[thread.thread_id] = failed
        fair_scores.extend(t.scores for t in fair if t.scores is not None)
        if fair:
            thread_composites[thread.thread_id] = sum(
                t.composite for t in fair if t.composite is not None
            ) / len(fair)
        else:
            unscored_threads.append(thread.thread_id)

    run_composite = (
        sum(thread_composites.values()) / len(thread_composites) if thread_composites else 0.0
    )

    # per_dimension is a micro-average over Fair judged turns, keyed by DIMENSION_NAMES —
    # the shape `eval compare` / print_comparison consume for single-turn runs.
    per_dimension: dict[str, float] = {}
    for name in DIMENSION_NAMES:
        values = [v for s in fair_scores if (v := s.get(name)) is not None]
        if values:
            per_dimension[name] = sum(values) / len(values)

    scores_summary: dict[str, Any] = {
        "per_dimension": per_dimension,
        "thread_composites": thread_composites,
        "unscored_threads": unscored_threads,
        "screened_turns": screened_turns,
        "failed_turns": failed_turns,
    }
    return scores_summary, run_composite


def print_summary(results: list[ThreadResult], summary: dict[str, Any] | None = None) -> None:
    """Print a human-friendly summary of thread + per-turn outcomes."""
    for thread in results:
        print(f"\n=== Thread {thread.thread_id} ===")
        if thread.description:
            print(f"  {thread.description}")
        print(
            f"  {'turn':<10} {'tools':<6} {'msgs':<6} {'ans_len':<8} "
            f"{'dur_ms':<8} {'comp':<6} status"
        )
        for turn in thread.turns:
            status = "ok" if turn.success else f"FAIL: {turn.error}"
            if turn.answerable is False:
                status = "screened (unfair)"
            comp = f"{turn.composite:.2f}" if turn.composite is not None else "-"
            print(
                f"  {turn.turn_id:<10} {len(turn.tools_used):<6} "
                f"{turn.message_count:<6} {len(turn.answer):<8} "
                f"{int(turn.duration_ms):<8} {comp:<6} {status}"
            )

    if summary is not None:
        failed_turns: dict[str, int] = summary.get("failed_turns") or {}
        print(f"\n=== Run {summary['run_id']} ===")
        for thread_id, composite in summary["thread_composites"].items():
            failed = failed_turns.get(thread_id)
            # Failed turns print next to every composite: composites exclude them, so a
            # brittle run would otherwise look clean (A/B pairing rule needs them visible).
            suffix = f"  ({failed} failed turn(s))" if failed else ""
            print(f"  {thread_id}: {composite:.2f}{suffix}")
        if summary["unscored_threads"]:
            print(f"  unscored: {', '.join(summary['unscored_threads'])}")
        if summary.get("screened_turns"):
            print(
                f"  screened (unfair) turns excluded from composites: {summary['screened_turns']}"
            )
        unscored_failed = {
            t: n for t, n in failed_turns.items() if t not in summary["thread_composites"]
        }
        if unscored_failed:
            print(
                "  failed turns in unscored threads: "
                + ", ".join(f"{t}={n}" for t, n in unscored_failed.items())
            )
        print(f"  run composite: {summary['composite_score']:.2f}")
