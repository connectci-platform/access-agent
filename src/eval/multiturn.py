"""Multi-turn eval harness.

Runs a thread of questions through the agent in ONE session so accumulated
message history can grow across turns. Designed to exercise context-management
behavior (SummarizationMiddleware) which never fires in single-turn eval.

Battery format (JSON):

    [
      {
        "thread_id": "compaction-thread-01",
        "description": "Tool-heavy thread to exercise SummarizationMiddleware",
        "questions": [
          {"turn_id": "t1", "question": "..."},
          {"turn_id": "t2", "question": "..."},
          ...
        ]
      }
    ]

The harness loads each thread, then issues each turn's question via
``run_agent(use_checkpointing=True, session_id=thread_id)``. The LangGraph
checkpointer loads the prior message list before each turn, so the agent sees
the full conversation context (subject to the middleware's compaction).

Per-turn output (printed + collected) includes message count growth,
duration, tools used, and answer length — enough to eyeball whether
compaction fired and whether answers degraded.

The harness writes to ``stdout`` and returns a structured result list. It
does NOT score answers with a judge — that's a separate concern and can be
layered on top by feeding the collected results into ``judge.score()``.
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
from src.config import settings
from src.tools import ToolRegistry, get_catalog_aggregator

from .db import EvalDB
from .formatting import format_node_trace, format_rag_matches, format_tool_results
from .judge import Judge
from .question_facts import resolve_required_facts
from .runner import report_battery_turn
from .scoring import persist_skipped_turn, score_and_persist_turn

if TYPE_CHECKING:
    from src.agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_QUESTION_ID_LEN = 64  # eval_scores.question_id is String(64)
FAILED_TURN_MARKER = "(no answer — turn failed)"


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

    Returns:
        ThreadResult with one TurnResult per question.
    """
    thread_id = thread_spec["thread_id"]
    description = thread_spec.get("description", "")
    questions = thread_spec["questions"]

    result = ThreadResult(thread_id=thread_id, description=description)
    session_id = f"eval_{session_namespace}_{thread_id}"
    history: list[tuple[str, str]] = []

    logger.info(f"=== Thread {thread_id}: {description} ===")
    logger.info(f"Turns: {len(questions)}")

    for i, q in enumerate(questions, 1):
        turn_id = q["turn_id"]
        question_text = q["question"]
        question_id = f"{thread_id}_{turn_id}"

        logger.info(f"[turn {i}/{len(questions)}] {turn_id}: {question_text[:80]}")

        start = time.monotonic()
        state: dict[str, Any] | AgentState = {}
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

        if scoring is not None:
            if turn_result.success:
                yaml_facts = q.get("required_facts")
                required_facts = resolve_required_facts(scoring.db, question_id, yaml_facts)
                judge_result = await score_and_persist_turn(
                    scoring.db,
                    scoring.judge,
                    run_id=scoring.run_id,
                    question_id=question_id,
                    question_text=question_text,
                    answer=turn_result.answer,
                    rag_context=format_rag_matches(state),
                    tool_results=format_tool_results(state),
                    node_trace=format_node_trace(state),
                    required_facts=required_facts,
                    conversation_history=list(history) or None,
                    extra_context={"thread_id": thread_id, "turn_index": i},
                    duration_ms=turn_result.duration_ms,
                )
                if judge_result is not None:
                    turn_result.composite = judge_result.composite
                    turn_result.score_source = "judge"
                else:
                    turn_result.score_source = "judge_error"
            else:
                persist_skipped_turn(
                    scoring.db,
                    run_id=scoring.run_id,
                    question_id=question_id,
                    question_text=question_text,
                    error=turn_result.error,
                    duration_ms=turn_result.duration_ms,
                    extra_context={"thread_id": thread_id, "turn_index": i},
                )
                turn_result.score_source = "skipped"

            await report_battery_turn(
                state=state if turn_result.success else {},
                session_id=session_id,
                question_id=question_id,
                query_text=question_text,
                duration_ms=turn_result.duration_ms,
                battery_id=scoring.battery_id,
                battery_run_id=scoring.run_id,
                success=turn_result.success,
                turn_index=i,
            )

        history.append(
            (question_text, turn_result.answer if turn_result.success else FAILED_TURN_MARKER)
        )

        result.turns.append(turn_result)

    return result


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
) -> tuple[list[ThreadResult], dict[str, Any] | None]:
    """Load a multi-turn battery JSON file and run every thread in it.

    When ``score`` is set, each turn is judged and persisted under a new
    eval_runs row, and a run-level summary (thread + run composites) is
    returned alongside the per-thread results.
    """
    threads = load_thread_battery(battery_path)

    aggregator = get_catalog_aggregator()
    catalog = await aggregator.fetch_catalog()
    registry = ToolRegistry(catalog=catalog)
    tool_catalog = registry.catalog

    scoring: ScoringContext | None = None
    if score:
        from src.llm.providers import active_model_name

        from .runner import gen_semantic_run_id, get_git_info

        db = EvalDB(database_url or settings.DATABASE_URL)
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
            )
        )

    summary: dict[str, Any] | None = None
    if scoring is not None:
        thread_composites: dict[str, float] = {}
        unscored_threads: list[str] = []
        for thread in results:
            scored = [t.composite for t in thread.turns if t.composite is not None]
            if scored:
                thread_composites[thread.thread_id] = sum(scored) / len(scored)
            else:
                unscored_threads.append(thread.thread_id)
        run_composite = (
            sum(thread_composites.values()) / len(thread_composites) if thread_composites else 0.0
        )
        scores_summary = {
            "thread_composites": thread_composites,
            "unscored_threads": unscored_threads,
        }
        scoring.db.update_run_summary(scoring.run_id, scores_summary, run_composite)
        summary = {
            "run_id": scoring.run_id,
            "composite_score": run_composite,
            "thread_composites": thread_composites,
            "unscored_threads": unscored_threads,
        }

    return results, summary


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
            comp = f"{turn.composite:.2f}" if turn.composite is not None else "-"
            print(
                f"  {turn.turn_id:<10} {len(turn.tools_used):<6} "
                f"{turn.message_count:<6} {len(turn.answer):<8} "
                f"{int(turn.duration_ms):<8} {comp:<6} {status}"
            )

    if summary is not None:
        print(f"\n=== Run {summary['run_id']} ===")
        for thread_id, composite in summary["thread_composites"].items():
            print(f"  {thread_id}: {composite:.2f}")
        if summary["unscored_threads"]:
            print(f"  unscored: {', '.join(summary['unscored_threads'])}")
        print(f"  run composite: {summary['composite_score']:.2f}")
