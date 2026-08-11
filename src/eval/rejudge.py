"""Re-judge an existing run with the current rubric and judge prompt.

Replays frozen question/answer/context rows through the current judge, writing
a new eval_run linked back to the original via metadata.rejudged_from. Does
not re-call the system — the answer_text and context blobs come from Postgres.

A rejudged run's summary is rebuilt under the SOURCE run's semantics: a
``mode="multiturn"`` run groups its rows by ``context.thread_id`` and rebuilds
Fair-only thread composites plus a macro run composite (the full
``scores_summary`` contract), so an original-vs-rejudge delta compares like
statistics. Single-turn runs keep the flat micro-averaged dimension map. Only
rows that represent a turn feed that rebuild (see ``TURN_SOURCES``): human
review rows and rows without a ``thread_id`` are left out entirely rather than
pooled into a synthetic thread.

A rejudged row's context is the ORIGINAL row's replayed transcript with one
exception — ``fact_verdicts`` is a judge OUTPUT, so it is replaced with the new
judge's verdicts. ``skipped`` rows are not judgeable but are copied forward into
the rejudged run, so a thread's ``failed_turns`` survives across successive
rejudge generations instead of decaying to a clean-looking zero.

Limitation: the stored `tool_results` field is pre-formatted text produced by
the formatter that was current at original-run time. Rubric prompt changes
(e.g. mission preamble) fully apply; tool-context-format changes only apply
to questions re-run from scratch, since the raw tool call objects aren't
preserved in storage.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings

from . import scoring
from .db import EvalDB
from .judge import Judge
from .multiturn import TurnRecord, build_multiturn_summary
from .rubric import DIMENSION_NAMES, compute_composite
from .runner import get_git_info

logger = logging.getLogger(__name__)


def _replay_history(stored: Any) -> list[tuple[str, str]] | None:
    """Convert a persisted conversation_history (list-of-lists) back to Q/A tuples."""
    if not stored:
        return None
    history: list[tuple[str, str]] = []
    for pair in stored:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            history.append((str(pair[0]), str(pair[1])))
    return history or None


# Row sources that represent a multiturn TURN. Human review rows (and any other
# non-turn source) describe the same question a judge row already covers, so
# ingesting them would double-count a turn or invent a thread that never ran.
TURN_SOURCES = ("judge", "judge_error", "skipped")


def _turn_record(
    context: dict[str, Any],
    *,
    source: str,
    composite: float | None = None,
    answerable: bool | None = None,
    scores: dict[str, int | None] | None = None,
) -> TurnRecord | None:
    """Adapt one persisted score row into the multiturn summary's turn shape.

    Returns None for rows that are not multiturn turns: a source outside
    ``TURN_SOURCES`` (human review rows), or a row with no
    ``context["thread_id"]``. Both would otherwise land in a synthetic
    "(no thread)" bucket and be reported as a real unscored thread — a human
    row on one question would make the run look like it had a thread that never
    produced a Fair turn.
    """
    if source not in TURN_SOURCES:
        return None
    thread_id = context.get("thread_id")
    if not thread_id:
        return None
    return TurnRecord(
        thread_id=str(thread_id),
        composite=composite,
        answerable=answerable,
        scores=scores,
        source=source,
    )


def _add_turn_record(records: list[TurnRecord], record: TurnRecord | None) -> None:
    """Append a turn record when the row qualifies as one (see ``_turn_record``)."""
    if record is not None:
        records.append(record)


def _rebuild_summary(
    all_scores: list[dict[str, int | None]],
    turn_records: list[TurnRecord],
    *,
    is_multiturn: bool,
) -> tuple[dict[str, Any], dict[str, float], float]:
    """Build the rejudged run's (scores_summary, per-dimension means, composite).

    A multiturn source run rebuilds under multiturn semantics — macro mean over
    Fair-only thread composites, full scores_summary contract — so the
    original-vs-rejudge delta compares like statistics. Scoring it as an all-rows
    micro mean against the stored macro would report phantom judge drift on
    identical verdicts. Single-turn runs keep the flat micro-averaged map.
    """
    if is_multiturn:
        summary, composite = build_multiturn_summary(turn_records)
        return summary, dict(summary["per_dimension"]), composite
    if not all_scores:
        return {}, {}, 0.0
    avg_scores = {
        name: (
            sum(v for s in all_scores if (v := s.get(name)) is not None)
            / max(1, sum(1 for s in all_scores if s.get(name) is not None))
        )
        for name in DIMENSION_NAMES
    }
    avg_composite = sum(compute_composite(s) for s in all_scores) / len(all_scores)
    return avg_scores, avg_scores, avg_composite


async def rejudge_run(
    original_run_id: str,
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    db_url = database_url or settings.DATABASE_URL
    j_base = judge_base_url or settings.EVAL_JUDGE_BASE_URL or None
    j_key = judge_api_key or settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY
    j_model = judge_model or settings.EVAL_JUDGE_MODEL

    db = EvalDB(db_url)
    original = db.get_run(original_run_id)
    if original is None:
        raise ValueError(f"Run {original_run_id} not found")

    judge = Judge(
        base_url=j_base, api_key=j_key, model=j_model, thinking=settings.EVAL_JUDGE_THINKING
    )
    git_info = get_git_info()
    original_meta: dict[str, Any] = original.metadata_ or {}  # type: ignore[assignment]

    # A rejudged multiturn run must stay excluded from the dashboard aggregate, so
    # mode rides along with the run rather than being reconstructed downstream.
    rejudge_meta: dict[str, Any] = {
        "system": original_meta.get("system"),
        "rejudged_from": original_run_id,
        "rejudge_commit": git_info.get("commit"),
        "rejudge_branch": git_info.get("branch"),
    }
    if original_meta.get("mode") is not None:
        rejudge_meta["mode"] = original_meta["mode"]

    new_run = db.create_run(
        run_type="rejudge",
        agent_commit=original.agent_commit,
        agent_branch=original.agent_branch,
        tool_catalog=original.tool_catalog,
        llm_model=original.llm_model,
        judge_model=j_model,
        question_set=original.question_set,
        question_count=original.question_count,
        metadata_=rejudge_meta,
    )
    logger.info(
        f"Rejudge run {new_run.id} created (source run {original_run_id}, "
        f"system={original_meta.get('system')}, judge={j_model})"
    )

    is_multiturn = original_meta.get("mode") == "multiturn"

    original_scores = db.get_scores_for_run(original_run_id)
    all_scores: list[dict[str, int | None]] = []
    # Per-turn records for the multiturn summary rebuild (thread-grouped, Fair-only).
    turn_records: list[TurnRecord] = []
    rescored = 0
    skipped = 0
    errors = 0

    for i, score in enumerate(original_scores, 1):
        context: dict[str, Any] = score.context or {}  # type: ignore[assignment]

        if score.source != "judge":
            skipped += 1
            # Not rejudged, but a multiturn summary must still count it against its
            # thread's failed_turns — composites exclude failed turns, so a run that
            # crashed half a thread would otherwise look clean.
            _add_turn_record(turn_records, _turn_record(context, source=str(score.source)))
            if score.source == "skipped":
                # Materialize the failure into the new run. Without this row the
                # rejudged run has no record that the turn failed, so rejudging the
                # rejudge (or any later read of the new run's rows) reports a clean
                # thread — failed_turns would decay to zero across generations.
                db.add_score(
                    run_id=new_run.id,
                    question_id=score.question_id,
                    source="skipped",
                    question_text=score.question_text,
                    answer_text=score.answer_text,
                    context=context,
                    duration_ms=score.duration_ms,
                    justifications=score.justifications,
                )
            continue

        logger.info(f"[{i}/{len(original_scores)}] rejudging {score.question_id}")

        judge_result = await judge.score(
            query=str(score.question_text or ""),
            answer=str(score.answer_text or ""),
            rag_context=context.get("rag_context"),
            tool_results=context.get("tool_results"),
            node_trace=context.get("node_trace"),
            # The scorer freezes required_facts in context; replay them so a
            # rejudged run keeps fact-grounded verdicts (and the fact-sized
            # token budget — omitting them starved verbose judges at 500 tokens).
            required_facts=context.get("required_facts"),
            # Multiturn rows persist the prior transcript as list-of-lists (JSON has
            # no tuples). Replaying it keeps reference-resolution turns judgeable —
            # without it the judge sees "how does that compare?" with no antecedent
            # and records a phantom quality drop.
            conversation_history=_replay_history(context.get("conversation_history")),
        )

        if judge_result is None:
            errors += 1
            _add_turn_record(turn_records, _turn_record(context, source="judge_error"))
            db.add_score(
                run_id=new_run.id,
                question_id=score.question_id,
                source="judge_error",
                question_text=score.question_text,
                answer_text=score.answer_text,
                context=context,
                context_completeness=score.context_completeness,
                duration_ms=score.duration_ms,
                justifications={"error": "Rejudge failed to produce valid scores"},
            )
            continue

        # The replayed context (transcript, facts, history) is the ORIGINAL run's —
        # it is what was re-judged. But fact_verdicts are a JUDGE OUTPUT, so the new
        # row must carry the new judge's verdicts; keeping the source row's would
        # make a rejudge report the old judge's per-fact calls beside new scores.
        rejudged_context = {**context, "fact_verdicts": judge_result.fact_verdicts}

        db.add_score(
            run_id=new_run.id,
            question_id=score.question_id,
            source="judge",
            question_text=score.question_text,
            answer_text=score.answer_text,
            context=rejudged_context,
            context_completeness=score.context_completeness,
            correctness=judge_result.scores["correctness"],
            specificity=judge_result.scores["specificity"],
            specificity_na=judge_result.specificity_na,
            answerable=judge_result.answerable,
            rubric_version=scoring.RUBRIC_VERSION,
            relevance=judge_result.scores["relevance"],
            citation_quality=judge_result.scores["citation_quality"],
            hedging=judge_result.scores["hedging"],
            composite_score=judge_result.composite,
            duration_ms=score.duration_ms,
            justifications=judge_result.justifications,
        )
        all_scores.append(judge_result.scores)
        _add_turn_record(
            turn_records,
            _turn_record(
                context,
                source="judge",
                composite=judge_result.composite,
                answerable=judge_result.answerable,
                scores=judge_result.scores,
            ),
        )
        rescored += 1

    new_summary, avg_scores, avg_composite = _rebuild_summary(
        all_scores, turn_records, is_multiturn=is_multiturn
    )
    db.update_run_summary(str(new_run.id), new_summary, avg_composite)

    summary = {
        "new_run_id": new_run.id,
        "original_run_id": original_run_id,
        "system": original_meta.get("system"),
        "question_set": original.question_set,
        "rescored": rescored,
        "skipped": skipped,
        "errors": errors,
        "original_composite": round(original.composite_score or 0.0, 2),
        "new_composite": round(avg_composite, 2),
        "delta": round(avg_composite - (original.composite_score or 0.0), 2),
        "per_dimension": {k: round(v, 2) for k, v in avg_scores.items()},
        "judge_model": j_model,
        "rejudge_commit": git_info.get("commit", "")[:8],
    }

    logger.info(
        f"Rejudge run {new_run.id} complete: {rescored} rescored, delta={summary['delta']:+.2f}"
    )
    return summary
