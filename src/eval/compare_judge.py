"""Comparison judge: evaluates pairs of system answers side-by-side.

Reads completed run data from Postgres (read-only), pairs answers by
question_id, calls an LLM to produce a comparison verdict per question,
then a run-level summary. Output is a JSON artifact on disk — no DB writes.

Uses the same OpenAI endpoint / model the per-answer judge uses
(EVAL_JUDGE_BASE_URL, EVAL_JUDGE_API_KEY, EVAL_JUDGE_MODEL). The
comparison judge is a distinct first-class stage in the eval pipeline,
parallel to `rejudge` and `argilla-push`; the HTML report generator can
optionally consume its JSON as a narrative spine.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.config import settings

from .db import EvalDB

logger = logging.getLogger(__name__)


MISSION_PREAMBLE = (
    "This agent supports researchers using ACCESS-CI (the US national "
    "cyberinfrastructure allocation system). Users ask about compute "
    "resources, software, allocations, system status, events, and how "
    "to get help. They need specific, current, actionable information — "
    "named resources, software versions, event dates, ticket confirmations, "
    "exact counts. Generic how-to guidance is less valuable than concrete "
    "data when concrete data is achievable, because the user's goal is "
    "usually to DO something, not to read documentation. When the system "
    "takes an action on the user's behalf (creates a support ticket, looks "
    "up live allocation data), that is a meaningfully better outcome than "
    "pointing the user at a URL and asking them to do it themselves."
)


def _format_context(ctx: dict[str, Any] | None) -> str:
    """Compose a readable context block from a stored eval_scores.context dict."""
    if not ctx:
        return "(no context available for this system)"
    parts = []
    rag = ctx.get("rag_context")
    if rag:
        parts.append(f"### RAG documents\n{rag}")
    tools = ctx.get("tool_results")
    if tools:
        parts.append(f"### Tool results\n{tools}")
    trace = ctx.get("node_trace")
    if trace:
        parts.append(f"### Agent decision trace\n{trace}")
    return "\n\n".join(parts) if parts else "(no context captured)"


def _extract_json(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Comparison judge returned invalid JSON")
        return None
    if not isinstance(data, dict):
        return None
    return data


def _build_comparison_prompt(
    *,
    query: str,
    system_a: str,
    system_b: str,
    answer_a: str,
    answer_b: str,
    context_a: dict[str, Any] | None,
    context_b: dict[str, Any] | None,
    score_a: float,
    score_b: float,
) -> str:
    return f"""You are a comparison judge evaluating two AI system answers to the same user question.

## Mission

{MISSION_PREAMBLE}

## The question

{query}

## System A — {system_a}

### Answer
{answer_a}

### Context System A had
{_format_context(context_a)}

### Prior per-answer judge composite score for A
{score_a:.2f} (on a 1.0-5.0 scale)

## System B — {system_b}

### Answer
{answer_b}

### Context System B had
{_format_context(context_b)}

### Prior per-answer judge composite score for B
{score_b:.2f} (on a 1.0-5.0 scale)

## Your task

Compare the two answers head-to-head for a user who needs specific, actionable ACCESS-CI information. Consider:
- Did each answer address what the user actually wanted?
- Was the information specific/current/actionable, or generic?
- Did either answer hallucinate, or misrepresent its sources?
- Given what each system HAD available (see context), did it use that well?

Return ONLY a JSON object with this exact structure (no other text):
```json
{{
  "winner": "A" | "B" | "tie",
  "margin": "large" | "small" | "none",
  "why": "<2-3 sentence explanation of the verdict>",
  "per_answer_judge_note": "<was the per-answer judge's relative ranking right? Short note — agree, disagree, or why.>"
}}
```"""


def _build_summary_prompt(
    *,
    n_questions: int,
    system_a: str,
    system_b: str,
    per_question: list[dict[str, Any]],
) -> str:
    verdicts_condensed = "\n".join(
        f"- {v.get('question_id', '?')}: winner={v.get('winner', '?')}, "
        f"margin={v.get('margin', '?')}. {v.get('why', '')[:200]}"
        for v in per_question
    )
    return f"""You have compared {n_questions} questions across two AI systems.

## Mission

{MISSION_PREAMBLE}

## The two systems

- System A: **{system_a}**
- System B: **{system_b}**

## Per-question verdicts from this comparison

{verdicts_condensed}

## Your task

Produce a run-level summary. Return ONLY a JSON object with this exact structure (no other text):
```json
{{
  "winner": "A" | "B" | "tie",
  "margin": "large" | "small" | "none",
  "why": "<a paragraph explaining the overall verdict>",
  "patterns": ["<where System A tends to win or lose>", "<where System B tends to win or lose>", "..."],
  "per_answer_judge_calibration": "<a paragraph on where the per-answer judge was systematically right or wrong>"
}}
```"""


async def _compare_one(
    *,
    client: AsyncOpenAI,
    model: str,
    query: str,
    system_a: str,
    system_b: str,
    answer_a: str,
    answer_b: str,
    context_a: dict[str, Any] | None,
    context_b: dict[str, Any] | None,
    score_a: float,
    score_b: float,
) -> dict[str, Any] | None:
    prompt = _build_comparison_prompt(
        query=query,
        system_a=system_a,
        system_b=system_b,
        answer_a=answer_a,
        answer_b=answer_b,
        context_a=context_a,
        context_b=context_b,
        score_a=score_a,
        score_b=score_b,
    )
    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
            )
            raw = response.choices[0].message.content or ""
            data = _extract_json(raw)
            if data is not None and data.get("winner") in ("A", "B", "tie"):
                return data
            if attempt == 0:
                logger.warning("Comparison judge parse failed, retrying")
        except Exception as e:
            logger.error(f"Comparison judge call failed (attempt {attempt + 1}): {e}")
    return None


async def _summarize(
    *,
    client: AsyncOpenAI,
    model: str,
    system_a: str,
    system_b: str,
    per_question: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prompt = _build_summary_prompt(
        n_questions=len(per_question),
        system_a=system_a,
        system_b=system_b,
        per_question=per_question,
    )
    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )
            raw = response.choices[0].message.content or ""
            data = _extract_json(raw)
            if data is not None and data.get("winner") in ("A", "B", "tie"):
                return data
            if attempt == 0:
                logger.warning("Summary parse failed, retrying")
        except Exception as e:
            logger.error(f"Summary call failed (attempt {attempt + 1}): {e}")
    return None


def _score_to_public(score: Any, include_context: bool) -> dict[str, Any]:
    """Pull the presentation-relevant fields off an EvalScore row."""
    out: dict[str, Any] = {
        "answer": str(score.answer_text or ""),
        "composite": round(float(score.composite_score or 0), 2),
        "dimensions": {
            "correctness": score.correctness,
            "completeness": score.completeness,
            "relevance": score.relevance,
            "citation_quality": score.citation_quality,
            "hedging": score.hedging,
        },
        "justifications": score.justifications or {},
        "duration_ms": (float(score.duration_ms) if score.duration_ms is not None else None),
    }
    ctx: dict[str, Any] = score.context or {}
    # node_trace is useful for the HTML's execution-path viz; always include it
    trace = ctx.get("node_trace")
    if isinstance(trace, str):
        try:
            out["node_trace"] = json.loads(trace)
        except json.JSONDecodeError:
            out["node_trace"] = None
    elif isinstance(trace, list):
        out["node_trace"] = trace
    else:
        out["node_trace"] = None

    # Required-facts grading (when the battery has authored required_facts and
    # the judge produced per-fact verdicts). Always included — it's part of the
    # presentation surface and small enough not to need gating on include_context.
    if ctx.get("fact_verdicts"):
        out["fact_verdicts"] = ctx["fact_verdicts"]
    if ctx.get("required_facts"):
        out["required_facts"] = ctx["required_facts"]
    if ctx.get("ground_truth_stability"):
        out["ground_truth_stability"] = ctx["ground_truth_stability"]

    if include_context:
        out["rag_context"] = ctx.get("rag_context")
        out["tool_results"] = ctx.get("tool_results")
    return out


async def compare_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    output_path: str | None = None,
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
    include_context: bool = True,
) -> dict[str, Any]:
    """Compare a baseline run against a candidate run, one question at a time.

    Writes a self-contained JSON artifact that carries everything needed to
    render a report for this pair — both answers, per-dim scores, justifications,
    durations, node traces, and (optionally) the full RAG/tool-result context.
    The artifact is an immutable snapshot: same JSON → same rendered report,
    regardless of later DB state.

    Args:
        include_context: if True (default), embed rag_context and tool_results
            for each system in each question. Adds ~1-5 MB per battery-pair but
            makes the artifact fully self-contained for rendering.
    """
    db_url = database_url or settings.DATABASE_URL
    j_base = judge_base_url or settings.EVAL_JUDGE_BASE_URL or None
    j_key = judge_api_key or settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY
    j_model = judge_model or settings.EVAL_JUDGE_MODEL

    db = EvalDB(db_url)
    baseline = db.get_run(baseline_run_id)
    candidate = db.get_run(candidate_run_id)
    if baseline is None:
        raise ValueError(f"Baseline run {baseline_run_id} not found")
    if candidate is None:
        raise ValueError(f"Candidate run {candidate_run_id} not found")

    baseline_meta: dict[str, Any] = baseline.metadata_ or {}  # type: ignore[assignment]
    candidate_meta: dict[str, Any] = candidate.metadata_ or {}  # type: ignore[assignment]
    system_a = str(baseline_meta.get("system", "A"))
    system_b = str(candidate_meta.get("system", "B"))

    baseline_scores = {s.question_id: s for s in db.get_scores_for_run(baseline_run_id)}
    candidate_scores = {s.question_id: s for s in db.get_scores_for_run(candidate_run_id)}

    common_qids = sorted(set(baseline_scores) & set(candidate_scores))
    if not common_qids:
        raise ValueError("No common question_ids between the two runs")

    client = AsyncOpenAI(base_url=j_base, api_key=j_key or "not-needed")

    per_question: list[dict[str, Any]] = []
    for i, qid in enumerate(common_qids, 1):
        base_s = baseline_scores[qid]
        cand_s = candidate_scores[qid]
        if base_s.source != "judge" or cand_s.source != "judge":
            logger.info(f"[{i}/{len(common_qids)}] {qid}: skipped (non-judge source)")
            continue

        logger.info(f"[{i}/{len(common_qids)}] comparing {qid}")
        verdict = await _compare_one(
            client=client,
            model=j_model,
            query=str(base_s.question_text or cand_s.question_text or ""),
            system_a=system_a,
            system_b=system_b,
            answer_a=str(base_s.answer_text or ""),
            answer_b=str(cand_s.answer_text or ""),
            context_a=base_s.context or {},  # type: ignore[arg-type]
            context_b=cand_s.context or {},  # type: ignore[arg-type]
            score_a=float(base_s.composite_score or 0),
            score_b=float(cand_s.composite_score or 0),
        )
        if verdict is None:
            logger.warning(f"{qid}: compare-judge returned no verdict")
            continue

        verdict["question_id"] = qid
        verdict["question_text"] = str(base_s.question_text or "")
        verdict["baseline"] = _score_to_public(base_s, include_context)
        verdict["candidate"] = _score_to_public(cand_s, include_context)
        # Convenience top-level composites so consumers don't have to dig
        verdict["baseline_composite"] = verdict["baseline"]["composite"]
        verdict["candidate_composite"] = verdict["candidate"]["composite"]
        per_question.append(verdict)

    logger.info("generating run-level summary")
    run_summary = await _summarize(
        client=client,
        model=j_model,
        system_a=system_a,
        system_b=system_b,
        per_question=per_question,
    )

    artifact = {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "baseline_system": system_a,
        "candidate_system": system_b,
        "question_set": candidate.question_set or baseline.question_set,
        "judge_model": j_model,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_composite": round(float(baseline.composite_score or 0), 2),
        "candidate_composite": round(float(candidate.composite_score or 0), 2),
        "questions_compared": len(per_question),
        "run_summary": run_summary,
        "per_question": per_question,
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(artifact, indent=2, default=str))
        logger.info(f"Wrote comparison artifact to {output_path}")

    return artifact
