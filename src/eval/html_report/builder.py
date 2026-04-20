"""Build the HTML report bundle from eval_runs / eval_scores.

This module is pure data: it queries Postgres, aggregates per-battery stats,
and stamps the result into template.html. No LLM calls, no AI involvement.
Given the same DB state and run IDs, output is byte-identical.

High-level flow:

  pick_run_ids()         — resolve which eval_runs to include
      ↓
  fetch_scores()         — pull eval_scores for those runs (one SQL round trip)
      ↓
  assemble_bundle()      — per-question pairs, per-battery averages, run_id list
      ↓
  render_html()          — substitute __BUNDLE_JSON__ in template.html

Edit notes.py for prose (battery descriptions, observations). Edit
template.html for layout. Neither requires touching this file.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, bindparam, create_engine, text
from sqlalchemy.orm import sessionmaker

from ..models import EvalRun, EvalScore
from .notes import BATTERY_INFO, BATTERY_ORDER, OBSERVATIONS, REPORT_SUBTITLE

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "template.html"
BUNDLE_MARKER = "__BUNDLE_JSON__"

SYSTEMS = ("raw_rag", "agent_full")


@dataclass(frozen=True)
class RunRef:
    """Resolved pointer to one eval_runs row."""

    id: str
    system: str  # raw_rag | agent_full
    question_set_key: str  # normalized: "friendly_battery" (no path, no .json)


def _normalize_question_set(raw: str | None) -> str:
    """Turn 'eval/questions/friendly_battery.json' into 'friendly_battery'."""
    if not raw:
        return ""
    base = os.path.basename(raw)
    return base[:-5] if base.endswith(".json") else base


def pick_run_ids(
    database_url: str,
    *,
    on_date: date | None = None,
    question_sets: list[str] | None = None,
) -> list[RunRef]:
    """Select the newest run per (system, question_set).

    If on_date is given, restrict to runs created on that date (UTC). This is
    the usual knob — "give me the report for 2026-04-17." If None, take the
    newest run of each (system, question_set) across all time.

    If question_sets is given, restrict to those keys (normalized names like
    "friendly_battery"). Defaults to BATTERY_ORDER from notes.py.

    Returns one RunRef per (system, question_set) where a run was found.
    Silent about missing combinations — callers decide whether to warn.
    """
    engine = create_engine(database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    Session = sessionmaker(bind=engine)

    wanted_sets = set(question_sets or BATTERY_ORDER)

    with Session() as session:
        q = session.query(
            EvalRun.id,
            EvalRun.metadata_,
            EvalRun.question_set,
            EvalRun.created_at,
        )
        if on_date is not None:
            start = datetime(on_date.year, on_date.month, on_date.day, tzinfo=timezone.utc)
            end = datetime(on_date.year, on_date.month, on_date.day, 23, 59, 59, tzinfo=timezone.utc)
            q = q.filter(and_(EvalRun.created_at >= start, EvalRun.created_at <= end))

        rows = q.all()

    # Group by (system, question_set_key) and keep the newest
    newest: dict[tuple[str, str], tuple[str, datetime]] = {}
    for row in rows:
        md = row.metadata_ or {}
        system = md.get("system") if isinstance(md, dict) else None
        if system not in SYSTEMS:
            continue
        qs_key = _normalize_question_set(row.question_set)
        if qs_key not in wanted_sets:
            continue
        key = (system, qs_key)
        prev = newest.get(key)
        if prev is None or row.created_at > prev[1]:
            newest[key] = (row.id, row.created_at)

    refs: list[RunRef] = []
    for (system, qs_key), (run_id, _) in newest.items():
        refs.append(RunRef(id=run_id, system=system, question_set_key=qs_key))
    return refs


def fetch_scores(database_url: str, refs: list[RunRef]) -> dict[str, list[dict[str, Any]]]:
    """Pull every judge score for the given runs. Returns {run_id: [score_dict]}.

    Includes answer_text and the full context JSON (for node_trace extraction).
    """
    engine = create_engine(database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    Session = sessionmaker(bind=engine)

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_ids = [r.id for r in refs]

    with Session() as session:
        rows = (
            session.query(EvalScore)
            .filter(EvalScore.run_id.in_(run_ids), EvalScore.source == "judge")
            .all()
        )
        for row in rows:
            by_run[row.run_id].append(
                {
                    "question_id": row.question_id,
                    "question_text": row.question_text,
                    "answer_text": row.answer_text,
                    "composite": row.composite_score,
                    "node_trace": _trace_from_context(row.context),
                }
            )
    return dict(by_run)


def _trace_from_context(context: Any) -> list[dict[str, Any]] | None:
    if not context:
        return None
    trace = context.get("node_trace") if isinstance(context, dict) else None
    if isinstance(trace, str):
        try:
            return json.loads(trace)
        except json.JSONDecodeError:
            return None
    if isinstance(trace, list):
        return trace
    return None


def _fetch_durations(database_url: str, run_ids: list[str]) -> dict[tuple[str, str], float]:
    """duration_ms is a column on eval_scores (not yet declared in models.py).

    Fetched via raw SQL so adding duration_ms support doesn't require a model change.
    """
    if not run_ids:
        return {}
    engine = create_engine(database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    stmt = text(
        "SELECT run_id, question_id, duration_ms "
        "FROM eval_scores "
        "WHERE source = 'judge' AND run_id IN :run_ids"
    ).bindparams(bindparam("run_ids", expanding=True))

    out: dict[tuple[str, str], float] = {}
    with engine.connect() as conn:
        for row in conn.execute(stmt, {"run_ids": run_ids}):
            if row.duration_ms is not None:
                out[(row.run_id, row.question_id)] = float(row.duration_ms)
    return out


def assemble_bundle(
    refs: list[RunRef],
    scores_by_run: dict[str, list[dict[str, Any]]],
    durations: dict[tuple[str, str], float],
) -> dict[str, Any]:
    """Fold per-run scores into the shape the template expects."""
    # Index refs by (system, qs_key) for quick lookup
    ref_lookup: dict[tuple[str, str], RunRef] = {
        (r.system, r.question_set_key): r for r in refs
    }

    # Per-question pairs keyed by (question_set_key, question_id)
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for ref in refs:
        for s in scores_by_run.get(ref.id, []):
            key = (ref.question_set_key, s["question_id"])
            entry = pairs.setdefault(
                key,
                {
                    "qid": s["question_id"],
                    "battery": ref.question_set_key.replace("_battery", ""),
                    "question": s["question_text"] or "",
                },
            )
            entry[ref.system] = {
                "composite": s["composite"] or 0.0,
                "duration_ms": durations.get((ref.id, s["question_id"])),
                "answer": s["answer_text"] or "",
                "node_trace": s.get("node_trace"),
            }

    # Keep only pairs that have both systems; compute delta
    all_pairs: list[dict[str, Any]] = []
    for entry in pairs.values():
        if "raw_rag" in entry and "agent_full" in entry:
            entry["delta"] = entry["agent_full"]["composite"] - entry["raw_rag"]["composite"]
            all_pairs.append(entry)
    all_pairs.sort(key=lambda e: (e["battery"], e["qid"]))

    # Per-battery aggregates (counts, wins/losses/ties, averages)
    per_battery: dict[str, dict[str, Any]] = {}
    for qs_key in BATTERY_ORDER:
        short = qs_key.replace("_battery", "")
        items = [e for e in all_pairs if e["battery"] == short]
        if not items:
            continue
        n = len(items)
        wins = sum(1 for e in items if e["delta"] > 0.01)
        losses = sum(1 for e in items if e["delta"] < -0.01)
        ties = n - wins - losses
        info = BATTERY_INFO[qs_key]
        per_battery[short] = {
            "battery": short,
            "label": info["name"],
            "n": n,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "agent_comp": round(sum(e["agent_full"]["composite"] for e in items) / n, 3),
            "raw_comp": round(sum(e["raw_rag"]["composite"] for e in items) / n, 3),
            "agent_dur_ms": _avg_ms(items, "agent_full"),
            "raw_dur_ms": _avg_ms(items, "raw_rag"),
        }

    battery_info_out: dict[str, dict[str, Any]] = {}
    battery_labels: dict[str, str] = {}
    battery_order_out: list[str] = []
    for qs_key in BATTERY_ORDER:
        short = qs_key.replace("_battery", "")
        if short not in per_battery:
            continue
        info = BATTERY_INFO[qs_key]
        battery_info_out[short] = {
            "name": info["name"],
            "count": info["count"],
            "what": info["what"],
            "why": info["why"],
        }
        battery_labels[short] = info["name"]
        battery_order_out.append(short)

    # Run-id strip for the footer
    run_ids_out = []
    for qs_key in BATTERY_ORDER:
        raw_ref = ref_lookup.get(("raw_rag", qs_key))
        agent_ref = ref_lookup.get(("agent_full", qs_key))
        if raw_ref and agent_ref:
            run_ids_out.append(
                {
                    "battery": qs_key.replace("_battery", ""),
                    "raw": raw_ref.id,
                    "agent": agent_ref.id,
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).date().isoformat(),
        "subtitle": REPORT_SUBTITLE,
        "battery_order": battery_order_out,
        "battery_labels": battery_labels,
        "battery_info": battery_info_out,
        "per_battery": per_battery,
        "all_pairs": all_pairs,
        "observations": OBSERVATIONS,
        "run_ids": run_ids_out,
    }


def _avg_ms(items: list[dict[str, Any]], system: str) -> int:
    vals = [e[system]["duration_ms"] for e in items if e[system].get("duration_ms") is not None]
    return round(sum(vals) / len(vals)) if vals else 0


def render_html(bundle: dict[str, Any]) -> str:
    """Stamp the bundle into template.html."""
    template = TEMPLATE_PATH.read_text()
    if BUNDLE_MARKER not in template:
        raise RuntimeError(f"Template is missing {BUNDLE_MARKER} marker")
    # Escape </script to avoid breaking out of the JSON script tag
    bundle_json = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(BUNDLE_MARKER, bundle_json)


def build_report(
    database_url: str,
    output_path: Path,
    *,
    on_date: date | None = None,
    question_sets: list[str] | None = None,
) -> dict[str, Any]:
    """End-to-end: pick runs → fetch → assemble → render → write.

    Returns the bundle (handy for tests / scripting).
    """
    refs = pick_run_ids(database_url, on_date=on_date, question_sets=question_sets)
    if not refs:
        raise RuntimeError("No matching eval_runs found. Check --date and --question-sets.")

    # Warn if any (system, battery) pair is missing
    _warn_missing(refs, question_sets)

    scores = fetch_scores(database_url, refs)
    durations = _fetch_durations(database_url, [r.id for r in refs])
    bundle = assemble_bundle(refs, scores, durations)
    html = render_html(bundle)
    output_path.write_text(html)
    logger.info("Wrote %s (%d bytes, %d question pairs)", output_path, len(html), len(bundle["all_pairs"]))
    return bundle


def _warn_missing(refs: list[RunRef], question_sets: list[str] | None) -> None:
    wanted = question_sets or BATTERY_ORDER
    present: set[tuple[str, str]] = {(r.system, r.question_set_key) for r in refs}
    for qs in wanted:
        for sys in SYSTEMS:
            if (sys, qs) not in present:
                logger.warning("No run found for system=%s question_set=%s", sys, qs)
