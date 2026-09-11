#!/usr/bin/env python3
"""Sync authored multi-turn facts into reporting.question_facts as draft rows.

Reads the multiturn battery files, where every fact carries an explicit
``fact_id``, and upserts by ``(question_id, fact_id)``. Identity is the id, never
the position: deleting, reordering, or inserting facts in the YAML is meaningless
to identity, so there are no positional guards here — no shrink counts, no
delete-shift detection, no bulk-edit flag. Duplicate fact_ids within a turn are
refused when the battery loads (``load_thread_battery``).

Per fact, the DB's latest version decides the branch:

  * absent → insert version 1 (draft),
  * latest is RETRACTED → SKIP with a warning, whether or not the text differs.
    Retraction is a reviewer's decision that the fact is not a requirement, and
    INSERT_VERSION does not carry the stamp forward, so versioning up would clear
    it and silently return the fact to scoring,
  * latest is a DRAFT and the text CHANGED → insert version latest+1 — the
    resolver takes the latest non-flagged version, so this is how an AUTHOR-pass
    edit actually reaches the judge,
  * latest is NOT a draft (confirmed or flagged) and the text differs → SKIP with
    a stale-YAML warning: the DB is authoritative after review and the script
    never overwrites reviewed content,
  * text unchanged → skip silently.

A fact_id present in the DB but absent from the YAML is not touched. Retirement
is flagging or retracting the fact in the dashboard; the resolver excludes facts
whose latest version is flagged or retracted.

Exit codes: 0 success, 2 completed with stale-YAML or retracted skips. Both mean
authored content did not land — a stale skip DROPPED the YAML edit, a retracted
skip means the battery still names a withdrawn fact — and neither must be a
silent success, or automation reports green while a correction never landed or a
retraction is pending resolution. The exit happens after all other work completes.

Run over the 5433 tunnel:

    scripts/eval-tunnel-open   # in another terminal
    DATABASE_URL=... uv run python scripts/insert-multiturn-facts.py [--dry-run]

--dry-run stays DB-free: it prints the rows the script would consider and makes
no connection.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Row

REPO_ROOT = Path(__file__).resolve().parent.parent


class FactRow(TypedDict):
    """One authored fact, in the shape INSERT_VERSION binds."""

    question_id: str
    fact_id: str
    fact_text: str
    display_order: int


BATTERIES = [
    str(REPO_ROOT / "eval/questions/multiturn_support_battery.yaml"),
    str(REPO_ROOT / "eval/questions/multiturn_compaction_battery.json"),
]

# The DB's current state for one (question_id, fact_id): highest version, its text,
# and its status — status decides whether an edit may version up or must be refused
# as stale YAML, so it is part of the branch, not decoration.
LATEST = text("""
    SELECT version, fact_text, status, retracted_at
    FROM reporting.question_facts
    WHERE question_id = :question_id AND fact_id = :fact_id
    ORDER BY version DESC
    LIMIT 1
""")

INSERT_VERSION = text("""
    INSERT INTO reporting.question_facts
        (question_id, fact_id, fact_text, status, version, display_order)
    VALUES (:question_id, :fact_id, :fact_text, 'draft', :version, :display_order)
""")

STALE_REMEDY = (
    "Not overwriting. Two paths that work: update the YAML to the reviewed text, "
    "or author the correction as a new draft version in the dashboard."
)

RETRACTED_REMEDY = (
    "Not reviving. A retraction is a reviewer's decision that this is not a "
    "requirement for the question; bumping a new version would clear it and put the "
    "fact back into scoring. Remove the fact from the battery file, or un-retract it "
    "in the dashboard first."
)


def build_rows(batteries: list[str]) -> list[FactRow]:
    """Flatten the batteries into insert-shaped rows keyed by their authored fact_id.

    ``display_order`` is the fact's position in its turn — display metadata for the
    dashboard, not identity. ``load_thread_battery`` has already refused bare-string
    facts and duplicate fact_ids within a turn, so every fact here has a usable id.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.eval.multiturn import load_thread_battery

    rows: list[FactRow] = []
    for battery in batteries:
        for thread in load_thread_battery(battery):
            for q in thread["questions"]:
                question_id = f"{thread['thread_id']}_{q['turn_id']}"
                for position, fact in enumerate(q.get("required_facts") or [], 1):
                    rows.append(
                        {
                            "question_id": question_id,
                            "fact_id": str(fact["fact_id"]),
                            "fact_text": str(fact["fact_text"]),
                            "display_order": position,
                        }
                    )
    return rows


def _latest(conn: Connection, question_id: str, fact_id: str) -> "Row[Any] | None":
    return conn.execute(LATEST, {"question_id": question_id, "fact_id": fact_id}).fetchone()


def apply_rows(conn: Connection, rows: list[FactRow]) -> dict[str, int]:
    """Upsert each row by (question_id, fact_id), returning the per-outcome counts.

    One ``_latest`` fetch per fact: the branch and the next version number both
    come from that single read.
    """
    counts = {
        "inserted": 0,
        "bumped": 0,
        "skipped_same": 0,
        "skipped_stale": 0,
        "skipped_retracted": 0,
    }
    for r in rows:
        latest = _latest(conn, r["question_id"], r["fact_id"])
        if latest is None:
            version = 1
            counts["inserted"] += 1
        elif latest.retracted_at is not None:
            # Checked before the text-equality skip: a retracted fact must be
            # reported even when the YAML still matches, because the battery file
            # naming it at all is what needs resolving. Drives a non-zero exit.
            counts["skipped_retracted"] += 1
            print(
                f"WARNING retracted fact in battery: {r['question_id']}/{r['fact_id']} "
                f"was retracted in the dashboard at {latest.retracted_at}. "
                f"{RETRACTED_REMEDY}",
                file=sys.stderr,
            )
            continue
        elif latest.fact_text == r["fact_text"]:
            counts["skipped_same"] += 1
            continue
        elif latest.status != "draft":
            # Reviewed content is authoritative; the YAML is the stale copy. The
            # author's edit is DROPPED here, so this also drives a non-zero exit.
            counts["skipped_stale"] += 1
            print(
                f"WARNING stale YAML: {r['question_id']}/{r['fact_id']} "
                f"v{latest.version} is {latest.status} in the DB and differs from "
                f"the battery file. {STALE_REMEDY}",
                file=sys.stderr,
            )
            continue
        else:
            version = latest.version + 1
            counts["bumped"] += 1
        conn.execute(INSERT_VERSION, {**r, "version": version})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = build_rows(BATTERIES)

    print(f"{len(rows)} fact rows from {len(BATTERIES)} batteries")
    if args.dry_run:
        # DB-free by contract, so the per-fact branch (which needs the DB) doesn't run.
        for r in rows[:5]:
            print(r)
        return 0

    url = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url)
    with engine.begin() as conn:
        counts = apply_rows(conn, rows)

    print(
        f"inserted {counts['inserted']} new, bumped {counts['bumped']} edited, "
        f"skipped {counts['skipped_same']} unchanged, skipped {counts['skipped_stale']} stale, "
        f"skipped {counts['skipped_retracted']} retracted"
    )
    if counts["skipped_stale"] or counts["skipped_retracted"]:
        # Non-zero so automation cannot report green while an authored correction
        # was silently dropped, or while a battery still names a retracted fact.
        # Everything else already committed.
        print(
            f"EXIT 2: {counts['skipped_stale']} authored edit(s) DROPPED as stale YAML, "
            f"{counts['skipped_retracted']} fact(s) skipped as retracted "
            "(see warnings above). All other inserts completed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
