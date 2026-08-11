#!/usr/bin/env python3
"""One-shot: insert authored multi-turn facts as draft rows in reporting.question_facts.

Reads the multiturn battery files and emits one draft row per fact with a stable
fact_id ({thread_id}-{turn_id}-f{n}). Re-runs are versioning, not just idempotent:
for each (question_id, fact_id) the script compares the authored text against the
DB's latest version AND its status, then

  * inserts version 1 when the fact is new,
  * inserts version latest+1 (same display_order) when the latest is a DRAFT and
    the text CHANGED — the resolver takes the latest non-flagged version, so this
    is how an AUTHOR-pass edit actually reaches the judge,
  * SKIPS WITH A WARNING when the latest is confirmed (or anything other than a
    draft) and the text differs: the DB is authoritative after review and the
    script never overwrites reviewed content — the YAML is the stale copy,
  * skips silently when the text is unchanged.

Positional fact_ids make a turn's YAML fact list append-only after the first
insert: deleting or reordering facts would remap ids onto different texts and
orphan rows the resolver keeps serving. So the script REFUSES outright when a
turn's YAML holds fewer facts than the DB has distinct fact_ids for that
question_id. Retire a fact by flagging it in the dashboard instead — the
resolver already excludes flagged-latest facts.

Run over the 5433 tunnel:

    scripts/eval-tunnel-open   # in another terminal
    DATABASE_URL=... uv run python scripts/insert-multiturn-facts.py [--dry-run]

--dry-run stays DB-free: it prints the rows the script would consider and makes
no connection, so it cannot perform the DB-dependent checks.
"""

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

REPO_ROOT = Path(__file__).resolve().parent.parent

BATTERIES = [
    str(REPO_ROOT / "eval/questions/multiturn_support_battery.yaml"),
    str(REPO_ROOT / "eval/questions/multiturn_compaction_battery.json"),
]

# The DB's current state for one (question_id, fact_id): highest version, its text,
# and its status — status decides whether an edit may version up or must be refused
# as stale YAML, so it is part of the branch, not decoration.
LATEST = text("""
    SELECT version, fact_text, status
    FROM reporting.question_facts
    WHERE question_id = :question_id AND fact_id = :fact_id
    ORDER BY version DESC
    LIMIT 1
""")

# How many distinct facts the DB already holds for a question. Positional fact_ids
# are append-only, so a YAML list shorter than this would remap ids onto different
# texts and orphan rows the resolver keeps serving.
DISTINCT_FACT_COUNT = text("""
    SELECT COUNT(DISTINCT fact_id)
    FROM reporting.question_facts
    WHERE question_id = :question_id
""")

INSERT_VERSION = text("""
    INSERT INTO reporting.question_facts
        (question_id, fact_id, fact_text, status, version, display_order)
    VALUES (:question_id, :fact_id, :fact_text, 'draft', :version, :display_order)
""")


def _fact_text(fact: object) -> str:
    """Authored facts are plain strings; tolerate dicts carrying fact_text."""
    if isinstance(fact, dict):
        return str(fact.get("fact_text", ""))
    return str(fact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from src.eval.multiturn import load_thread_battery

    rows = []
    yaml_fact_counts: dict[str, int] = {}
    for battery in BATTERIES:
        for thread in load_thread_battery(battery):
            for q in thread["questions"]:
                question_id = f"{thread['thread_id']}_{q['turn_id']}"
                facts = q.get("required_facts") or []
                yaml_fact_counts[question_id] = len(facts)
                for n, fact in enumerate(facts, 1):
                    rows.append(
                        {
                            "question_id": question_id,
                            "fact_id": f"{thread['thread_id']}-{q['turn_id']}-f{n}",
                            "fact_text": _fact_text(fact),
                            "display_order": n,
                        }
                    )

    print(f"{len(rows)} fact rows from {len(BATTERIES)} batteries")
    if args.dry_run:
        # DB-free by contract, so the shrink/stale checks (which need the DB) don't run.
        for r in rows[:5]:
            print(r)
        return 0

    url = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url)
    inserted = bumped = skipped_same = skipped_stale = 0
    with engine.begin() as conn:
        shrunk = _shrunken_questions(conn, yaml_fact_counts)
        if shrunk:
            print(
                "REFUSING: these question_ids have fewer facts in YAML than in the DB, so "
                "positional fact_ids would remap onto different texts. Retire facts by "
                "flagging them in the dashboard instead:\n  " + "\n  ".join(shrunk),
                file=sys.stderr,
            )
            return 1

        for r in rows:
            latest = conn.execute(
                LATEST, {"question_id": r["question_id"], "fact_id": r["fact_id"]}
            ).fetchone()
            if latest is None:
                version = 1
                inserted += 1
            elif latest.fact_text == r["fact_text"]:
                skipped_same += 1
                continue
            elif latest.status != "draft":
                # Reviewed content is authoritative; the YAML is the stale copy.
                skipped_stale += 1
                print(
                    f"WARNING stale YAML: {r['question_id']}/{r['fact_id']} v{latest.version} "
                    f"is {latest.status} in the DB and differs from the battery file. "
                    "Not overwriting — update the YAML from the DB, or edit via the dashboard.",
                    file=sys.stderr,
                )
                continue
            else:
                version = latest.version + 1
                bumped += 1
            conn.execute(INSERT_VERSION, {**r, "version": version})
    print(
        f"inserted {inserted} new, bumped {bumped} edited, "
        f"skipped {skipped_same} unchanged, skipped {skipped_stale} stale"
    )
    return 0


def _shrunken_questions(conn: Connection, yaml_fact_counts: dict[str, int]) -> list[str]:
    """question_ids whose YAML fact list is shorter than the DB's distinct fact_ids.

    Checked for every question the batteries mention, including ones with zero
    YAML facts — dropping a turn's facts entirely is exactly the deletion the
    append-only rule forbids.
    """
    offending = []
    for question_id, yaml_count in sorted(yaml_fact_counts.items()):
        db_count = conn.execute(DISTINCT_FACT_COUNT, {"question_id": question_id}).scalar_one()
        if yaml_count < db_count:
            offending.append(f"{question_id}: yaml={yaml_count} db={db_count}")
    return offending


if __name__ == "__main__":
    raise SystemExit(main())
