#!/usr/bin/env python3
"""One-shot: insert authored multi-turn facts as draft rows in reporting.question_facts.

Reads the multiturn battery files and emits one draft row per fact with a stable
fact_id ({thread_id}-{turn_id}-f{n}). Re-runs are versioning, not just idempotent:
for each (question_id, fact_id) the script compares the authored text against the
DB's latest version and

  * inserts version 1 when the fact is new,
  * inserts version latest+1 (same display_order) when the text CHANGED — the
    resolver takes the latest non-flagged version, so this is how an AUTHOR-pass
    edit actually reaches the judge,
  * skips when the text is unchanged.

Run over the 5433 tunnel:

    scripts/eval-tunnel-open   # in another terminal
    DATABASE_URL=... uv run python scripts/insert-multiturn-facts.py [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent

BATTERIES = [
    str(REPO_ROOT / "eval/questions/multiturn_support_battery.yaml"),
    str(REPO_ROOT / "eval/questions/multiturn_compaction_battery.json"),
]

# The DB's current state for one (question_id, fact_id): highest version and its text.
LATEST = text("""
    SELECT version, fact_text
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
    for battery in BATTERIES:
        for thread in load_thread_battery(battery):
            for q in thread["questions"]:
                for n, fact in enumerate(q.get("required_facts") or [], 1):
                    rows.append(
                        {
                            "question_id": f"{thread['thread_id']}_{q['turn_id']}",
                            "fact_id": f"{thread['thread_id']}-{q['turn_id']}-f{n}",
                            "fact_text": _fact_text(fact),
                            "display_order": n,
                        }
                    )

    print(f"{len(rows)} fact rows from {len(BATTERIES)} batteries")
    if args.dry_run:
        for r in rows[:5]:
            print(r)
        return 0

    url = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url)
    inserted = bumped = skipped = 0
    with engine.begin() as conn:
        for r in rows:
            latest = conn.execute(
                LATEST, {"question_id": r["question_id"], "fact_id": r["fact_id"]}
            ).fetchone()
            if latest is None:
                version = 1
                inserted += 1
            elif latest.fact_text == r["fact_text"]:
                skipped += 1
                continue
            else:
                version = latest.version + 1
                bumped += 1
            conn.execute(INSERT_VERSION, {**r, "version": version})
    print(f"inserted {inserted} new, bumped {bumped} edited, skipped {skipped} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
