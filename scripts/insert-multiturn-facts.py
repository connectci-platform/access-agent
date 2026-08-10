#!/usr/bin/env python3
"""One-shot: insert authored multi-turn facts as draft rows in reporting.question_facts.

Reads the multiturn battery files, emits one draft version-1 row per fact with a
stable fact_id ({thread_id}-{turn_id}-f{n}), skipping any (question_id, fact_id)
that already has rows (idempotent re-run). Run over the 5433 tunnel:

    scripts/eval-tunnel-open   # in another terminal
    DATABASE_URL=... uv run python scripts/insert-multiturn-facts.py [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

BATTERIES = [
    "eval/questions/multiturn_support_battery.yaml",
    "eval/questions/multiturn_compaction_battery.json",
]

INSERT = text("""
    INSERT INTO reporting.question_facts
        (question_id, fact_id, fact_text, status, version, display_order)
    SELECT :question_id, :fact_id, :fact_text, 'draft', 1, :display_order
    WHERE NOT EXISTS (
        SELECT 1 FROM reporting.question_facts
        WHERE question_id = :question_id AND fact_id = :fact_id
    )
""")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
                            "fact_text": str(fact),
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
    inserted = 0
    with engine.begin() as conn:
        for r in rows:
            inserted += conn.execute(INSERT, r).rowcount
    print(f"inserted {inserted} new draft rows ({len(rows) - inserted} already present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
