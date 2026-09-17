#!/usr/bin/env python3
"""Load a YAML battery's positional required_facts into reporting.question_facts as drafts.

Unlike ``insert-multiturn-facts.py``, the batteries this handles carry facts as
bare strings with no authored ``fact_id``. The column is a bigint fed by
``reporting.question_facts_fact_id_seq``, so ids are assigned by the database
exactly as the dashboard's own add-fact path assigns them.

That makes identity the problem this script has to solve. With no authored id,
the only stable handle on a fact is its TEXT, so rows are keyed on
``(question_id, fact_text)``:

  * text not present for the question → insert version 1 as 'draft',
  * text already present in any version → skip, whatever its status.

The consequence is deliberate and worth stating: **editing a fact's wording in
the YAML inserts a new fact rather than versioning the old one.** A positional
loader cannot tell "fact 3 was reworded" from "fact 3 was replaced", and
guessing wrong either overwrites a reviewer's decision or silently drops an
authored correction. So this script only ever adds. Retiring the superseded
fact is a human action in the dashboard (flag or retract), which is also where
the audit trail belongs.

Reviewed content is never touched: the script writes no UPDATE and no DELETE.

Refuses to run when the battery names a question absent from
reporting.questions — a fact hanging off a non-existent question is invisible in
the dashboard and unreachable by the resolver, so it is a silent no-op rather
than an error the operator would notice.

Run over the tunnel:

    scripts/eval-tunnel-open   # in another terminal
    DATABASE_URL=... uv run python scripts/insert-battery-facts.py \
        eval/questions/student_authored_battery.yaml [--dry-run]

--dry-run stays DB-free: it prints what it would consider and never connects.

Exit codes: 0 success, 2 completed but some questions were missing from
reporting.questions (their facts were not loaded).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import yaml
from sqlalchemy import create_engine, text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

REPO_ROOT = Path(__file__).resolve().parent.parent


class FactRow(TypedDict):
    question_id: str
    fact_text: str


EXISTING_TEXTS = text("""
    SELECT fact_text
    FROM reporting.question_facts
    WHERE question_id = :question_id
""")

QUESTION_EXISTS = text("""
    SELECT 1 FROM reporting.questions WHERE question_id = :question_id
""")

# fact_id from the sequence and display_order from max+1, matching the
# dashboard's addFactOn so rows from either path are indistinguishable.
INSERT_FACT = text("""
    INSERT INTO reporting.question_facts
        (fact_id, version, question_id, display_order, fact_text, status, updated_by)
    SELECT nextval('reporting.question_facts_fact_id_seq'),
           1,
           :question_id,
           COALESCE(
               (SELECT max(display_order) + 1
                FROM reporting.question_facts
                WHERE question_id = :question_id),
               0),
           :fact_text,
           'draft',
           :updated_by
""")


def build_rows(battery_path: Path) -> list[FactRow]:
    """Flatten the battery into (question_id, fact_text) rows, in file order."""
    data = yaml.safe_load(battery_path.read_text())
    questions = data if isinstance(data, list) else data.get("questions", data)
    rows: list[FactRow] = []
    for q in questions:
        qid = str(q["id"])
        for fact in q.get("required_facts") or []:
            if not isinstance(fact, str):
                raise SystemExit(
                    f"{qid}: expected a bare-string fact, got {type(fact).__name__}. "
                    "Batteries with authored fact_ids belong in insert-multiturn-facts.py."
                )
            text_value = fact.strip()
            if text_value:
                rows.append({"question_id": qid, "fact_text": text_value})
    return rows


def apply_rows(conn: Connection, rows: list[FactRow], updated_by: str) -> dict[str, Any]:
    """Insert each fact whose text is new for its question. Returns per-outcome counts."""
    counts: dict[str, Any] = {"inserted": 0, "skipped_present": 0, "missing_questions": []}
    seen_texts: dict[str, set[str]] = {}

    for r in rows:
        qid = r["question_id"]
        if qid not in seen_texts:
            if conn.execute(QUESTION_EXISTS, {"question_id": qid}).fetchone() is None:
                counts["missing_questions"].append(qid)
                seen_texts[qid] = set()
                continue
            seen_texts[qid] = {
                row.fact_text.strip() for row in conn.execute(EXISTING_TEXTS, {"question_id": qid})
            }
        elif qid in counts["missing_questions"]:
            continue

        if r["fact_text"] in seen_texts[qid]:
            counts["skipped_present"] += 1
            continue

        conn.execute(
            INSERT_FACT,
            {"question_id": qid, "fact_text": r["fact_text"], "updated_by": updated_by},
        )
        # Track within this run too, so a battery that repeats a fact verbatim
        # for one question inserts it once.
        seen_texts[qid].add(r["fact_text"])
        counts["inserted"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("battery", help="Path to the YAML battery file")
    parser.add_argument("--dry-run", action="store_true", help="Print rows; never connect")
    parser.add_argument(
        "--updated-by",
        default="battery-loader",
        help="Value for question_facts.updated_by (default: battery-loader)",
    )
    args = parser.parse_args()

    path = Path(args.battery)
    if not path.is_absolute():
        path = REPO_ROOT / path
    rows = build_rows(path)

    by_question: dict[str, int] = {}
    for r in rows:
        by_question[r["question_id"]] = by_question.get(r["question_id"], 0) + 1
    print(f"{len(rows)} facts across {len(by_question)} questions in {path.name}")

    if args.dry_run:
        for qid, n in by_question.items():
            print(f"  {qid}: {n}")
        return

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL must be set (or use --dry-run)")

    engine = create_engine(database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    with engine.begin() as conn:
        counts = apply_rows(conn, rows, args.updated_by)

    print(f"inserted={counts['inserted']} skipped_present={counts['skipped_present']}")
    missing = counts["missing_questions"]
    if missing:
        print(
            f"WARNING {len(missing)} question(s) absent from reporting.questions; "
            f"their facts were NOT loaded: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
