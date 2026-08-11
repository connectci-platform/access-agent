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
orphan rows the resolver keeps serving. Three guards enforce that, all checked
before any insert for the battery:

  1. SHRINK. The script REFUSES when a turn's YAML holds fewer facts than the
     DB has LIVE fact_ids for that question_id. "Live" means the fact's latest
     version is not flagged, matching the resolver's retirement semantics — the
     sanctioned way to retire a fact is to flag it in the dashboard and then
     drop it from the YAML, and counting flagged facts would refuse exactly that.
  2. DELETE-SHIFT. The script REFUSES when a changed position's YAML text equals
     the DB-latest text of the NEXT position for the same question. That is the
     signature of a fact having been deleted from the middle of the list: every
     later fact slides up one id, so the "edits" are really a positional remap
     that would rewrite each surviving fact under its neighbour's id.
  3. BULK EDIT. The script REFUSES when 2 or more positions of one turn change
     text in a single run, unless ``--allow-bulk-edit`` is passed. A genuine
     multi-fact re-authoring is legitimate but rare; the same pattern is what a
     reorder produces, and a reorder is unrecoverable once inserted.

Exit codes: 0 success, 1 refusal (nothing inserted for the offending battery),
2 completed with stale-YAML skips. A stale skip means the DB's confirmed text
differs from the YAML, so the YAML edit was DROPPED — that must not be a silent
success, or automation reports green while a correction never landed.

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
    SELECT version, fact_text, status
    FROM reporting.question_facts
    WHERE question_id = :question_id AND fact_id = :fact_id
    ORDER BY version DESC
    LIMIT 1
""")

# How many LIVE facts the DB holds for a question: fact_ids whose highest version
# is not flagged. Positional fact_ids are append-only, so a YAML list shorter than
# this would remap ids onto different texts. Flagged-latest facts are excluded to
# match the resolver, which already skips them — flagging in the dashboard and then
# dropping from the YAML is the SANCTIONED retirement path and must not be refused.
# Written with a correlated MAX(version) rather than DISTINCT ON so the same SQL
# runs on the SQLite fixture the guard tests use.
LIVE_FACT_COUNT = text("""
    SELECT COUNT(*)
    FROM reporting.question_facts f
    WHERE f.question_id = :question_id
      AND f.status <> 'flagged'
      AND f.version = (
          SELECT MAX(v.version)
          FROM reporting.question_facts v
          WHERE v.question_id = f.question_id AND v.fact_id = f.fact_id
      )
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


def build_rows(batteries: list[str]) -> tuple[list[FactRow], dict[str, int]]:
    """Flatten the batteries into insert-shaped rows plus each turn's YAML fact count.

    The count map covers every question the batteries mention, INCLUDING turns with
    zero facts — dropping a turn's facts entirely is exactly the deletion the
    append-only rule forbids, so it must reach the shrink guard.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from src.eval.multiturn import load_thread_battery

    rows: list[FactRow] = []
    yaml_fact_counts: dict[str, int] = {}
    for battery in batteries:
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
    return rows, yaml_fact_counts


def shrunken_questions(conn: Connection, yaml_fact_counts: dict[str, int]) -> list[str]:
    """question_ids whose YAML fact list is shorter than the DB's LIVE fact_ids.

    Live = latest version not flagged, matching the resolver. Flagging a fact in
    the dashboard and then removing it from the YAML is the sanctioned retirement
    path, so a flagged fact must not keep the guard tripping forever.
    """
    offending = []
    for question_id, yaml_count in sorted(yaml_fact_counts.items()):
        db_count = conn.execute(LIVE_FACT_COUNT, {"question_id": question_id}).scalar_one()
        if yaml_count < db_count:
            offending.append(f"{question_id}: yaml={yaml_count} live_db={db_count}")
    return offending


def _latest(conn: Connection, question_id: str, fact_id: str) -> "Row[Any] | None":
    return conn.execute(LATEST, {"question_id": question_id, "fact_id": fact_id}).fetchone()


def remap_suspects(conn: Connection, rows: list[FactRow]) -> list[str]:
    """Changed positions that look like a positional remap rather than an edit.

    Two signatures, both unrecoverable once inserted because each surviving fact
    would be rewritten under its neighbour's id:

    - DELETE-SHIFT: a changed position whose YAML text equals the DB-latest text
      of the NEXT position for the same question. Deleting a fact from the middle
      of a list slides every later fact up one id, which reaches this script as a
      run of "edits" that are really the neighbours' texts.
    - BULK EDIT: 2+ positions of one turn changing text in a single run. Genuine
      multi-fact re-authoring is legitimate but rare, and a reorder is
      indistinguishable from it here, so it needs an explicit opt-in.

    Returns the delete-shift findings only; bulk edits are reported by
    ``bulk_edited_questions`` so the two can carry different messages.
    """
    findings: list[str] = []
    by_question: dict[str, list[FactRow]] = {}
    for r in rows:
        by_question.setdefault(r["question_id"], []).append(r)

    for question_id, group in sorted(by_question.items()):
        ordered = sorted(group, key=lambda r: r["display_order"])
        for position, row in enumerate(ordered):
            latest = _latest(conn, question_id, row["fact_id"])
            if latest is None or latest.fact_text == row["fact_text"]:
                continue
            nxt = ordered[position + 1] if position + 1 < len(ordered) else None
            if nxt is None:
                continue
            next_latest = _latest(conn, question_id, nxt["fact_id"])
            if next_latest is not None and next_latest.fact_text == row["fact_text"]:
                findings.append(
                    f"{question_id}/{row['fact_id']}: its new YAML text is the DB text of "
                    f"{nxt['fact_id']} — looks like a fact was deleted and the rest shifted up"
                )
    return findings


def bulk_edited_questions(conn: Connection, rows: list[FactRow]) -> list[str]:
    """question_ids where 2+ positions change text in this run (see ``remap_suspects``)."""
    changed: dict[str, list[str]] = {}
    for r in rows:
        question_id = r["question_id"]
        latest = _latest(conn, question_id, r["fact_id"])
        if latest is not None and latest.fact_text != r["fact_text"]:
            changed.setdefault(question_id, []).append(r["fact_id"])
    return [
        f"{question_id}: {', '.join(sorted(fact_ids))}"
        for question_id, fact_ids in sorted(changed.items())
        if len(fact_ids) >= 2
    ]


def apply_rows(conn: Connection, rows: list[FactRow]) -> dict[str, int]:
    """Insert/version each row, returning the per-outcome counts."""
    counts = {"inserted": 0, "bumped": 0, "skipped_same": 0, "skipped_stale": 0}
    for r in rows:
        latest = _latest(conn, r["question_id"], r["fact_id"])
        if latest is None:
            version = 1
            counts["inserted"] += 1
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
                "the battery file. Not overwriting. Two sanctioned paths: update the "
                "YAML from the confirmed DB text, or flag that version in the "
                "dashboard and re-author the fact so this run can version it up.",
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
    parser.add_argument(
        "--allow-bulk-edit",
        action="store_true",
        help="Permit 2+ fact texts of one turn changing in a single run (reorder risk).",
    )
    args = parser.parse_args()

    rows, yaml_fact_counts = build_rows(BATTERIES)

    print(f"{len(rows)} fact rows from {len(BATTERIES)} batteries")
    if args.dry_run:
        # DB-free by contract, so the shrink/remap/stale checks (which need the DB) don't run.
        for r in rows[:5]:
            print(r)
        return 0

    url = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url)
    with engine.begin() as conn:
        shrunk = shrunken_questions(conn, yaml_fact_counts)
        if shrunk:
            print(
                "REFUSING: these question_ids have fewer facts in YAML than live "
                "(non-flagged) facts in the DB, so positional fact_ids would remap "
                "onto different texts. Retire a fact by flagging it in the dashboard "
                "first, then remove it from the YAML:\n  " + "\n  ".join(shrunk),
                file=sys.stderr,
            )
            return 1

        shifted = remap_suspects(conn, rows)
        if shifted:
            print(
                "REFUSING: a changed fact's new text matches the DB text of the NEXT "
                "position, which is what deleting a fact mid-list looks like. Inserting "
                "would rewrite each fact under its neighbour's id:\n  " + "\n  ".join(shifted),
                file=sys.stderr,
            )
            return 1

        bulk = bulk_edited_questions(conn, rows)
        if bulk and not args.allow_bulk_edit:
            print(
                "REFUSING: 2+ fact texts changed for these turns in one run. A reorder "
                "is indistinguishable from genuine multi-fact re-authoring here, and a "
                "reorder remaps positional ids irreversibly. Re-run with "
                "--allow-bulk-edit if the edits are intentional:\n  " + "\n  ".join(bulk),
                file=sys.stderr,
            )
            return 1

        counts = apply_rows(conn, rows)

    print(
        f"inserted {counts['inserted']} new, bumped {counts['bumped']} edited, "
        f"skipped {counts['skipped_same']} unchanged, skipped {counts['skipped_stale']} stale"
    )
    if counts["skipped_stale"]:
        # Non-zero so automation cannot report green while an authored correction
        # was silently dropped. Everything else already committed.
        print(
            f"EXIT 2: {counts['skipped_stale']} authored edit(s) were DROPPED as stale YAML "
            "(see warnings above). All other inserts completed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
