"""Guard logic in scripts/insert-multiturn-facts.py, exercised over SQLite.

Positional fact_ids make a turn's fact list append-only, so the script's job is
less "insert rows" than "refuse the edits that would silently remap ids". These
tests drive the guard functions directly against a SQLite fixture shaped like
reporting.question_facts, since the branch each guard takes depends on the DB's
latest version and status for a fact.

The script is a hyphenated filename, so it is loaded via importlib rather than a
plain import; that load also asserts the module is importable without running
main().
"""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "insert-multiturn-facts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("insert_multiturn_facts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_script_imports_without_running_main():
    """The module is importable (main() sits behind the __main__ guard), which is
    what makes these guard functions testable at all."""
    assert callable(script.main)
    assert callable(script.shrunken_questions)
    assert callable(script.remap_suspects)
    assert callable(script.bulk_edited_questions)


@pytest.fixture
def conn(tmp_path):
    """A SQLite stand-in for reporting.question_facts.

    SQLite has no schemas, so "reporting" is attached as a second database file —
    that makes the script's unmodified `reporting.question_facts` SQL run as-is.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/main.db")
    with engine.begin() as c:
        c.execute(text(f"ATTACH DATABASE '{tmp_path}/reporting.db' AS reporting"))
        c.execute(
            text("""
                CREATE TABLE reporting.question_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    fact_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    display_order INTEGER NOT NULL
                )
            """)
        )
        yield c


def _seed(conn, question_id, fact_id, fact_text, *, status="draft", version=1, order=1):
    conn.execute(
        text("""
            INSERT INTO reporting.question_facts
                (question_id, fact_id, fact_text, status, version, display_order)
            VALUES (:q, :f, :t, :s, :v, :o)
        """),
        {"q": question_id, "f": fact_id, "t": fact_text, "s": status, "v": version, "o": order},
    )


def _row(question_id, fact_id, fact_text, order):
    return {
        "question_id": question_id,
        "fact_id": fact_id,
        "fact_text": fact_text,
        "display_order": order,
    }


# --- shrink guard (fix 9) ---------------------------------------------------


def test_shrink_guard_counts_only_live_facts(conn):
    """Flagging a fact in the dashboard then dropping it from the YAML is the
    SANCTIONED retirement path. The guard counts live (latest-not-flagged) facts,
    so that sequence must not trip it."""
    _seed(conn, "q1", "q1-f1", "kept", order=1)
    _seed(conn, "q1", "q1-f2", "retired", status="flagged", order=2)

    # YAML now holds one fact; the DB holds two fact_ids but only one is live.
    assert script.shrunken_questions(conn, {"q1": 1}) == []


def test_shrink_guard_uses_the_latest_version_status(conn):
    """A fact flagged at v1 and re-authored as a draft v2 is live again — the guard
    reads the LATEST version's status, not any version's."""
    _seed(conn, "q1", "q1-f1", "kept", order=1)
    _seed(conn, "q1", "q1-f2", "old", status="flagged", version=1, order=2)
    _seed(conn, "q1", "q1-f2", "re-authored", status="draft", version=2, order=2)

    offending = script.shrunken_questions(conn, {"q1": 1})
    assert len(offending) == 1
    assert "live_db=2" in offending[0]


def test_shrink_guard_still_refuses_a_real_deletion(conn):
    """Removing a live fact from the YAML without flagging it is still refused."""
    _seed(conn, "q1", "q1-f1", "one", order=1)
    _seed(conn, "q1", "q1-f2", "two", order=2)

    offending = script.shrunken_questions(conn, {"q1": 1})
    assert len(offending) == 1
    assert "q1: yaml=1 live_db=2" in offending[0]


def test_shrink_guard_catches_a_turn_losing_all_its_facts(conn):
    """Zero YAML facts against live DB facts is the same deletion, and turns with no
    facts are in the count map precisely so this is reachable."""
    _seed(conn, "q1", "q1-f1", "one", order=1)
    assert script.shrunken_questions(conn, {"q1": 0}) != []


def test_shrink_guard_allows_growth(conn):
    """Appending facts is always fine — positional ids stay stable."""
    _seed(conn, "q1", "q1-f1", "one", order=1)
    assert script.shrunken_questions(conn, {"q1": 3}) == []


# --- delete-shift guard (fix 10a) -------------------------------------------


def test_delete_shift_signature_is_refused(conn):
    """Deleting the middle fact slides later facts up one id. Position 1's new YAML
    text is then the DB text of position 2, which would rewrite each fact under its
    neighbour's id."""
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    _seed(conn, "q1", "q1-f2", "beta", order=2)
    _seed(conn, "q1", "q1-f3", "gamma", order=3)

    # Author deleted "alpha"; beta/gamma shifted up into f1/f2.
    rows = [_row("q1", "q1-f1", "beta", 1), _row("q1", "q1-f2", "gamma", 2)]

    # f1's new text is f2's DB text, which is enough to refuse. f2 is the LAST
    # YAML position, so it has no next position to compare against — the guard
    # only needs one hit to stop the run, not a finding per shifted fact.
    findings = script.remap_suspects(conn, rows)
    assert len(findings) == 1
    assert "q1/q1-f1" in findings[0]
    assert "q1-f2" in findings[0]
    assert "deleted and the rest shifted up" in findings[0]


def test_delete_shift_flags_every_detectable_position(conn):
    """With a longer list, each shifted position except the last matches its
    neighbour's DB text, so the refusal names them all."""
    for n, txt in enumerate(("alpha", "beta", "gamma", "delta"), 1):
        _seed(conn, "q1", f"q1-f{n}", txt, order=n)

    # "alpha" deleted; beta/gamma/delta shifted up into f1/f2/f3.
    rows = [
        _row("q1", "q1-f1", "beta", 1),
        _row("q1", "q1-f2", "gamma", 2),
        _row("q1", "q1-f3", "delta", 3),
    ]
    findings = script.remap_suspects(conn, rows)
    assert len(findings) == 2  # f1 vs f2, f2 vs f3; f3 is last with no successor


def test_genuine_edit_is_not_a_delete_shift(conn):
    """Rewording one fact in place does not match any neighbour's text."""
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    _seed(conn, "q1", "q1-f2", "beta", order=2)

    rows = [_row("q1", "q1-f1", "alpha, clarified", 1), _row("q1", "q1-f2", "beta", 2)]
    assert script.remap_suspects(conn, rows) == []


def test_new_facts_are_not_delete_shifts(conn):
    """A fact with no DB row yet is an append, never a remap."""
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    rows = [_row("q1", "q1-f1", "alpha", 1), _row("q1", "q1-f2", "brand new", 2)]
    assert script.remap_suspects(conn, rows) == []


def test_delete_shift_does_not_cross_questions(conn):
    """The neighbour comparison is per question_id — an unrelated turn's text
    matching is coincidence, not a shift."""
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    _seed(conn, "q2", "q2-f1", "beta", order=1)

    rows = [_row("q1", "q1-f1", "beta", 1)]
    assert script.remap_suspects(conn, rows) == []


def test_delete_shift_catches_what_the_shrink_guard_cannot(conn):
    """The two guards are not redundant. Deleting one fact AND appending another
    keeps the YAML count stable, so the shrink guard passes — only the delete-shift
    signature sees that every surviving fact moved to its neighbour's id."""
    for n, txt in enumerate(("alpha", "beta", "gamma"), 1):
        _seed(conn, "q1", f"q1-f{n}", txt, order=n)

    # "alpha" deleted, beta/gamma shift up, "delta" appended — still 3 facts.
    rows = [
        _row("q1", "q1-f1", "beta", 1),
        _row("q1", "q1-f2", "gamma", 2),
        _row("q1", "q1-f3", "delta", 3),
    ]
    assert script.shrunken_questions(conn, {"q1": 3}) == []  # shrink guard blind here
    assert script.remap_suspects(conn, rows) != []


# --- bulk-edit guard (fix 10b) ----------------------------------------------


def test_two_changed_positions_in_one_turn_are_flagged(conn):
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    _seed(conn, "q1", "q1-f2", "beta", order=2)

    rows = [_row("q1", "q1-f1", "alpha v2", 1), _row("q1", "q1-f2", "beta v2", 2)]
    bulk = script.bulk_edited_questions(conn, rows)
    assert len(bulk) == 1
    assert "q1-f1" in bulk[0] and "q1-f2" in bulk[0]


def test_single_changed_position_is_not_bulk(conn):
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    _seed(conn, "q1", "q1-f2", "beta", order=2)

    rows = [_row("q1", "q1-f1", "alpha v2", 1), _row("q1", "q1-f2", "beta", 2)]
    assert script.bulk_edited_questions(conn, rows) == []


def test_bulk_guard_counts_per_question_not_across_the_run(conn):
    """One edit in each of two turns is a normal pass, not a bulk edit."""
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    _seed(conn, "q2", "q2-f1", "beta", order=1)

    rows = [_row("q1", "q1-f1", "alpha v2", 1), _row("q2", "q2-f1", "beta v2", 1)]
    assert script.bulk_edited_questions(conn, rows) == []


# --- apply_rows outcomes (drives the exit code) -----------------------------


def test_apply_rows_versions_a_draft_edit(conn):
    _seed(conn, "q1", "q1-f1", "alpha", status="draft", version=1, order=1)
    counts = script.apply_rows(conn, [_row("q1", "q1-f1", "alpha v2", 1)])

    assert counts["bumped"] == 1
    latest = conn.execute(
        text(
            "SELECT fact_text, version, status FROM reporting.question_facts "
            "WHERE fact_id = 'q1-f1' ORDER BY version DESC LIMIT 1"
        )
    ).fetchone()
    assert (latest.fact_text, latest.version, latest.status) == ("alpha v2", 2, "draft")


def test_apply_rows_skips_confirmed_edits_as_stale(conn):
    """Reviewed content is authoritative — the edit is DROPPED, which is what makes
    the non-zero exit necessary."""
    _seed(conn, "q1", "q1-f1", "reviewed text", status="confirmed", version=1, order=1)
    counts = script.apply_rows(conn, [_row("q1", "q1-f1", "yaml text", 1)])

    assert counts["skipped_stale"] == 1
    assert counts["bumped"] == 0
    rows = conn.execute(
        text("SELECT COUNT(*) FROM reporting.question_facts WHERE fact_id = 'q1-f1'")
    ).scalar_one()
    assert rows == 1  # nothing inserted over the reviewed row


def test_apply_rows_inserts_new_and_skips_unchanged(conn):
    _seed(conn, "q1", "q1-f1", "alpha", order=1)
    counts = script.apply_rows(
        conn, [_row("q1", "q1-f1", "alpha", 1), _row("q1", "q1-f2", "brand new", 2)]
    )
    assert counts == {"inserted": 1, "bumped": 0, "skipped_same": 1, "skipped_stale": 0}
