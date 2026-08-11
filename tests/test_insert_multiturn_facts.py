"""Upsert logic in scripts/insert-multiturn-facts.py, exercised over SQLite.

Facts carry explicit fact_ids, so the script's whole job is the per-id branch:
insert, version up a draft, refuse to overwrite reviewed text, or do nothing.
Each branch depends on the DB's latest version and status for that id, so the
tests drive the functions against a SQLite fixture shaped like
reporting.question_facts.

The script is a hyphenated filename, so it is loaded via importlib rather than a
plain import; that load also asserts the module is importable without running
main().
"""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    what makes these functions testable at all."""
    assert callable(script.main)
    assert callable(script.build_rows)
    assert callable(script.apply_rows)


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


def _row(question_id, fact_id, fact_text, order=1):
    return {
        "question_id": question_id,
        "fact_id": fact_id,
        "fact_text": fact_text,
        "display_order": order,
    }


def _versions(conn, fact_id):
    return conn.execute(
        text(
            "SELECT version, fact_text, status FROM reporting.question_facts "
            "WHERE fact_id = :f ORDER BY version"
        ),
        {"f": fact_id},
    ).fetchall()


# --- the four upsert branches ------------------------------------------------


def test_absent_fact_inserts_version_one_as_draft(conn):
    counts = script.apply_rows(conn, [_row("q1", "q1-new", "brand new")])

    assert counts["inserted"] == 1
    assert [(v.version, v.fact_text, v.status) for v in _versions(conn, "q1-new")] == [
        (1, "brand new", "draft")
    ]


def test_edited_draft_versions_up(conn):
    """The resolver serves the latest non-flagged version, so bumping is how an
    AUTHOR-pass edit actually reaches the judge."""
    _seed(conn, "q1", "q1-a", "alpha", status="draft", version=1)
    counts = script.apply_rows(conn, [_row("q1", "q1-a", "alpha v2")])

    assert counts["bumped"] == 1
    assert [(v.version, v.fact_text, v.status) for v in _versions(conn, "q1-a")] == [
        (1, "alpha", "draft"),
        (2, "alpha v2", "draft"),
    ]


@pytest.mark.parametrize("status", ["confirmed", "flagged"])
def test_edited_non_draft_latest_is_skipped_as_stale(conn, capsys, status):
    """Reviewed content is authoritative — the edit is DROPPED, and the warning
    states the two remedies that actually work."""
    _seed(conn, "q1", "q1-a", "reviewed text", status=status, version=1)
    counts = script.apply_rows(conn, [_row("q1", "q1-a", "yaml text")])

    assert counts["skipped_stale"] == 1
    assert counts["bumped"] == 0
    assert len(_versions(conn, "q1-a")) == 1  # nothing inserted over the reviewed row

    warning = capsys.readouterr().err
    assert "q1/q1-a" in warning
    assert "update the YAML to the reviewed text" in warning
    assert "new draft version in the dashboard" in warning


def test_identical_text_is_skipped_silently(conn, capsys):
    _seed(conn, "q1", "q1-a", "alpha", status="confirmed", version=1)
    counts = script.apply_rows(conn, [_row("q1", "q1-a", "alpha")])

    assert counts == {"inserted": 0, "bumped": 0, "skipped_same": 1, "skipped_stale": 0}
    assert capsys.readouterr().err == ""


# --- identity is the id, not the position ------------------------------------


def test_reordering_and_deleting_facts_touches_nothing_else(conn):
    """Positional remap defects cannot exist: a fact dropped from the YAML is left
    alone, and the survivors match on their ids regardless of new positions."""
    _seed(conn, "q1", "q1-a", "alpha", order=1)
    _seed(conn, "q1", "q1-b", "beta", order=2)
    _seed(conn, "q1", "q1-c", "gamma", order=3)

    # "alpha" removed from the YAML; the rest reordered.
    counts = script.apply_rows(
        conn, [_row("q1", "q1-c", "gamma", 1), _row("q1", "q1-b", "beta", 2)]
    )

    assert counts == {"inserted": 0, "bumped": 0, "skipped_same": 2, "skipped_stale": 0}
    # The dropped fact keeps its row untouched — retirement is flagging, not deletion.
    assert [(v.version, v.fact_text) for v in _versions(conn, "q1-a")] == [(1, "alpha")]


def test_latest_version_status_drives_the_branch(conn):
    """A fact confirmed at v1 and re-authored as draft v2 versions up again; the
    branch reads the LATEST version's status, not any version's."""
    _seed(conn, "q1", "q1-a", "v1 text", status="confirmed", version=1)
    _seed(conn, "q1", "q1-a", "v2 text", status="draft", version=2)

    counts = script.apply_rows(conn, [_row("q1", "q1-a", "v3 text")])
    assert counts["bumped"] == 1
    assert _versions(conn, "q1-a")[-1].version == 3


def test_one_latest_fetch_per_fact(conn):
    """The branch and the next version both come from a single read per fact."""
    _seed(conn, "q1", "q1-a", "alpha")
    rows = [_row("q1", "q1-a", "alpha v2"), _row("q1", "q1-b", "new")]

    with patch.object(script, "_latest", wraps=script._latest) as spy:
        script.apply_rows(conn, rows)

    assert spy.call_count == len(rows)


# --- exit codes --------------------------------------------------------------


@contextmanager
def _noop_transaction(conn):
    """Stand in for engine.begin(): the fixture connection is already in one."""
    yield conn


def _run_main(conn, rows, monkeypatch):
    """Drive main() against the SQLite fixture, bypassing the real engine."""
    engine = SimpleNamespace(begin=lambda: _noop_transaction(conn))

    monkeypatch.setenv("DATABASE_URL", "postgresql://ignored/db")
    monkeypatch.setattr(script.sys, "argv", ["insert-multiturn-facts.py"])
    with (
        patch.object(script, "build_rows", return_value=rows),
        patch.object(script, "create_engine", return_value=engine),
    ):
        return script.main()


def test_main_exits_2_when_an_edit_was_dropped_as_stale(conn, monkeypatch, capsys):
    """A dropped edit must not report green — but everything else still lands."""
    _seed(conn, "q1", "q1-a", "reviewed", status="confirmed", version=1)
    rows = [_row("q1", "q1-a", "stale yaml"), _row("q1", "q1-b", "brand new")]

    assert _run_main(conn, rows, monkeypatch) == 2
    assert "EXIT 2" in capsys.readouterr().err
    # The other work completed before the non-zero exit.
    assert [(v.version, v.fact_text) for v in _versions(conn, "q1-b")] == [(1, "brand new")]


def test_main_exits_0_on_a_clean_run(conn, monkeypatch):
    assert _run_main(conn, [_row("q1", "q1-b", "brand new")], monkeypatch) == 0


def test_dry_run_is_db_free(monkeypatch, capsys):
    """--dry-run makes no connection, so it works without DATABASE_URL."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(script.sys, "argv", ["insert-multiturn-facts.py", "--dry-run"])

    with patch.object(script, "create_engine", side_effect=AssertionError("connected")):
        assert script.main() == 0

    assert "fact rows from" in capsys.readouterr().out


def test_build_rows_uses_the_authored_fact_ids():
    """The real batteries: every row carries its explicit id, ids are unique, and
    display_order is per-turn position."""
    rows = script.build_rows(script.BATTERIES)

    assert len(rows) == 36
    assert len({(r["question_id"], r["fact_id"]) for r in rows}) == len(rows)
    assert all(r["fact_id"].startswith(r["question_id"].replace("_", "-")) for r in rows)
    first = next(r for r in rows if r["question_id"] == "mt-followup-01_t2")
    assert first["fact_id"] == "mt-followup-01-t2-subset"
    assert first["display_order"] == 1
