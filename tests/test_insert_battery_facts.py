"""Insert logic in scripts/insert-battery-facts.py, exercised over SQLite.

These batteries carry facts as bare strings with no authored id, so identity is
the fact TEXT and the script's whole job is: insert what is new for the
question, skip what is already there in any version, and refuse to load facts
for a question the dashboard does not know about.

The script is a hyphenated filename, so it loads via importlib; that load also
asserts the module is importable without running main().
"""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "insert-battery-facts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("insert_battery_facts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_script_imports_without_running_main():
    assert callable(script.main)
    assert callable(script.build_rows)
    assert callable(script.apply_rows)


@pytest.fixture
def conn(tmp_path):
    """SQLite stand-in for the reporting schema.

    SQLite has no schemas, so "reporting" is attached as a second database file.
    It also has no sequences, so nextval() is registered as a function over an
    AUTOINCREMENT table — the script's unmodified SQL then runs as-is.
    """
    engine = create_engine(f"sqlite:///{tmp_path}/main.db")
    with engine.begin() as c:
        c.execute(text(f"ATTACH DATABASE '{tmp_path}/reporting.db' AS reporting"))
        c.execute(
            text("""
                CREATE TABLE reporting.question_facts (
                    fact_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    question_id TEXT NOT NULL,
                    display_order INTEGER NOT NULL,
                    fact_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_by TEXT
                )
            """)
        )
        c.execute(text("CREATE TABLE reporting.questions (question_id TEXT PRIMARY KEY)"))
        c.execute(text("CREATE TABLE reporting.seq (n INTEGER PRIMARY KEY AUTOINCREMENT)"))
        raw = c.connection.driver_connection

        def nextval(_name):
            cur = raw.execute("INSERT INTO reporting.seq DEFAULT VALUES")
            return cur.lastrowid

        raw.create_function("nextval", 1, nextval)
        yield c


def _add_question(conn, qid):
    conn.execute(text("INSERT INTO reporting.questions (question_id) VALUES (:q)"), {"q": qid})


def _seed_fact(conn, qid, fact_text, *, status="draft", version=1, order=0, fact_id=900):
    conn.execute(
        text("""
            INSERT INTO reporting.question_facts
                (fact_id, version, question_id, display_order, fact_text, status)
            VALUES (:f, :v, :q, :o, :t, :s)
        """),
        {"f": fact_id, "v": version, "q": qid, "o": order, "t": fact_text, "s": status},
    )


def _rows(conn, qid):
    return conn.execute(
        text(
            "SELECT fact_text, status, version, display_order FROM reporting.question_facts "
            "WHERE question_id = :q ORDER BY display_order"
        ),
        {"q": qid},
    ).fetchall()


def test_inserts_new_facts_as_draft_version_one(conn):
    _add_question(conn, "sa-1")
    counts = script.apply_rows(
        conn,
        [{"question_id": "sa-1", "fact_text": "A"}, {"question_id": "sa-1", "fact_text": "B"}],
        "tester",
    )
    assert counts["inserted"] == 2
    got = _rows(conn, "sa-1")
    assert [r.fact_text for r in got] == ["A", "B"]
    assert {r.status for r in got} == {"draft"}
    assert {r.version for r in got} == {1}
    # display_order starts at 0 and advances, matching the dashboard's addFactOn
    assert [r.display_order for r in got] == [0, 1]


def test_skips_a_fact_whose_text_is_already_present(conn):
    _add_question(conn, "sa-1")
    _seed_fact(conn, "sa-1", "A")
    counts = script.apply_rows(conn, [{"question_id": "sa-1", "fact_text": "A"}], "tester")
    assert counts == {"inserted": 0, "skipped_present": 1, "missing_questions": []}
    assert len(_rows(conn, "sa-1")) == 1


def test_skips_present_text_whatever_its_status(conn):
    """A confirmed or flagged fact is reviewed content; re-running never touches it."""
    _add_question(conn, "sa-1")
    _seed_fact(conn, "sa-1", "A", status="confirmed")
    _seed_fact(conn, "sa-1", "B", status="flagged", fact_id=901)
    counts = script.apply_rows(
        conn,
        [{"question_id": "sa-1", "fact_text": "A"}, {"question_id": "sa-1", "fact_text": "B"}],
        "tester",
    )
    assert counts["inserted"] == 0
    assert counts["skipped_present"] == 2
    assert {r.status for r in _rows(conn, "sa-1")} == {"confirmed", "flagged"}


def test_refuses_facts_for_a_question_the_dashboard_lacks(conn):
    """A fact on an unknown question is invisible and unreachable, so report it."""
    counts = script.apply_rows(conn, [{"question_id": "ghost", "fact_text": "A"}], "tester")
    assert counts["inserted"] == 0
    assert counts["missing_questions"] == ["ghost"]
    assert _rows(conn, "ghost") == []


def test_missing_question_is_reported_once_across_its_facts(conn):
    counts = script.apply_rows(
        conn,
        [{"question_id": "ghost", "fact_text": "A"}, {"question_id": "ghost", "fact_text": "B"}],
        "tester",
    )
    assert counts["missing_questions"] == ["ghost"]
    assert counts["inserted"] == 0


def test_is_idempotent_across_two_runs(conn):
    _add_question(conn, "sa-1")
    rows = [{"question_id": "sa-1", "fact_text": "A"}, {"question_id": "sa-1", "fact_text": "B"}]
    script.apply_rows(conn, rows, "tester")
    second = script.apply_rows(conn, rows, "tester")
    assert second["inserted"] == 0
    assert second["skipped_present"] == 2
    assert len(_rows(conn, "sa-1")) == 2


def test_a_fact_repeated_in_the_battery_inserts_once(conn):
    _add_question(conn, "sa-1")
    counts = script.apply_rows(
        conn,
        [{"question_id": "sa-1", "fact_text": "A"}, {"question_id": "sa-1", "fact_text": "A"}],
        "tester",
    )
    assert counts["inserted"] == 1
    assert counts["skipped_present"] == 1


def test_build_rows_reads_bare_string_facts_in_file_order(tmp_path):
    battery = tmp_path / "b.yaml"
    battery.write_text(
        "- id: q1\n"
        "  question: Q?\n"
        "  required_facts:\n"
        "  - first\n"
        "  - '  second  '\n"
        "  - ''\n"
        "- id: q2\n"
        "  question: Q2?\n"
    )
    rows = script.build_rows(battery)
    assert rows == [
        {"question_id": "q1", "fact_text": "first"},
        {"question_id": "q1", "fact_text": "second"},
    ]


def test_build_rows_rejects_authored_fact_id_batteries(tmp_path):
    """Those belong to insert-multiturn-facts.py, which keys on the id."""
    battery = tmp_path / "b.yaml"
    battery.write_text(
        "- id: q1\n  question: Q?\n  required_facts:\n  - fact_id: F1\n    fact_text: x\n"
    )
    with pytest.raises(SystemExit, match="insert-multiturn-facts"):
        script.build_rows(battery)
