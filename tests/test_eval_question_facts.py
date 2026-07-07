"""Tests for stable fact_id keying from reporting.question_facts (Task 12).

The eval reads required facts (with stable ids) from the dashboard-owned
reporting.question_facts table instead of minting positional F{n} ids from YAML.
Selection rule: latest version per fact_id, excluding any whose latest is flagged.
When the table/schema/question is absent, the loader returns None so the caller
falls back to the YAML battery's q.metadata["required_facts"].
"""

from sqlalchemy import text

from src.eval.db import EvalDB
from src.eval.question_facts import load_question_facts, resolve_required_facts


def _make_db(tmp_path) -> EvalDB:
    return EvalDB(f"sqlite:///{tmp_path}/facts.db")


def _seeded_db(tmp_path, rows: list[dict]) -> EvalDB:
    """A db whose engine has a persistent `reporting` schema seeded with rows.

    Uses a single shared in-memory attached db via a static pool so every
    connection from the engine sees the same reporting.question_facts.
    """
    from sqlalchemy import create_engine, event
    from sqlalchemy.pool import StaticPool

    from src.eval.models import EvalBase

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _attach(dbapi_conn, _):
        dbapi_conn.execute("ATTACH DATABASE ':memory:' AS reporting")

    EvalBase.metadata.create_all(engine)

    db = EvalDB.__new__(EvalDB)
    db._engine = engine
    from sqlalchemy.orm import sessionmaker

    db._session_factory = sessionmaker(bind=engine)

    with db._session_factory() as session:
        session.execute(
            text(
                """
                CREATE TABLE reporting.question_facts (
                    fact_id       INTEGER NOT NULL,
                    version       INTEGER NOT NULL DEFAULT 1,
                    question_id   TEXT    NOT NULL,
                    display_order INTEGER NOT NULL DEFAULT 0,
                    fact_text     TEXT    NOT NULL,
                    status        TEXT    NOT NULL DEFAULT 'draft',
                    PRIMARY KEY (fact_id, version)
                )
                """
            )
        )
        for r in rows:
            session.execute(
                text(
                    "INSERT INTO reporting.question_facts "
                    "(fact_id, version, question_id, display_order, fact_text, status) "
                    "VALUES (:fact_id, :version, :question_id, :display_order, "
                    ":fact_text, :status)"
                ),
                r,
            )
        session.commit()
    return db


def test_returns_none_when_reporting_table_absent(tmp_path):
    # Local dev without the dashboard schema: must not crash, must fall back.
    db = _make_db(tmp_path)
    assert load_question_facts(db, "tc-software-02") is None


def test_returns_none_when_question_absent(tmp_path):
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 1,
                "version": 1,
                "question_id": "other-q",
                "display_order": 0,
                "fact_text": "x",
                "status": "confirmed",
            }
        ],
    )
    assert load_question_facts(db, "tc-software-02") is None


def test_loads_facts_ordered_by_display_order(tmp_path):
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 20,
                "version": 1,
                "question_id": "tc-software-02",
                "display_order": 1,
                "fact_text": "second",
                "status": "confirmed",
            },
            {
                "fact_id": 10,
                "version": 1,
                "question_id": "tc-software-02",
                "display_order": 0,
                "fact_text": "first",
                "status": "draft",
            },
        ],
    )
    facts = load_question_facts(db, "tc-software-02")
    assert facts == [
        {"fact_id": 10, "fact_text": "first"},
        {"fact_id": 20, "fact_text": "second"},
    ]


def test_selects_latest_version_and_excludes_flagged(tmp_path):
    # fact 100: latest version (v2) is confirmed -> included with v2 text.
    # fact 200: latest version (v3) is flagged -> excluded entirely, even though
    #           an earlier v1 was confirmed. (latest-non-flagged)
    # fact 300: latest version (v2) is draft -> included (draft is NOT excluded).
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 100,
                "version": 1,
                "question_id": "q",
                "display_order": 0,
                "fact_text": "v1 old text",
                "status": "confirmed",
            },
            {
                "fact_id": 100,
                "version": 2,
                "question_id": "q",
                "display_order": 0,
                "fact_text": "v2 latest text",
                "status": "confirmed",
            },
            {
                "fact_id": 200,
                "version": 1,
                "question_id": "q",
                "display_order": 1,
                "fact_text": "was confirmed once",
                "status": "confirmed",
            },
            {
                "fact_id": 200,
                "version": 3,
                "question_id": "q",
                "display_order": 1,
                "fact_text": "now flagged",
                "status": "flagged",
            },
            {
                "fact_id": 300,
                "version": 2,
                "question_id": "q",
                "display_order": 2,
                "fact_text": "draft latest",
                "status": "draft",
            },
        ],
    )
    facts = load_question_facts(db, "q")
    # fact 200 dropped (latest flagged); 100 uses v2 text; 300 draft kept.
    assert facts == [
        {"fact_id": 100, "fact_text": "v2 latest text"},
        {"fact_id": 300, "fact_text": "draft latest"},
    ]


# --- resolve_required_facts: prefer DB facts, fall back to YAML ---


def test_resolve_prefers_db_facts_over_yaml(tmp_path):
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 42,
                "version": 1,
                "question_id": "tc-software-02",
                "display_order": 0,
                "fact_text": "DB fact wins",
                "status": "confirmed",
            }
        ],
    )
    yaml_facts = ["stale yaml fact"]
    resolved = resolve_required_facts(db, "tc-software-02", yaml_facts)
    assert resolved == [{"fact_id": 42, "fact_text": "DB fact wins"}]


def test_resolve_falls_back_to_yaml_when_no_db_facts(tmp_path):
    # Table exists but has no rows for this question -> YAML fallback.
    db = _seeded_db(tmp_path, [])
    yaml_facts = ["yaml fallback fact"]
    resolved = resolve_required_facts(db, "tc-software-02", yaml_facts)
    assert resolved == ["yaml fallback fact"]


def test_resolve_falls_back_to_yaml_when_reporting_absent(tmp_path):
    # No reporting schema at all (local dev) -> YAML fallback, no crash.
    db = _make_db(tmp_path)
    yaml_facts = ["yaml fallback fact"]
    resolved = resolve_required_facts(db, "tc-software-02", yaml_facts)
    assert resolved == ["yaml fallback fact"]


def test_resolve_returns_none_when_neither_source(tmp_path):
    db = _make_db(tmp_path)
    assert resolve_required_facts(db, "tc-software-02", None) is None
