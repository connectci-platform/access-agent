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


def _seeded_db(
    tmp_path,
    rows: list[dict],
    *,
    retraction_columns: bool = True,
    kind_columns: bool = False,
) -> EvalDB:
    """A db whose engine has a persistent `reporting` schema seeded with rows.

    Uses a single shared in-memory attached db via a static pool so every
    connection from the engine sees the same reporting.question_facts.

    ``retraction_columns=False`` reproduces the pre-migration dashboard schema, where
    reporting.question_facts has no retracted_by/retracted_at yet.
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

    retraction_ddl = ",\n                    retracted_at TEXT" if retraction_columns else ""
    kind_ddl = ",\n                    fact_kind TEXT" if kind_columns else ""
    with db._session_factory() as session:
        session.execute(
            text(
                f"""
                CREATE TABLE reporting.question_facts (
                    fact_id       INTEGER NOT NULL,
                    version       INTEGER NOT NULL DEFAULT 1,
                    question_id   TEXT    NOT NULL,
                    display_order INTEGER NOT NULL DEFAULT 0,
                    fact_text     TEXT    NOT NULL,
                    status        TEXT    NOT NULL DEFAULT 'draft'{retraction_ddl}{kind_ddl},
                    PRIMARY KEY (fact_id, version)
                )
                """
            )
        )
        for r in rows:
            row = dict(r)
            retracted_at = row.pop("retracted_at", None)
            fact_kind = row.pop("fact_kind", None)
            cols = "fact_id, version, question_id, display_order, fact_text, status"
            vals = ":fact_id, :version, :question_id, :display_order, :fact_text, :status"
            if retraction_columns:
                cols += ", retracted_at"
                vals += ", :retracted_at"
                row["retracted_at"] = retracted_at
            if kind_columns:
                cols += ", fact_kind"
                vals += ", :fact_kind"
                row["fact_kind"] = fact_kind
            session.execute(
                text(f"INSERT INTO reporting.question_facts ({cols}) VALUES ({vals})"),
                row,
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


# --- retraction: a fact withdrawn in the dashboard must stop being scored ---


def test_excludes_retracted_facts(tmp_path):
    # The dashboard retracts by appending a version that keeps status and text and
    # stamps retracted_at. Status alone cannot distinguish it, so a scorer filtering
    # only on status would keep grading a withdrawn requirement.
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 1,
                "version": 1,
                "question_id": "q",
                "display_order": 1,
                "fact_text": "still required",
                "status": "confirmed",
            },
            {
                "fact_id": 2,
                "version": 1,
                "question_id": "q",
                "display_order": 2,
                "fact_text": "withdrawn requirement",
                "status": "confirmed",
            },
            {
                "fact_id": 2,
                "version": 2,
                "question_id": "q",
                "display_order": 2,
                "fact_text": "withdrawn requirement",
                "status": "confirmed",
                "retracted_at": "2026-09-11T12:00:00Z",
            },
        ],
    )
    assert load_question_facts(db, "q") == [{"fact_id": 1, "fact_text": "still required"}]


def test_retraction_does_not_resurrect_previous_version(tmp_path):
    # Guards the filter's placement: applied before latest-version selection, the
    # retracting version would be skipped and v1 would surface as "latest", silently
    # un-retracting the fact.
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 7,
                "version": 1,
                "question_id": "q",
                "display_order": 0,
                "fact_text": "original text",
                "status": "confirmed",
            },
            {
                "fact_id": 7,
                "version": 2,
                "question_id": "q",
                "display_order": 0,
                "fact_text": "original text",
                "status": "confirmed",
                "retracted_at": "2026-09-11T12:00:00Z",
            },
        ],
    )
    assert load_question_facts(db, "q") is None


def test_retraction_of_only_fact_falls_back_to_yaml(tmp_path):
    # Retracting every fact leaves no DB rows, which the loader reports as None —
    # so resolve_required_facts falls back to YAML rather than scoring zero facts.
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 9,
                "version": 1,
                "question_id": "q",
                "display_order": 0,
                "fact_text": "gone",
                "status": "confirmed",
                "retracted_at": "2026-09-11T12:00:00Z",
            }
        ],
    )
    assert resolve_required_facts(db, "q", ["yaml fact"]) == ["yaml fact"]


def test_works_against_schema_without_retraction_columns(tmp_path):
    # The dashboard added the retraction columns after this table shipped, and the two
    # services deploy independently. Against the older schema the query must still run
    # (flag filter only) rather than raise and silently fall back to stale YAML.
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 1,
                "version": 1,
                "question_id": "q",
                "display_order": 0,
                "fact_text": "pre-migration fact",
                "status": "confirmed",
            }
        ],
        retraction_columns=False,
    )
    assert load_question_facts(db, "q") == [{"fact_id": 1, "fact_text": "pre-migration fact"}]


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


def test_fact_kind_is_surfaced_when_the_column_exists(tmp_path):
    """fact_kind reaches the returned dict so eval_scores.context carries it.

    The drift classifier attributes a verdict flip using this value; without it
    a world fact going stale is indistinguishable from a regression.
    """
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 40,
                "version": 1,
                "question_id": "q1",
                "display_order": 0,
                "fact_text": "Anvil outage postponed as of July 2",
                "status": "draft",
                "fact_kind": "snapshot",
            },
            {
                "fact_id": 41,
                "version": 1,
                "question_id": "q1",
                "display_order": 1,
                "fact_text": "Answer does not invent a firm date",
                "status": "draft",
                "fact_kind": None,
            },
        ],
        kind_columns=True,
    )

    facts = load_question_facts(db, "q1")

    assert facts is not None
    assert facts[0]["fact_kind"] == "snapshot"
    # An untyped fact keeps the pre-fact_kind dict shape rather than carrying None,
    # so nothing downstream has to special-case a null kind.
    assert "fact_kind" not in facts[1]


def test_fact_kind_absent_from_schema_is_not_referenced(tmp_path):
    """An agent ahead of the dashboard must not raise and fall back to YAML."""
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 1,
                "version": 1,
                "question_id": "q1",
                "display_order": 0,
                "fact_text": "f",
                "status": "draft",
            }
        ],
        kind_columns=False,
    )

    facts = load_question_facts(db, "q1")

    assert facts == [{"fact_id": 1, "fact_text": "f"}]


def test_fact_kind_present_without_retraction_columns(tmp_path):
    """The two columns are probed independently, so either migration order works.

    The dashboard added retracted_at and fact_kind in separate migrations. This
    covers the ordering the other tests miss (kind present, retraction absent),
    which the module docstring claims to support.
    """
    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 5,
                "version": 1,
                "question_id": "q1",
                "display_order": 0,
                "fact_text": "f",
                "status": "draft",
                "fact_kind": "world",
            }
        ],
        retraction_columns=False,
        kind_columns=True,
    )

    facts = load_question_facts(db, "q1")

    assert facts == [{"fact_id": 5, "fact_text": "f", "fact_kind": "world"}]


def test_column_probe_failure_is_not_cached(tmp_path, monkeypatch):
    """A transient probe error must not disable the retraction filter for the run.

    Caching an empty column set would drop the retraction filter on every
    subsequent question, silently scoring retracted facts.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from src.eval import question_facts as qf

    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 6,
                "version": 1,
                "question_id": "q1",
                "display_order": 0,
                "fact_text": "f",
                "status": "draft",
            }
        ],
    )
    qf._FACT_COLUMNS_CACHE.clear()

    calls = {"n": 0}
    real_inspect = qf.inspect

    def flaky_inspect(engine):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SQLAlchemyError("transient")
        return real_inspect(engine)

    monkeypatch.setattr(qf, "inspect", flaky_inspect)

    assert qf._fact_columns(db) == set()  # first probe fails
    assert "retracted_at" in qf._fact_columns(db)  # retried, not cached
    assert calls["n"] == 2


def test_column_probe_is_cached_per_engine(tmp_path, monkeypatch):
    """The probe runs once per engine, not once per question.

    load_question_facts is called for every battery question (~50 a run) and the
    schema cannot change mid-run, so a second call must not re-inspect.
    """
    from src.eval import question_facts as qf

    db = _seeded_db(
        tmp_path,
        [
            {
                "fact_id": 7,
                "version": 1,
                "question_id": "q1",
                "display_order": 0,
                "fact_text": "f",
                "status": "draft",
            }
        ],
    )
    qf._FACT_COLUMNS_CACHE.clear()

    calls = {"n": 0}
    real_inspect = qf.inspect

    def counting_inspect(engine):
        calls["n"] += 1
        return real_inspect(engine)

    monkeypatch.setattr(qf, "inspect", counting_inspect)

    first = qf._fact_columns(db)
    second = qf._fact_columns(db)

    assert "retracted_at" in first
    assert second is first  # same cached object, not a fresh probe
    assert calls["n"] == 1
