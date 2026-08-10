"""Dashboard aggregate excludes multiturn runs — fail-closed on missing runs."""

from src.eval.db import EvalDB


def _score_kwargs(run_id, qid="q1"):
    return {
        "run_id": run_id,
        "question_id": qid,
        "source": "judge",
        "question_text": "q",
        "answer_text": "a",
        "answerable": True,
        "correctness": 2,
        "specificity": 2,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
        "composite_score": 0.9,
    }


def test_multiturn_runs_excluded(tmp_path):
    from src.eval.report_data import build_report_data

    db = EvalDB(f"sqlite:///{tmp_path}/e.db")
    st = db.create_run(run_type="pre_production", metadata_={"system": "agent_full"})
    mt = db.create_run(run_type="pre_production", metadata_={"mode": "multiturn"})
    db.add_score(**_score_kwargs(str(st.id), "single_q"))
    db.add_score(**_score_kwargs(str(mt.id), "mt-f-01_t1"))

    data = build_report_data(db, since="7d")
    # build_report_data returns aggregates, not raw rows: total_scored + worst_answers
    # (capped top-10; with 2 candidate rows, every kept row appears there).
    assert data["total_scored"] == 1
    assert {w["question_id"] for w in data["worst_answers"]} == {"single_q"}


def test_missing_run_excluded_fail_closed(tmp_path):
    from src.eval.report_data import build_report_data

    db = EvalDB(f"sqlite:///{tmp_path}/e.db")
    ok = db.create_run(run_type="pre_production")
    db.add_score(**_score_kwargs(str(ok.id), "kept"))
    db.add_score(**_score_kwargs("no-such-run", "orphan"))  # SQLite doesn't enforce the FK

    data = build_report_data(db, since="7d")
    assert data["total_scored"] == 1
    assert {w["question_id"] for w in data["worst_answers"]} == {"kept"}


def test_missing_run_lookup_memoized_across_multiple_scores(tmp_path, caplog):
    """Two scores referencing the same orphaned run_id trigger exactly one db.get_run
    call and one warning, not one per score (review finding: _index_runs must memoize
    misses, not just hits)."""
    import logging

    from src.eval.report_data import build_report_data

    db = EvalDB(f"sqlite:///{tmp_path}/e.db")
    ok = db.create_run(run_type="pre_production")
    db.add_score(**_score_kwargs(str(ok.id), "kept"))
    db.add_score(**_score_kwargs("no-such-run", "orphan1"))
    db.add_score(**_score_kwargs("no-such-run", "orphan2"))

    calls = {"n": 0}
    real_get_run = db.get_run

    def counting_get_run(run_id):
        calls["n"] += 1
        return real_get_run(run_id)

    db.get_run = counting_get_run  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="src.eval.report_data"):
        data = build_report_data(db, since="7d")

    assert data["total_scored"] == 1
    assert {w["question_id"] for w in data["worst_answers"]} == {"kept"}
    # One get_run call for "ok" (found) + one for "no-such-run" (missing) = 2, not 3.
    assert calls["n"] == 2
    missing_warnings = [r for r in caplog.records if "no-such-run" in r.message]
    assert len(missing_warnings) == 1
