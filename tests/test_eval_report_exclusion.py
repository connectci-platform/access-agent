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
