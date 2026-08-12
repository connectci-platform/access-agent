"""Tests for the `coverage` CLI handler — seeds a real SQLite EvalDB.

The handler builds EvalDB(settings.DATABASE_URL) internally, so we point
settings at a temp-file SQLite DB (an in-memory one would not survive the fresh
connection the handler opens) and seed a run + judge scores.
"""

from __future__ import annotations

import argparse

import pytest

from src.eval.db import EvalDB


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'eval.db'}"
    monkeypatch.setattr("src.config.settings.DATABASE_URL", url, raising=False)
    db = EvalDB(url)
    return db, url


def _args(**kw):
    ns = argparse.Namespace(run_id=None, battery=None, format="table")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _seed_run(db, run_id, system="agent_full", served=("search_events",)):
    db.create_run(
        id=run_id,
        run_type="pre_production",
        tool_catalog={"tool_count": len(served), "tools": list(served)},
        metadata_={"system": system},
    )


def _seed_score(db, run_id, qid="q1", tool="search_events"):
    db.add_score(
        run_id=run_id,
        question_id=qid,
        source="judge",
        context={
            "node_trace": [{"node": "loop", "tools_called": [tool]}],
            "tool_results": f"### Tool call: {tool}\n- arguments: {{}}",
            "required_facts": ["f0"],
            "fact_verdicts": [{"id": "0", "verdict": "yes"}],
        },
    )


def _run_handler(args):
    from src.eval.__main__ import _handle_coverage

    _handle_coverage(args)


class TestCoverageParser:
    """Cover the coverage subcommand's argparse wiring via build_parser()."""

    def _parser(self):
        from src.eval.__main__ import build_parser

        return build_parser()

    def test_parses_all_coverage_args(self):
        args = self._parser().parse_args(
            ["coverage", "--run-id", "loop-1", "--battery", "gapfill", "--format", "json"]
        )
        assert args.command == "coverage"
        assert args.run_id == "loop-1"
        assert args.battery == "gapfill"
        assert args.format == "json"

    def test_coverage_defaults(self):
        args = self._parser().parse_args(["coverage"])
        assert args.run_id is None
        assert args.battery is None
        assert args.format == "table"

    def test_format_choices_enforced(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["coverage", "--format", "pdf"])


class TestMainDispatch:
    def test_main_dispatches_to_handler(self, monkeypatch):
        import src.eval.__main__ as m

        called = {}
        monkeypatch.setattr(m.sys, "argv", ["prog", "coverage", "--run-id", "loop-1"])
        monkeypatch.setattr(
            m, "_handle_coverage", lambda args: called.setdefault("run_id", args.run_id)
        )
        m.main()
        assert called["run_id"] == "loop-1"

    def test_main_no_command_prints_help_and_exits(self, monkeypatch, capsys):
        import src.eval.__main__ as m

        monkeypatch.setattr(m.sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            m.main()


class TestHandleCoverage:
    def test_explicit_run_id_renders(self, seeded_db, capsys):
        db, _ = seeded_db
        _seed_run(db, "loop-1")
        _seed_score(db, "loop-1")
        _run_handler(_args(run_id="loop-1"))
        out = capsys.readouterr().out
        assert "loop-1" in out
        assert "search_events" in out

    def test_run_id_not_found(self, seeded_db, capsys):
        _run_handler(_args(run_id="nope"))
        assert "not found" in capsys.readouterr().out

    def test_raw_rag_run_rejected(self, seeded_db, capsys):
        db, _ = seeded_db
        _seed_run(db, "rag-1", system="raw_rag")
        _run_handler(_args(run_id="rag-1"))
        assert "not 'agent_full'" in capsys.readouterr().out

    def test_json_format(self, seeded_db, capsys):
        import json

        db, _ = seeded_db
        _seed_run(db, "loop-2")
        _seed_score(db, "loop-2")
        _run_handler(_args(run_id="loop-2", format="json"))
        data = json.loads(capsys.readouterr().out)
        assert "tools" in data and "summary" in data

    def test_run_with_empty_catalog_renders(self, seeded_db, capsys):
        # a run whose tool_catalog has no names still renders (all tools show
        # NOT-IN-SNAPSHOT via the invoked set); exercises the served=[] path
        db, _ = seeded_db
        db.create_run(
            id="loop-empty",
            run_type="pre_production",
            tool_catalog={"tool_count": 0, "tools": []},
            metadata_={"system": "agent_full"},
        )
        _seed_score(db, "loop-empty")
        _run_handler(_args(run_id="loop-empty"))
        out = capsys.readouterr().out
        assert "NOT-IN-SNAPSHOT" in out
