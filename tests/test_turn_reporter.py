from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.turn_reporter import (
    ReportToolCall,
    TurnReport,
    TurnReportBase,
    TurnReporter,
    _assemble_turn_report,
)


class TestTurnReportModels:
    def setup_method(self):
        self.engine = create_engine("sqlite:///:memory:")
        TurnReportBase.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def test_tables_and_key_columns_exist(self):
        cols = {c["name"] for c in inspect(self.engine).get_columns("turn_reports")}
        assert {
            "id",
            "session_id",
            "turn_index",
            "origin",
            "agent_version",
            "capabilities",
            "tool_count",
            "any_tool_failed",
            "invoked_write",
            "payload",
        }.issubset(cols)
        child = {c["name"] for c in inspect(self.engine).get_columns("report_tool_calls")}
        assert {
            "id",
            "report_id",
            "step_index",
            "tool_name",
            "server",
            "success",
            "args_hash",
            "arguments",
            "duration_ms",
        }.issubset(child)

    def test_a2_columns_exist(self):
        cols = {c["name"] for c in inspect(self.engine).get_columns("turn_reports")}
        assert {
            "rag_chunk_count",
            "rag_zero_hits",
            "total_tokens",
            "citation_count",
            "summarized",
            "query_intent",
            "refused",
            "is_deflection",
        }.issubset(cols)

    def test_parent_child_insert(self):
        s = self.Session()
        r = TurnReport(session_id="s1", turn_index=1, query_text="hi", origin="real")
        s.add(r)
        s.flush()
        s.add(
            ReportToolCall(
                report_id=r.id,
                step_index=0,
                tool_name="search_software",
                server="software-discovery",
                success=True,
                args_hash="abc",
            )
        )
        s.commit()
        assert s.query(ReportToolCall).filter_by(report_id=r.id).count() == 1
        s.close()


def _tool_result(name, server, success=True, args=None):
    return {
        "step_id": "x",
        "tool_name": name,
        "server": server,
        "success": success,
        "data": "ok",
        "error": None,
        "arguments": args or {},
        "duration_ms": 0,
    }


class TestAssemble:
    def test_derived_columns_and_children(self):
        final_state = {
            "final_answer": "Delta has Python.",
            "tools_used": ["software-discovery__search_software", "jsm__create_support_ticket"],
            "tool_results": [
                _tool_result("search_software", "software-discovery", True, {"q": "python"}),
                _tool_result("create_support_ticket", "jsm", False, {"summary": "x"}),
            ],
            "node_trace": [{"node": "loop"}],
            "resource_context": "delta",
        }
        report, tool_calls = _assemble_turn_report(
            final_state=final_state,
            session_id="s1",
            turn_index=2,
            question_id="q1",
            query_text="is python on delta?",
            duration_ms=1234.5,
            acting_user="user@x.edu",
            success=True,
            capabilities=["search_software", "open_ticket"],
        )
        assert report["tool_count"] == 2
        assert report["tool_failure_count"] == 1
        assert report["any_tool_failed"] is True
        assert report["invoked_write"] is True
        assert report["was_authenticated"] is True
        assert report["resource_context"] == "delta"
        assert report["capabilities"] == ["search_software", "open_ticket"]
        assert report["payload"]["answer"] == "Delta has Python."
        assert len(tool_calls) == 2
        assert tool_calls[0]["step_index"] == 0
        assert tool_calls[1]["tool_name"] == "create_support_ticket"
        assert tool_calls[0]["args_hash"]

    def test_total_tokens_from_final_state(self):
        report, _ = _assemble_turn_report(
            final_state={"tools_used": [], "tool_results": [], "total_tokens": 123},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hi",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert report["total_tokens"] == 123

    def test_total_tokens_absent_is_none_not_zero(self):
        # NULL ("not measured") must stay distinct from a measured 0.
        report, _ = _assemble_turn_report(
            final_state={"tools_used": [], "tool_results": []},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hi",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert report["total_tokens"] is None

    def test_anonymous_user_not_authenticated(self):
        report, _ = _assemble_turn_report(
            final_state={"tools_used": [], "tool_results": []},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hi",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert report["was_authenticated"] is False
        assert report["user_hash"] is None
        assert report["invoked_write"] is False


class TestWrite:
    def _reporter(self):
        r = TurnReporter()
        r._engine = create_engine("sqlite:///:memory:")
        TurnReportBase.metadata.create_all(r._engine)
        r._session_factory = sessionmaker(bind=r._engine)
        r._initialized = True
        return r

    def test_write_persists_parent_and_children(self):
        r = self._reporter()
        final_state = {
            "final_answer": "ok",
            "tools_used": ["jsm__create_support_ticket"],
            "tool_results": [
                {
                    "tool_name": "create_support_ticket",
                    "server": "jsm",
                    "success": True,
                    "arguments": {"summary": "x"},
                    "duration_ms": 0,
                }
            ],
        }
        r.log_turn_report(
            final_state=final_state,
            session_id="s1",
            turn_index=1,
            question_id="q1",
            query_text="open a ticket",
            duration_ms=10.0,
            acting_user="u@x.edu",
            success=True,
            capabilities=["open_ticket"],
        )
        s = r._session_factory()
        row = s.query(TurnReport).one()
        assert row.invoked_write is True
        assert s.query(ReportToolCall).filter_by(report_id=row.id).count() == 1
        s.close()

    def test_count_turns_for_session(self):
        r = self._reporter()
        assert r.count_turns_for_session("sX") == 0
        for _ in range(2):
            r.log_turn_report(
                final_state={"tools_used": [], "tool_results": []},
                session_id="sX",
                turn_index=1,
                question_id="q",
                query_text="hi",
                duration_ms=1.0,
                acting_user=None,
                success=True,
                capabilities=[],
            )
        assert r.count_turns_for_session("sX") == 2
        assert r.count_turns_for_session("other") == 0

    def test_write_never_raises_on_bad_state(self):
        r = self._reporter()
        r.log_turn_report(
            final_state={},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="x",
            duration_ms=None,
            acting_user=None,
            success=True,
            capabilities=[],
        )
