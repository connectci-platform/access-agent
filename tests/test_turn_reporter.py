from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.turn_reporter import ReportToolCall, TurnReport, TurnReportBase


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
