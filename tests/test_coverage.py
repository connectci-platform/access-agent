"""Unit tests for the coverage audit — fixture strings, no DB."""

from __future__ import annotations

import json

from src.eval.coverage import (
    build_coverage,
    parse_tool_result_calls,
    reconcile_warnings,
    structural_class,
)
from src.eval.coverage_report import render_coverage


class TestStructuralClass:
    def test_write_by_registry(self):
        assert structural_class("register_for_event") == "write"
        assert structural_class("cancel_registration") == "write"
        assert structural_class("create_announcement") == "write"
        # events-organizer writes whose names do NOT match the write-prefix regex
        # (create_/update_/delete_/register_/cancel_/report_): classified as writes
        # only because they're in WRITE_MCP_TOOL_NAMES, proving registry membership
        # is doing the work, not the prefix fallback.
        assert structural_class("restore_event") == "write"
        assert structural_class("send_for_review") == "write"
        assert structural_class("add_occurrence") == "write"

    def test_write_by_prefix_fallback(self):
        # a plausible new write tool not yet in WRITE_MCP_TOOL_NAMES
        assert structural_class("delete_something_new") == "write"

    def test_auth_read(self):
        assert structural_class("get_my_registrations") == "auth-read"
        assert structural_class("get_my_events") == "auth-read"

    def test_auth_read_by_explicit_set(self):
        # acting-user-scoped read that does NOT match the get_my_ prefix
        assert structural_class("get_rp_account") == "auth-read"

    def test_unauth_read(self):
        assert structural_class("search_events") == "unauth-read"
        assert structural_class("get_infrastructure_news") == "unauth-read"

    def test_compositional(self):
        # XDMoD plumbing + authoring helpers: read-only but never user-facing
        assert structural_class("get_dimension_values") == "composition"
        assert structural_class("suggest_tags") == "composition"
        assert structural_class("get_ticket_types") == "composition"


class TestParseToolResultCalls:
    def test_single_call(self):
        tr = "### Tool call: search_events\n- arguments: " + json.dumps(
            {"date": "upcoming", "tags": "gpu"}, sort_keys=True
        )
        calls = parse_tool_result_calls(tr)
        assert calls == [("search_events", {"date": "upcoming", "tags": "gpu"})]

    def test_multi_call_joined(self):
        a = "### Tool call: search_events\n- arguments: " + json.dumps({"date": "upcoming"})
        b = "### Tool call: get_event\n- arguments: " + json.dumps({"id": "9150"})
        calls = parse_tool_result_calls(a + "\n\n---\n\n" + b)
        assert [c[0] for c in calls] == ["search_events", "get_event"]

    def test_payload_cannot_spoof_a_block(self):
        # a tool whose data payload itself contains a `---` line and a `### ` line
        poison = (
            "### Tool call: search_docs\n"
            "- arguments: " + json.dumps({"query": "x"}) + "\n"
            "- data: some text\n---\n### Not A Real Call\nmore text"
        )
        calls = parse_tool_result_calls(poison)
        # only the real header+arguments block is a call; the payload noise is not
        assert calls == [("search_docs", {"query": "x"})]

    def test_bad_json_args_degrade_to_empty(self):
        tr = "### Tool call: search_events\n- arguments: {not valid json"
        assert parse_tool_result_calls(tr) == [("search_events", {})]

    def test_empty(self):
        assert parse_tool_result_calls(None) == []
        assert parse_tool_result_calls("") == []

    def test_header_without_arguments_line(self):
        # a header block with no following '- arguments:' line -> call with empty args
        assert parse_tool_result_calls("### Tool call: lonely_tool\nsome prose") == [
            ("lonely_tool", {})
        ]

    def test_block_with_no_header(self):
        # a block that is pure prose (no '### Tool call:') contributes no call
        assert parse_tool_result_calls("just some text\nno header here") == []

    def test_args_is_json_but_not_object(self):
        # arguments line parses as JSON but is a list, not a dict -> empty args
        assert parse_tool_result_calls("### Tool call: t\n- arguments: [1, 2]") == [("t", {})]


def _score(qid, tools_called, tool_results, n_facts, n_pass=0):
    verdicts = [{"id": str(i), "verdict": "yes"} for i in range(n_pass)]
    return {
        "question_id": qid,
        "context": {
            "node_trace": [{"node": "loop", "tools_called": tools_called}],
            "tool_results": tool_results,
            "required_facts": [f"fact{i}" for i in range(n_facts)],
            "fact_verdicts": verdicts,
        },
    }


class TestBuildCoverage:
    def test_invocation_vs_total_calls(self):
        # one question, same tool called twice: invocations=1, total_calls=2
        tr = (
            "### Tool call: search_events\n- arguments: "
            + json.dumps({"date": "upcoming"})
            + "\n\n---\n\n"
            + "### Tool call: search_events\n- arguments: "
            + json.dumps({"date": "past"})
        )
        cov = {
            c.tool: c
            for c in build_coverage(["search_events"], [_score("q1", ["search_events"], tr, 3)])
        }
        c = cov["search_events"]
        assert c.invocations == 1  # node_trace dedups
        assert c.total_calls == 2  # tool_results does not
        assert c.distinct_arg_shapes == 1  # both calls have key-set {date}

    def test_exercised_not_evaluated(self):
        # invoked, but the question has zero facts -> gate blind spot
        tr = "### Tool call: get_my_events\n- arguments: {}"
        cov = {
            c.tool: c
            for c in build_coverage(["get_my_events"], [_score("q1", ["get_my_events"], tr, 0)])
        }
        assert cov["get_my_events"].tier == "EXERCISED-NOT-EVALUATED"

    def test_uncovered_structural_write(self):
        # served, never invoked, and a write tool -> structural, not negligence
        cov = {c.tool: c for c in build_coverage(["register_for_event"], [])}
        assert cov["register_for_event"].tier == "UNCOVERED-STRUCTURAL"

    def test_uncovered_testable_read(self):
        # served, never invoked, plain read tool -> actionable gap
        cov = {c.tool: c for c in build_coverage(["search_new_thing"], [])}
        assert cov["search_new_thing"].tier == "UNCOVERED-TESTABLE"

    def test_uncovered_compositional(self):
        # served, never invoked, but a helper tool -> not an actionable gap
        cov = {c.tool: c for c in build_coverage(["get_dimension_values"], [])}
        assert cov["get_dimension_values"].tier == "UNCOVERED-COMPOSITIONAL"

    def test_compositional_still_evaluated_when_exercised(self):
        # a helper invoked via a parent capability + checked -> EVALUATED, not demoted
        tr = "### Tool call: get_dimension_values\n- arguments: {}"
        cov = {
            c.tool: c
            for c in build_coverage(
                ["get_dimension_values"],
                [_score("q1", ["get_dimension_values"], tr, 2, 2)],
            )
        }
        assert cov["get_dimension_values"].tier == "EVALUATED"

    def test_evaluated(self):
        tr = "### Tool call: search_events\n- arguments: " + json.dumps({"date": "upcoming"})
        cov = {
            c.tool: c
            for c in build_coverage(["search_events"], [_score("q1", ["search_events"], tr, 3, 3)])
        }
        assert cov["search_events"].tier == "EVALUATED"

    def test_facts_realized_is_max_not_sum(self):
        # two questions invoke the tool with 2 and 3 facts -> upper bound 3, never 5
        tr = "### Tool call: search_events\n- arguments: {}"
        scores = [_score("q1", ["search_events"], tr, 2), _score("q2", ["search_events"], tr, 3)]
        cov = {c.tool: c for c in build_coverage(["search_events"], scores)}
        assert cov["search_events"].facts_realized == 3
        assert cov["search_events"].invocations == 2

    def test_non_enum_arg_key_ignored_for_enum_values(self):
        # a non-enum-like arg key (e.g. 'query') is tracked in arg_shapes but not enum_values
        tr = "### Tool call: search_events\n- arguments: " + json.dumps({"query": "gpu training"})
        cov = {
            c.tool: c
            for c in build_coverage(["search_events"], [_score("q1", ["search_events"], tr, 1)])
        }
        c = cov["search_events"]
        assert c.enum_values == {}
        assert frozenset({"query"}) in c.arg_shapes

    def test_invoked_but_not_in_snapshot_flagged(self):
        tr = "### Tool call: brand_new_tool\n- arguments: {}"
        cov = {c.tool: c for c in build_coverage([], [_score("q1", ["brand_new_tool"], tr, 1)])}
        assert cov["brand_new_tool"].in_snapshot is False

    def test_enum_values_collected(self):
        a = "### Tool call: get_infrastructure_news\n- arguments: " + json.dumps(
            {"time": "current"}
        )
        b = "### Tool call: get_infrastructure_news\n- arguments: " + json.dumps(
            {"time": "scheduled"}
        )
        tr = a + "\n\n---\n\n" + b
        cov = {
            c.tool: c
            for c in build_coverage(
                ["get_infrastructure_news"], [_score("q1", ["get_infrastructure_news"], tr, 1)]
            )
        }
        assert cov["get_infrastructure_news"].enum_values["time"] == {"current", "scheduled"}


class TestReconcile:
    def test_multi_call_same_name_no_warning(self):
        # counts differ (2 blocks vs 1 deduped name) but name-sets match -> silent
        tr = (
            "### Tool call: search_events\n- arguments: {}"
            + "\n\n---\n\n"
            + "### Tool call: search_events\n- arguments: {}"
        )
        assert reconcile_warnings([_score("q1", ["search_events"], tr, 1)]) == []

    def test_name_set_drift_warns(self):
        tr = "### Tool call: get_event\n- arguments: {}"
        w = reconcile_warnings([_score("q1", ["search_events"], tr, 1)])
        assert len(w) == 1 and "drift" in w[0]


class TestNodeTraceEdgeCases:
    def test_empty_node_trace_yields_no_tools(self):
        # node_trace None/empty -> the tool is served but never invoked
        cov = {c.tool: c for c in build_coverage(["search_events"], [_score("q1", [], "", 2)])}
        assert cov["search_events"].invocations == 0
        assert cov["search_events"].tier == "UNCOVERED-TESTABLE"

    def test_node_trace_as_json_string(self):
        # context.node_trace can arrive as a JSON string instead of a list
        score = {
            "question_id": "q1",
            "context": {
                "node_trace": json.dumps([{"node": "loop", "tools_called": ["search_events"]}]),
                "tool_results": "### Tool call: search_events\n- arguments: {}",
                "required_facts": ["f0", "f1"],
                "fact_verdicts": [],
            },
        }
        cov = {c.tool: c for c in build_coverage(["search_events"], [score])}
        assert cov["search_events"].invocations == 1

    def test_node_trace_key_absent(self):
        # context with no node_trace key at all -> _extract_invoked_tools sees None
        score = {
            "question_id": "q1",
            "context": {
                "tool_results": "",
                "required_facts": [],
                "fact_verdicts": [],
            },
        }
        cov = {c.tool: c for c in build_coverage(["search_events"], [score])}
        assert cov["search_events"].invocations == 0

    def test_node_trace_entry_not_a_dict(self):
        # a node_trace list whose element is not a dict is skipped, not fatal
        score = {
            "question_id": "q1",
            "context": {
                "node_trace": ["not-a-dict", {"tools_called": ["search_events"]}],
                "tool_results": "",
                "required_facts": ["f0"],
                "fact_verdicts": [],
            },
        }
        cov = {c.tool: c for c in build_coverage(["search_events"], [score])}
        assert cov["search_events"].invocations == 1

    def test_bad_json_node_trace_degrades(self):
        score = {
            "question_id": "q1",
            "context": {
                "node_trace": "{not valid json",
                "tool_results": "",
                "required_facts": [],
                "fact_verdicts": [],
            },
        }
        # served tool stays uninvoked; the bad trace does not raise
        cov = {c.tool: c for c in build_coverage(["search_events"], [score])}
        assert cov["search_events"].invocations == 0

    def test_call_block_name_not_in_trace_or_snapshot(self):
        # a tool present only in the tool_results call blocks (not node_trace, not served)
        score = {
            "question_id": "q1",
            "context": {
                "node_trace": [{"node": "loop", "tools_called": []}],
                "tool_results": "### Tool call: surprise_tool\n- arguments: {}",
                "required_facts": [],
                "fact_verdicts": [],
            },
        }
        cov = {c.tool: c for c in build_coverage([], [score])}
        assert cov["surprise_tool"].total_calls == 1
        assert cov["surprise_tool"].in_snapshot is False


class TestRenderCoverage:
    def _cov(self):
        return build_coverage(
            ["search_events", "register_for_event", "get_dimension_values", "search_new_thing"],
            [
                _score(
                    "q1",
                    ["search_events"],
                    "### Tool call: search_events\n- arguments: "
                    + json.dumps({"date": "upcoming"}),
                    2,
                    2,
                )
            ],
        )

    def _render(self, fmt):
        return render_coverage(
            self._cov(),
            run_id="run-x",
            battery="b.yaml",
            model="m",
            warnings=["w1"],
            fmt=fmt,
        )

    def test_table_groups_by_tier(self):
        out = self._render("table")
        assert "run-x" in out
        assert "[EVALUATED]" in out
        assert "[UNCOVERED-STRUCTURAL]" in out
        assert "[UNCOVERED-COMPOSITIONAL]" in out
        assert "search_events" in out
        assert "parser warnings (1)" in out

    def test_md_has_pipe_tables(self):
        out = self._render("md")
        assert "## EVALUATED" in out
        assert "| tool |" in out

    def test_csv_is_parseable(self):
        import csv
        import io

        out = self._render("csv")
        # first line is the summary comment; the rest is CSV
        body = "\n".join(out.splitlines()[1:])
        rows = list(csv.DictReader(io.StringIO(body)))
        assert any(r["tool"] == "search_events" for r in rows)
        assert {"tier", "invocations", "facts_realized_ub"} <= set(rows[0].keys())

    def test_json_shape(self):
        out = self._render("json")
        data = json.loads(out)
        assert data["warnings"] == ["w1"]
        assert "summary" in data
        assert any(t["tool"] == "search_events" for t in data["tools"])

    def test_csv_empty_coverage(self):
        out = render_coverage([], run_id="r", battery=None, model=None, warnings=[], fmt="csv")
        assert "0 served" in out

    def test_not_in_snapshot_marker(self):
        cov = build_coverage(
            [], [_score("q1", ["brand_new"], "### Tool call: brand_new\n- arguments: {}", 1)]
        )
        out = render_coverage(cov, run_id="r", battery=None, model=None, warnings=[], fmt="table")
        assert "NOT-IN-SNAPSHOT" in out
