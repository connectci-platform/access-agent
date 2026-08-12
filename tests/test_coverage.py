"""Unit tests for the coverage audit — fixture strings, no DB."""

from __future__ import annotations

import json

from src.eval.coverage import (
    build_coverage,
    parse_tool_result_calls,
    reconcile_warnings,
    structural_class,
)


class TestStructuralClass:
    def test_write_by_registry(self):
        assert structural_class("register_for_event") == "write"
        assert structural_class("cancel_registration") == "write"
        assert structural_class("create_announcement") == "write"

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
