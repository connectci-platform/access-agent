"""Unit tests for the judge-input formatting helpers shared by the single- and
multi-turn runners (moved verbatim from runner.py in ef7f854)."""

from src.eval.formatting import format_node_trace, format_rag_matches, format_tool_results


class _RagMatch:
    """Stand-in for a RAGMatch-like object: has .answer, may or may not have .score."""

    def __init__(self, answer, score=None):
        self.answer = answer
        if score is not None:
            self.score = score


class _ToolResult:
    """Stand-in for a ToolResult-like object (attribute access, not dict)."""

    def __init__(
        self,
        tool_name="unknown",
        arguments=None,
        success=True,
        duration_ms=0,
        data=None,
        error=None,
    ):
        self.tool_name = tool_name
        self.arguments = arguments
        self.success = success
        self.duration_ms = duration_ms
        self.data = data
        self.error = error


# --- format_rag_matches -----------------------------------------------------


def test_format_rag_matches_returns_none_when_missing():
    assert format_rag_matches({}) is None


def test_format_rag_matches_returns_none_when_empty():
    assert format_rag_matches({"rag_matches": []}) is None


def test_format_rag_matches_formats_objects_with_answer_and_score():
    matches = [_RagMatch("The answer.", score=0.83)]
    out = format_rag_matches({"rag_matches": matches})
    assert out == "Score: 0.83\nThe answer."


def test_format_rag_matches_object_without_score_falls_back_to_na():
    matches = [_RagMatch("The answer.")]
    out = format_rag_matches({"rag_matches": matches})
    assert out == "Score: N/A\nThe answer."


def test_format_rag_matches_formats_dicts():
    matches = [{"answer": "Dict answer.", "score": 0.5}]
    out = format_rag_matches({"rag_matches": matches})
    assert out == "Score: 0.5\nDict answer."


def test_format_rag_matches_dict_missing_keys_uses_defaults():
    out = format_rag_matches({"rag_matches": [{}]})
    assert out == "Score: N/A\n"


def test_format_rag_matches_joins_multiple_with_separator():
    matches = [
        _RagMatch("First.", score=0.9),
        {"answer": "Second.", "score": 0.4},
    ]
    out = format_rag_matches({"rag_matches": matches})
    assert out == "Score: 0.9\nFirst.\n---\nScore: 0.4\nSecond."


def test_format_rag_matches_skips_entries_that_are_neither_object_nor_dict():
    # A bare string has no .answer and isn't a dict, so it contributes nothing.
    # If every entry is skipped, parts is empty and the function returns None.
    out = format_rag_matches({"rag_matches": ["not-a-match"]})
    assert out is None


# --- format_tool_results -----------------------------------------------------


def test_format_tool_results_returns_none_when_missing():
    assert format_tool_results({}) is None


def test_format_tool_results_returns_none_when_empty():
    assert format_tool_results({"tool_results": []}) is None


def test_format_tool_results_extracts_object_attributes():
    results = [
        _ToolResult(
            tool_name="search_resources",
            arguments={"q": "gpu"},
            success=True,
            duration_ms=42,
            data=["a", "b"],
        )
    ]
    out = format_tool_results({"tool_results": results})
    assert "### Tool call: search_resources" in out
    assert '"q": "gpu"' in out
    assert "- success: True" in out
    assert "- duration_ms: 42" in out
    assert "- result_count: 2" in out
    assert "- empty: False" in out
    assert "- error:" not in out
    assert "['a', 'b']" in out


def test_format_tool_results_extracts_dict_shape():
    results = [
        {
            "tool_name": "get_event",
            "arguments": {"id": 1},
            "success": False,
            "duration_ms": 7,
            "data": None,
            "error": "not found",
        }
    ]
    out = format_tool_results({"tool_results": results})
    assert "### Tool call: get_event" in out
    assert "- success: False" in out
    assert "- duration_ms: 7" in out
    assert "- result_count: 0" in out
    assert "- empty: True" in out
    assert "- error: not found" in out


def test_format_tool_results_bare_value_falls_back_to_unknown():
    # Neither hasattr(tool_name) nor a dict -> the catch-all branch.
    out = format_tool_results({"tool_results": ["just a string"]})
    assert "### Tool call: unknown" in out
    assert "- arguments: {}" in out
    assert "- success: True" in out
    assert "- result_count:" not in out  # data is a non-empty str -> count is None
    assert "- empty: False" in out
    assert "- data:" in out
    assert "just a string" in out


def test_format_tool_results_dict_data_with_items_key_counts_list():
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": {"items": [1, 2, 3]}}]})
    assert "- result_count: 3" in out
    assert "- empty: False" in out


def test_format_tool_results_dict_data_with_empty_items_list():
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": {"items": []}}]})
    assert "- result_count: 0" in out
    assert "- empty: True" in out


def test_format_tool_results_dict_data_with_total_key_counts_int():
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": {"total": 5}}]})
    assert "- result_count: 5" in out
    assert "- empty: False" in out


def test_format_tool_results_dict_data_with_zero_total_is_empty():
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": {"total_outages": 0}}]})
    assert "- result_count: 0" in out
    assert "- empty: True" in out


def test_format_tool_results_bare_empty_dict_data_is_empty_zero():
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": {}}]})
    assert "- result_count: 0" in out
    assert "- empty: True" in out


def test_format_tool_results_bare_nonempty_dict_data_is_indeterminate():
    # No known list/count key and non-empty -> count is None, not empty.
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": {"weird": "shape"}}]})
    assert "- result_count:" not in out
    assert "- empty: False" in out


def test_format_tool_results_string_data_empty_and_nonempty():
    empty_out = format_tool_results({"tool_results": [{"tool_name": "t", "data": "   "}]})
    assert "- result_count:" not in empty_out
    assert "- empty: True" in empty_out

    nonempty_out = format_tool_results({"tool_results": [{"tool_name": "t", "data": "hello"}]})
    assert "- result_count:" not in nonempty_out
    assert "- empty: False" in nonempty_out


def test_format_tool_results_none_data_is_empty_zero():
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": None}]})
    assert "- result_count: 0" in out
    assert "- empty: True" in out


def test_format_tool_results_data_of_other_type_is_indeterminate_not_empty():
    # data that is none of None/list/dict/str (e.g. an int) falls through to the
    # final catch-all: count indeterminate, not flagged empty.
    out = format_tool_results({"tool_results": [{"tool_name": "t", "data": 42}]})
    assert "- result_count:" not in out
    assert "- empty: False" in out


def test_format_tool_results_joins_multiple_calls_with_separator():
    results = [
        {"tool_name": "first", "data": None},
        {"tool_name": "second", "data": [1]},
    ]
    out = format_tool_results({"tool_results": results})
    assert "\n\n---\n\n" in out
    assert out.index("### Tool call: first") < out.index("### Tool call: second")


# --- format_node_trace -------------------------------------------------------


def test_format_node_trace_returns_none_when_missing():
    assert format_node_trace({}) is None


def test_format_node_trace_returns_none_when_empty():
    assert format_node_trace({"node_trace": []}) is None


def test_format_node_trace_renders_json():
    trace = [{"node": "tool_calling_loop", "step": 1}]
    out = format_node_trace({"node_trace": trace})
    assert out is not None
    assert '"node": "tool_calling_loop"' in out
    assert '"step": 1' in out
