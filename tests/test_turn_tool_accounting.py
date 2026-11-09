"""Each turn's state describes that turn's own tool calls, under their own names.

Runs the real outer graph with a checkpointer, the real loop node and the real
middleware over a scripted chat model, as tests/test_eval_multiturn_compaction_real_stack.py
does. The only seams are ``get_llm`` and the tool list. Scripted tool calls reuse
the vLLM-style id ``chatcmpl-tool-0``, and AIMessage ids are left unset because
``add_messages`` replaces messages by id.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agent.domains.capabilities import get_capability_registry
from src.agent.graph import create_checkpointed_graph
from src.agent.nodes import tool_calling_loop as loop_mod
from src.agent.nodes.tool_calling_loop import _this_turn_messages
from src.agent.turn_capture import get_turn_capture, record_tool_timing, reset_turn_capture
from src.turn_reporter import _assemble_turn_report
from tests.agent_runtime import context_for, run_loop

REUSED_ID = "chatcmpl-tool-0"


def _call(name: str, n: int = 0, *, pad: str = "", call_id: str = REUSED_ID) -> dict[str, Any]:
    return {"name": name, "args": {"n": n, "pad": pad}, "id": call_id}


class _ScriptedModel(BaseChatModel):
    """Serves each turn's scripted responses in order, then a plain answer.

    ``script[turn]`` is a list of model responses, each a list of tool calls.
    For a turn in ``forever``, the last response repeats and it never answers.
    The middleware's summary request is answered with plain text.
    """

    script: dict[int, list[list[dict[str, Any]]]] = Field(default_factory=dict)
    turn: int = 0
    served: dict[int, int] = Field(default_factory=dict)
    forever: set[int] = Field(default_factory=set)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1] if messages else None
        if isinstance(last, HumanMessage) and "summar" in str(last.content).lower():
            msg = AIMessage(content=f"Summary through turn {self.turn}.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        responses = self.script.get(self.turn, [])
        i = self.served.get(self.turn, 0)
        self.served[self.turn] = i + 1
        if self.turn in self.forever and responses:
            i = min(i, len(responses) - 1)
        if i < len(responses):
            msg = AIMessage(content="", tool_calls=[dict(tc) for tc in responses[i]])
        else:
            msg = AIMessage(content=f"Answer for turn {self.turn}.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _ServerTool(StructuredTool):
    tool_server: str = ""


def _tool(name: str, server: str = "", *, timed: bool = True) -> StructuredTool:
    """A tool whose result names itself and its argument, so pairing is checkable."""

    def _run(n: int = 0, pad: str = "") -> str:
        if timed:
            record_tool_timing(name, 10 + n)
        return json.dumps({"tool": name, "n": n, "pad": pad})

    return _ServerTool.from_function(
        func=_run, name=name, description=f"{name} test tool", tool_server=server
    )


async def _run_thread(
    model: _ScriptedModel, tools: list[StructuredTool], turns: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = create_checkpointed_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "accounting-thread"}}
    states: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    with (
        patch.object(loop_mod, "get_llm", return_value=model),
        patch.object(loop_mod, "_build_prompt_and_tools", return_value=("system", tools, False)),
    ):
        messages: list = []
        for turn in range(1, turns + 1):
            model.turn = turn
            reset_turn_capture()
            messages = [*messages, HumanMessage(content=f"Question {turn}?")]
            state = await graph.ainvoke(
                {
                    "messages": messages,
                    "query": f"Question {turn}?",
                    "session_id": "s",
                    "question_id": f"q{turn}",
                    "tool_catalog": {"servers": []},
                },
                config,
                # Signed in with write access, so the gate lets write tools run.
                context=context_for("user@access-ci.org"),
            )
            messages = list(state["messages"])
            states.append(state)
            captures.append(dict(get_turn_capture()))
    return states, captures


def _results(state: dict[str, Any]) -> list[tuple[str, int]]:
    return [(r.tool_name, r.arguments["n"]) for r in state["tool_results"]]


def _report(state: dict[str, Any]) -> dict[str, Any]:
    report, _ = _assemble_turn_report(
        final_state=state,
        session_id="s",
        turn_index=None,
        question_id="q",
        query_text="q",
        duration_ms=None,
        acting_user="user@access-ci.org",
        success=True,
        capabilities=get_capability_registry().infer_capability_ids(state["tool_results"]),
    )
    return report


@pytest.mark.asyncio
async def test_a_reused_call_id_does_not_relabel_an_earlier_turns_result():
    model = _ScriptedModel(
        script={1: [[_call("search_events", 1)]], 2: [[_call("register_for_event", 2)]]}
    )
    tools = [_tool("search_events", "events"), _tool("register_for_event", "events")]

    states, _ = await _run_thread(model, tools, turns=2)

    assert _results(states[0]) == [("search_events", 1)]
    assert _results(states[1]) == [("register_for_event", 2)]
    assert states[1]["tools_used"] == ["register_for_event"]
    assert states[1]["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_a_reused_call_id_within_one_turn_pairs_each_result_with_its_own_call():
    model = _ScriptedModel(
        script={1: [[_call("search_events", 1)], [_call("register_for_event", 2)]]}
    )
    tools = [_tool("search_events", "events"), _tool("register_for_event", "events")]

    (state,), _ = await _run_thread(model, tools, turns=1)

    assert _results(state) == [("search_events", 1), ("register_for_event", 2)]
    for result in state["tool_results"]:
        assert result.data["tool"] == result.tool_name


@pytest.mark.parametrize("timed", [True, False], ids=["timings", "message-scan"])
@pytest.mark.asyncio
async def test_a_turn_with_no_tool_calls_reports_none(timed):
    model = _ScriptedModel(
        script={1: [[_call("search_events", 1)]], 2: [[_call("register_for_event", 2)]]}
    )
    tools = [
        _tool("search_events", "events", timed=timed),
        _tool("register_for_event", "events", timed=timed),
    ]

    states, _ = await _run_thread(model, tools, turns=3)

    assert states[1]["tool_call_count"] == 1
    assert states[1]["tools_used"] == ["register_for_event"]
    assert states[2]["tool_results"] == []
    assert states[2]["tools_used"] == []
    assert states[2]["tool_call_count"] == 0


@pytest.mark.asyncio
async def test_a_read_and_a_write_in_one_response_are_both_reported():
    model = _ScriptedModel(
        script={
            1: [
                [
                    _call("search_events", 1, call_id="c1"),
                    _call("register_for_event", 2, call_id="c2"),
                ]
            ]
        }
    )
    tools = [_tool("search_events", "events"), _tool("register_for_event", "events")]

    (state,), _ = await _run_thread(model, tools, turns=1)

    assert sorted(_results(state)) == [("register_for_event", 2), ("search_events", 1)]
    assert sorted(state["tools_used"]) == ["register_for_event", "search_events"]
    assert state["tool_call_count"] == 2
    assert _report(state)["invoked_write"] is True


@pytest.mark.asyncio
async def test_parallel_calls_to_one_tool_each_get_a_result_and_a_timing():
    model = _ScriptedModel(
        script={
            1: [[_call("search_events", 1, call_id="c1"), _call("search_events", 2, call_id="c2")]]
        }
    )

    (state,), _ = await _run_thread(model, [_tool("search_events", "events")], turns=1)

    assert sorted(_results(state)) == [("search_events", 1), ("search_events", 2)]
    # FIFO pairing may swap durations between same-tool calls; both are used.
    assert sorted(r.duration_ms for r in state["tool_results"]) == [11, 12]


@pytest.mark.asyncio
async def test_duplicate_ids_in_one_response_pair_with_their_calls_in_order():
    model = _ScriptedModel(
        script={1: [[_call("search_events", 1), _call("register_for_event", 2)]]}
    )
    tools = [_tool("search_events", "events"), _tool("register_for_event", "events")]

    (state,), _ = await _run_thread(model, tools, turns=1)

    assert len(state["tool_results"]) == 2
    for result in state["tool_results"]:
        # Each result is labeled with the call that produced it.
        assert result.data["tool"] == result.tool_name
        assert result.data["n"] == result.arguments["n"]


@pytest.mark.asyncio
async def test_the_report_for_a_later_turn_covers_only_that_turn():
    model = _ScriptedModel(
        script={
            1: [[_call("search_resources", 1)]],
            2: [[_call("register_for_event", 2)]],
        }
    )
    tools = [
        _tool("search_resources", "compute-resources"),
        _tool("register_for_event", "events"),
    ]

    states, _ = await _run_thread(model, tools, turns=3)

    second, third = _report(states[1]), _report(states[2])
    assert second["tool_count"] == 1
    assert second["invoked_write"] is True
    assert second["capabilities"] == ["browse_events"]
    assert third["tool_count"] == 0
    assert third["invoked_write"] is False
    assert third["capabilities"] == []


@pytest.mark.asyncio
async def test_exhausted_budget_reports_counts_but_no_results():
    """Known limit: the recursion path keeps no messages, so only timings remain,
    and an earlier turn's results aren't reported in their place."""
    model = _ScriptedModel(
        script={1: [[_call("register_for_event", 1)]], 2: [[_call("search_events", 2)]]},
        forever={2},
    )
    tools = [_tool("register_for_event", "events"), _tool("search_events", "events")]

    states, _ = await _run_thread(model, tools, turns=2)

    exhausted = states[1]
    assert exhausted["answer_unavailable"] is True
    assert exhausted["tools_used"] == ["search_events"]
    # The step cost per round depends on instrumentation, so not an exact count.
    assert exhausted["tool_call_count"] >= 1
    assert exhausted["tool_results"] == []
    assert _report(exhausted)["invoked_write"] is False


@pytest.mark.asyncio
async def test_a_long_turn_is_not_compacted_so_every_call_keeps_its_result(monkeypatch):
    """Compaction only takes earlier turns, so no call is summarized away mid-turn."""
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_TRIGGER_TOKENS", 700)
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_KEEP_TOKENS", 300)
    pad = "x" * 400
    model = _ScriptedModel(
        script={1: [[_call("search_events", n, pad=pad, call_id=f"c{n}")] for n in range(1, 5)]}
    )

    (state,), (capture,) = await _run_thread(model, [_tool("search_events", "events")], turns=1)

    assert not capture.get("summarized")
    assert state["tool_call_count"] == 4
    assert len(state["tool_results"]) == 4


@pytest.mark.asyncio
async def test_a_turn_after_compaction_reports_exactly_its_own_calls(monkeypatch):
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_TRIGGER_TOKENS", 700)
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_KEEP_TOKENS", 300)
    pad = "x" * 400
    model = _ScriptedModel(
        script={turn: [[_call("search_events", turn, pad=pad)]] for turn in range(1, 5)}
    )

    states, captures = await _run_thread(model, [_tool("search_events", "events")], turns=4)

    assert any(c.get("summarized") for c in captures[:-1])
    for turn, state in enumerate(states, start=1):
        assert _results(state) == [("search_events", turn)]
        assert state["tool_call_count"] == 1


# --- The turn slice and pairing, without the graph ----------------------------


def test_this_turn_is_the_messages_the_node_was_not_given():
    old = [HumanMessage("q1", id="h1"), AIMessage("a1", id="a1"), HumanMessage("q2", id="h2")]
    new = [AIMessage("", id="a2"), ToolMessage("{}", tool_call_id="x", id="t2")]

    assert _this_turn_messages(old, [*old, *new]) == new


def test_messages_without_ids_count_as_this_turns():
    """Mocked create_agent results carry no ids; a None in the input set would
    otherwise mark every one of them as old."""
    old = [HumanMessage("q1")]
    new = [AIMessage(""), ToolMessage("{}", tool_call_id="x")]

    assert _this_turn_messages(old, [*old, *new]) == [*old, *new]


async def _run_node(earlier: list, this_turn: list) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the node once over a mocked create_agent that returns earlier + this_turn.

    Returns the node's update and the span attributes it set.
    """
    agent = AsyncMock()
    agent.ainvoke.return_value = {"messages": [*earlier, *this_turn]}
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    with (
        patch.object(loop_mod, "create_agent", return_value=agent),
        patch.object(loop_mod, "get_llm", return_value=MagicMock()),
        patch.object(loop_mod, "get_tracer", return_value=tracer),
        patch.object(loop_mod, "_build_prompt_and_tools", return_value=("system", [], False)),
    ):
        reset_turn_capture()
        result = await run_loop({"messages": earlier, "tool_catalog": {}})
    attributes = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    return result, attributes


@pytest.mark.asyncio
async def test_a_result_whose_call_is_not_in_this_turn_is_an_orphan():
    """Its call was summarized away; it must not bind to an older call with the
    same reused id."""
    earlier = [
        HumanMessage("q1", id="h1"),
        AIMessage("", tool_calls=[_call("register_for_event", 1)], id="a1"),
        ToolMessage('{"n": 1}', tool_call_id=REUSED_ID, id="t1"),
        AIMessage("a1 answer", id="a1b"),
        HumanMessage("q2", id="h2"),
    ]
    this_turn = [
        ToolMessage('{"n": 2}', tool_call_id=REUSED_ID, id="t2"),
        AIMessage("a2 answer", id="a2b"),
    ]

    result, attributes = await _run_node(earlier, this_turn)

    assert result["tool_results"] == []
    assert attributes["agent.tool_results_orphaned"] == 1


@pytest.mark.asyncio
async def test_telemetry_counts_only_this_turns_results():
    """An orphan from an earlier turn isn't counted again."""
    earlier = [
        HumanMessage("q1", id="h1"),
        ToolMessage("{}", tool_call_id="lost", id="t-orphan"),
        AIMessage("a1", id="a1"),
        HumanMessage("q2", id="h2"),
    ]
    this_turn = [
        AIMessage("", tool_calls=[_call("search_events", 1)], id="a2"),
        ToolMessage('{"n": 1}', tool_call_id=REUSED_ID, id="t2"),
        AIMessage("a2 answer", id="a3"),
    ]

    result, attributes = await _run_node(earlier, this_turn)

    assert len(result["tool_results"]) == 1
    assert result["node_trace"][0]["tool_results"] == 1
    assert attributes["agent.tool_results_received"] == 1
    assert "agent.tool_results_orphaned" not in attributes
