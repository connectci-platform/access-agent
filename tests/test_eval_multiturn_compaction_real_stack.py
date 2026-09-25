"""Per-turn delta under REAL compaction, on the real stack.

The delta mechanism has now shipped twice with a green suite over a live blocker,
each time because the compaction case was asserted against a hand-built state
fixture that encoded what the author *believed* the system produced. The
positional-index mechanism died on exactly that: ``message_index`` is stamped
against the INNER list ``create_agent`` returns (which compaction rewrites),
while the runner's boundary reads the OUTER merged ``state["messages"]`` (which
never shrinks, because compaction's ``RemoveMessage`` is consumed inside the
inner subgraph, and which adopts the summary HumanMessage as its last human).
No fixture spelled that out, so every post-compaction turn's delta was empty and
the suite stayed green.

So this module builds no state by hand. It runs the real outer
``StateGraph(AgentState)`` with a real checkpointer, the real
``tool_calling_loop_node``, and the real ``_FlaggingSummarizationMiddleware``,
over a fake chat model that scripts tool calls and answers. The ONLY seam is
``get_llm`` — the provider boundary. Everything downstream of it, including the
compaction that rewrites the inner thread, is production code.

``SUMMARIZATION_TRIGGER_TOKENS`` is monkeypatched low so compaction actually
fires by turn 2-3; the assertion is that post-compaction turns still produce
NON-EMPTY deltas holding exactly that turn's calls.
"""

from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.graph import create_checkpointed_graph
from src.eval.multiturn import _turn_delta_state


class _ScriptedChatModel(BaseChatModel):
    """A chat model that emits one scripted tool call per turn, then an answer.

    ``create_agent`` binds tools to the model and drives the loop, so the model
    must implement ``bind_tools`` (returning itself — the fake ignores the
    schemas) and produce real ``tool_calls`` on an AIMessage. Each *agent* turn
    is two model calls: first a tool call, then the final answer. The summary
    call the middleware makes is served plain text by the same script.
    """

    turn: int = 0
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1

        # The middleware's summarization request is the only call whose last
        # message is a HumanMessage carrying the summary instruction.
        last = messages[-1] if messages else None
        if isinstance(last, HumanMessage) and "summar" in str(last.content).lower():
            msg = AIMessage(content=f"Summary of the conversation through turn {self.turn}.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Has this turn already made its tool call? Look for our own marker.
        already_called = any(
            isinstance(m, AIMessage)
            and m.tool_calls
            and any(tc["args"].get("turn") == self.turn for tc in m.tool_calls)
            for m in messages
        )

        if not already_called:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "probe_tool",
                        "args": {"turn": self.turn, "pad": "x" * 400},
                        # Deterministic vLLM-style id: identical every turn, so an
                        # identity diff on step_id would empty the delta.
                        "id": "chatcmpl-tool-0",
                    }
                ],
            )
        else:
            msg = AIMessage(content=f"Answer for turn {self.turn}. " + "detail " * 60)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _probe_tool():
    """A real StructuredTool the loop can execute, returning a per-turn payload."""
    from langchain_core.tools import StructuredTool

    def _run(turn: int, pad: str = "") -> str:
        import json

        return json.dumps({"turn": turn, "evidence": f"payload-turn{turn}", "pad": pad})

    # No tool_server attribute: _build_tool_results reads it via getattr with a
    # "" default, which is the same path a non-MCP tool takes in production.
    return StructuredTool.from_function(
        func=_run,
        name="probe_tool",
        description="Probe tool used by the compaction test.",
    )


@pytest.mark.asyncio
async def test_post_compaction_turns_get_nonempty_correctly_scoped_deltas(monkeypatch):
    """Real graph + checkpointer + loop + SummarizationMiddleware: once compaction
    fires, each turn's delta is NON-EMPTY and holds exactly that turn's call.

    This is the case the positional mechanism silently failed: the inner list it
    stamped positions against no longer corresponds to the outer list the runner
    reads. Message ids survive that crossing, so the delta stays correct.
    """
    from src.agent.nodes import tool_calling_loop as loop_mod
    from src.agent.turn_capture import reset_turn_capture

    # Compaction must actually fire mid-thread, so trigger well below what a few
    # padded turns accumulate. Real middleware, real thresholds plumbing.
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_TRIGGER_TOKENS", 700)
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_KEEP_TOKENS", 300)

    model = _ScriptedChatModel()
    probe = _probe_tool()

    # Only seam: the provider. create_agent, the middleware, the outer graph,
    # the checkpointer and the loop node are all the real thing.
    graph = create_checkpointed_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "compaction-thread"}}

    deltas = []
    summarized_turns = []

    with (
        patch.object(loop_mod, "get_llm", return_value=model),
        patch.object(loop_mod, "_build_prompt_and_tools", return_value=("system", [probe], False)),
    ):
        messages: list = []
        for turn in (1, 2, 3, 4):
            model.turn = turn
            reset_turn_capture()

            messages = [*messages, HumanMessage(content=f"Question for turn {turn}?")]
            state = await graph.ainvoke(
                {
                    "messages": messages,
                    "query": f"Question for turn {turn}?",
                    "session_id": "s",
                    "question_id": f"q_t{turn}",
                    "tool_catalog": {"servers": []},
                },
                config,
            )
            # Carry the merged thread forward the way run_agent does when resuming.
            messages = list(state["messages"])

            summarized_turns.append(loop_mod.get_turn_capture().get("summarized", False))
            deltas.append(_turn_delta_state(state))

    # The premise: compaction really fired on this thread. Without it the test
    # would be asserting nothing about the post-compaction path.
    assert any(summarized_turns), (
        f"compaction never fired — the test's premise is void; summarized flags={summarized_turns}"
    )
    first_compacted = summarized_turns.index(True)

    for idx, delta in enumerate(deltas):
        turn = idx + 1
        names = [r.tool_name for r in delta["tool_results"]]
        turns_in_delta = sorted({r.data["turn"] for r in delta["tool_results"]})

        # NON-EMPTY: the failure mode of the positional mechanism was an empty
        # delta on exactly these turns.
        assert names, f"turn {turn} delta was empty (compacted={summarized_turns[idx]})"
        # CORRECTLY SCOPED: exactly this turn's call, no prior turn's evidence —
        # even though every tool_call_id is the identical 'chatcmpl-tool-0'.
        assert turns_in_delta == [turn], (
            f"turn {turn} delta carried turns {turns_in_delta}; expected only [{turn}]"
        )
        assert delta["tools_used"] == ["probe_tool"]
        assert len(delta["node_trace"]) == 1

    # And at least one asserted turn was genuinely post-compaction.
    assert first_compacted < len(deltas) - 1 or summarized_turns[-1], (
        "no post-compaction turn was asserted"
    )


@pytest.mark.asyncio
async def test_post_compaction_turn_reports_get_the_same_delta(monkeypatch):
    """report_battery_turn receives the same per-turn view the judge does, so a
    post-compaction turn's report is neither empty nor credited with prior calls."""
    from src.agent.nodes import tool_calling_loop as loop_mod
    from src.agent.turn_capture import reset_turn_capture

    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_TRIGGER_TOKENS", 700)
    monkeypatch.setattr(loop_mod.settings, "SUMMARIZATION_KEEP_TOKENS", 300)

    model = _ScriptedChatModel()
    probe = _probe_tool()
    graph = create_checkpointed_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "compaction-report-thread"}}

    with (
        patch.object(loop_mod, "get_llm", return_value=model),
        patch.object(loop_mod, "_build_prompt_and_tools", return_value=("system", [probe], False)),
    ):
        messages: list = []
        states = []
        for turn in (1, 2, 3, 4):
            model.turn = turn
            reset_turn_capture()
            messages = [*messages, HumanMessage(content=f"Question for turn {turn}?")]
            state = await graph.ainvoke(
                {
                    "messages": messages,
                    "query": f"Question for turn {turn}?",
                    "session_id": "s",
                    "question_id": f"q_t{turn}",
                    "tool_catalog": {"servers": []},
                },
                config,
            )
            messages = list(state["messages"])
            states.append(state)

    last = _turn_delta_state(states[-1])
    assert [r.data["turn"] for r in last["tool_results"]] == [4]
    # The cumulative state genuinely held more than one turn's results, so the
    # slice did real work rather than trivially matching a one-entry list.
    assert len(states[-1]["tool_results"]) > 1
