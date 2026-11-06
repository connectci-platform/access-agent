"""Answer tokens from the nested create_agent model reach stream_agent callers.

The model runs inside create_agent, which tool_calling_loop ainvokes from a
parent graph node. Those tokens only surface with subgraphs=True, so these tests
use a real parent StateGraph around a real create_agent.
"""

from typing import Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import END, START, StateGraph

from src.agent.graph import _flatten_subgraph_events
from src.api.answer_tokens import ANSWER_MODEL_NODE, is_answer_token

ANSWER = "Anvil supports Python and R."


class _StreamingFake(GenericFakeChatModel):
    """Streams word by word and tolerates bind_tools, as create_agent requires."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_StreamingFake":
        return self


class _State(TypedDict, total=False):
    messages: list[Any]
    done: bool


def _parent_graph() -> Any:
    from langchain.agents import create_agent

    async def loop(state: _State) -> dict[str, Any]:
        model = _StreamingFake(messages=iter([AIMessage(content=ANSWER)]))
        agent = create_agent(model=model, tools=[])
        await agent.ainvoke({"messages": [("user", "hi")]})
        return {"done": True}

    builder = StateGraph(_State)
    builder.add_node("tool_calling_loop", loop)
    builder.add_edge(START, "tool_calling_loop")
    builder.add_edge("tool_calling_loop", END)
    return builder.compile()


async def _events() -> list[tuple[str, Any]]:
    graph = _parent_graph()
    stream = graph.astream(
        {"messages": []},
        stream_mode=["custom", "messages", "updates"],
        subgraphs=True,
    )
    return [event async for event in _flatten_subgraph_events(stream)]


async def test_nested_answer_tokens_arrive():
    events = await _events()
    tokens = [
        chunk[0].content
        for stream_type, chunk in events
        if stream_type == "messages" and is_answer_token(*chunk)
    ]
    assert len(tokens) > 1
    assert "".join(tokens) == ANSWER


async def test_nested_updates_are_dropped():
    events = await _events()
    updates = [chunk for stream_type, chunk in events if stream_type == "updates"]
    assert all("model" not in chunk for chunk in updates)


async def test_top_level_update_survives():
    events = await _events()
    updates = [chunk for stream_type, chunk in events if stream_type == "updates"]
    assert {"tool_calling_loop": {"done": True}} in updates


async def test_nested_custom_events_pass():
    async def stream():
        yield ("inner", "x"), "custom", {"event": "tool_start"}

    assert [e async for e in _flatten_subgraph_events(stream())] == [
        ("custom", {"event": "tool_start"})
    ]


def test_summarization_is_not_answer_text():
    chunk = AIMessageChunk(content="summary")
    meta = {"langgraph_node": "SummarizationMiddleware.before_model"}
    assert not is_answer_token(chunk, meta)
    assert is_answer_token(chunk, {"langgraph_node": ANSWER_MODEL_NODE})


async def test_stream_agent_yields_flattened_pairs(monkeypatch):
    import src.agent.graph as graph_mod
    from src.agent.context import AgentContext

    seen: dict[str, Any] = {}

    class _Graph:
        async def astream(self, state, config=None, **kwargs):
            seen.update(kwargs)
            yield (), "updates", {"tool_calling_loop": {"done": True}}
            yield ("tool_calling_loop:1",), "updates", {"model": {}}
            yield ("tool_calling_loop:1",), "messages", ("tok", {"langgraph_node": "model"})

    monkeypatch.setattr(graph_mod, "create_agent_graph", _Graph)
    out = [
        event
        async for event in graph_mod.stream_agent(
            query="q",
            session_id="s",
            question_id="qid",
            tool_catalog={},
            context=AgentContext.anonymous(),
        )
    ]
    assert seen["subgraphs"] is True
    assert out == [
        ("updates", {"tool_calling_loop": {"done": True}}),
        ("messages", ("tok", {"langgraph_node": "model"})),
    ]
