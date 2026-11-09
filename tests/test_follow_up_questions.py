"""Follow-up question suggestions after a full-screen answer (D8-3053).

The model is never called for real: ``get_llm`` in the node's module is
replaced with a fake whose reply, delay, or error each test picks.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

import src.agent.graph as g
import src.agent.nodes.follow_up_questions as fu
from src.agent.context import AgentContext
from src.agent.turn_capture import record_turn_deadline, reset_turn_capture
from src.api.identity import CLAIM_KEY_HEADER
from src.llm import EMPTY_ANSWER_SENTINEL
from src.main import app

THREAD = AgentContext(acting_user=None, thread_run=True)
WIDGET = AgentContext.anonymous()
QUESTION = "Which resources have A100 GPUs?"
ANSWER = "Delta and Bridges-2 both have A100 nodes."


class _FakeLLM:
    def __init__(self, reply: Any = None, *, delay: float = 0.0, error: Exception | None = None):
        self.reply = reply
        self.delay = delay
        self.error = error
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []
        self.kwargs: dict[str, Any] = {}

    async def ainvoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> Any:
        self.calls.append((messages, config or {}))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)


@pytest.fixture
def fake_llm(monkeypatch):
    llm = _FakeLLM(json.dumps(["How do I get time on Delta?", "Does Bridges-2 have H100s?"]))

    def _get_llm(**kwargs: Any) -> _FakeLLM:
        llm.kwargs = kwargs
        return llm

    monkeypatch.setattr(fu, "get_llm", _get_llm)
    return llm


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "query": QUESTION,
        "messages": [
            HumanMessage(content=QUESTION, id="h1"),
            AIMessage(content=ANSWER, id="a1", additional_kwargs={"keep": 1}),
        ],
        "tool_catalog": {},
        "answer_unavailable": False,
        "tool_refusals": [],
    }
    state.update(overrides)
    return state


async def _node(state: dict[str, Any], context: AgentContext = THREAD) -> dict[str, Any]:
    return await fu.suggest_follow_ups_node(state, Runtime(context=context))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('["A?", "B?"]', ["A?", "B?"]),
        ('```json\n["A?", "B?"]\n```', ["A?", "B?"]),
        ('```\n["A?"]\n```', ["A?"]),
        ('Here are some:\n["A?", "B?"]\nHope that helps.', ["A?", "B?"]),
        ('["A?", 3, null, "B?"]', ["A?", "B?"]),
        ("[]", []),
        ("no array here", []),
        ("[not json]", []),
        ('{"questions": ["A?"]}', ["A?"]),
        ('"just a string"', []),
        ("] backwards [", []),
        ("", []),
        (None, []),
        (EMPTY_ANSWER_SENTINEL, []),
    ],
)
def test_parse_suggestions(text, expected):
    assert fu.parse_suggestions(text) == expected


def test_parse_suggestions_keeps_only_strings_from_the_outermost_array():
    assert fu.parse_suggestions('[1, 2] and {"a": 1}') == []
    assert fu.parse_suggestions('x ["A?"] y') == ["A?"]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Register me for the GPU workshop",
        "Sign me up for the next training",
        "Can you sign me up for the Delta webinar?",
        "Cancel my registration",
        "Please delete my announcement",
        "Submit a ticket about my login",
        "Could you create an event for next week?",
        "Update my announcement with the new date",
        "Post an announcement about the outage",
        "How do I open a ticket about this?",
        "Can you file a support ticket for me?",
        "I'd like you to add me to the waitlist",
        "Help me register for the workshop",
        "Go ahead and book a seat for me",
    ],
)
def test_action_requests_are_dropped(question):
    assert fu.asks_to_act(question)
    assert fu.filter_suggestions([question], QUESTION) == []


@pytest.mark.parametrize(
    "question",
    [
        "What GPUs does Delta have?",
        "When does registration for the workshop close?",
        "How do I cancel a job on Bridges-2?",
        "Is there an update on the Expanse outage?",
        "Which events are coming up next month?",
    ],
)
def test_information_questions_are_kept(question):
    assert not fu.asks_to_act(question)
    assert fu.filter_suggestions([question], QUESTION) == [question]


def test_duplicates_the_original_and_long_ones_are_dropped():
    long_one = "Is there " + "a very " * 20 + "long question?"
    assert len(long_one) > fu.MAX_QUESTION_CHARS
    candidates = [
        "  which resources have A100 GPUs  ",
        "What about H100s?",
        "what about h100s",
        '"What about H100s?"',
        long_one,
        "",
        "   ",
        "Where is GROMACS installed?",
    ]

    assert fu.filter_suggestions(candidates, QUESTION) == [
        "What about H100s?",
        "Where is GROMACS installed?",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Yes, cancel my registration",
        "Now delete my announcement",
        "Unenroll me from the workshop",
        "Drop my registration for Friday",
        "Go ahead with the registration",
        "Please proceed with the cancellation",
        "Yes, please confirm",
        "Proceed",
        "Ok, so mark it as attended",
        "Sure, approve the request",
    ],
)
def test_confirmations_and_marked_imperatives_are_dropped(question):
    assert fu.asks_to_act(question)
    assert fu.asks_to_act(question + "?")
    assert fu.filter_suggestions([question, question + "?"], QUESTION) == []


@pytest.mark.parametrize(
    "question",
    ["So what GPUs does Delta have?", "And which events are next week?", "Now, is Expanse up?"],
)
def test_a_leading_discourse_marker_alone_does_not_drop_a_question(question):
    assert fu.filter_suggestions([question], QUESTION) == [question]


def test_only_questions_are_kept():
    assert fu.filter_suggestions(["Delta has A100s.", "Tell me about Delta"], QUESTION) == []


@pytest.mark.parametrize(
    "question",
    [
        "Is https://delta.ncsa.illinois.edu up?",
        "What is on www.access-ci.org?",
        "Can help@access-ci.org answer this?",
    ],
)
def test_links_and_emails_are_dropped(question):
    assert fu.filter_suggestions([question], QUESTION) == []


def test_the_length_limit_is_the_one_the_prompt_asks_for():
    assert fu.MAX_QUESTION_CHARS == 100
    assert "under 100 characters" in fu._SYSTEM_PROMPT
    at_limit = "Q" * 99 + "?"
    assert fu.filter_suggestions([at_limit, "R" * 100 + "?"], QUESTION) == [at_limit]


def test_at_most_three_are_kept():
    candidates = [f"Question number {n}?" for n in range(5)]
    assert fu.filter_suggestions(candidates, QUESTION) == candidates[:3]


def test_whitespace_inside_a_question_is_collapsed():
    assert fu.filter_suggestions(["What   about\nH100s?"], QUESTION) == ["What about H100s?"]


# ---------------------------------------------------------------------------
# The tool list given to the model
# ---------------------------------------------------------------------------


CATALOG = {
    "servers": [
        {
            "server": "events",
            "tools": [
                {"name": "search_events", "description": "Find events.\nMore detail."},
                {"name": "register_for_event", "description": "Register."},
                {"name": "get_my_registrations", "description": "Your registrations."},
                {"name": "hinted_write", "description": "Writes.", "read_only": False},
                {"name": "", "description": "nameless"},
                {"name": "no_description"},
            ],
        },
        {"server": "down", "status": "unavailable", "tools": [{"name": "hidden_tool"}]},
    ]
}


def test_tool_summaries_leave_out_writes_and_sign_in_tools_for_anonymous():
    lines = fu.tool_summaries(CATALOG, signed_in=False)
    names = [line.split(":", 1)[0] for line in lines]

    assert names == ["search_access_documents", "search_events", "no_description"]
    assert "search_events: Find events." in lines
    assert "no_description: " in lines


def test_tool_summaries_include_sign_in_tools_when_signed_in():
    names = [line.split(":", 1)[0] for line in fu.tool_summaries(CATALOG, signed_in=True)]
    assert "get_my_registrations" in names
    assert "register_for_event" not in names


def test_tool_list_puts_this_turns_tools_first_and_stops_at_25():
    tools = [{"name": f"tool_{n}", "description": "d"} for n in range(40)]
    catalog = {"servers": [{"server": "s", "tools": tools}]}

    lines = fu.tool_summaries(
        catalog, signed_in=False, used=["tool_30", "tool_30", "register_for_event", "tool_5"]
    )
    names = [line.split(":", 1)[0] for line in lines]

    assert len(names) == fu.MAX_TOOLS_LISTED == 25
    assert names[:4] == ["tool_30", "tool_5", "search_access_documents", "tool_0"]
    assert "register_for_event" not in names


def test_tool_descriptions_are_one_short_line():
    catalog = {"servers": [{"server": "s", "tools": [{"name": "t", "description": "x" * 500}]}]}
    line = fu.tool_summaries(catalog, signed_in=False)[-1]
    assert line == "t: " + "x" * 60


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


async def test_suggestions_are_attached_to_the_answer_by_id(fake_llm):
    state = _state()

    update = await _node(state)

    (updated,) = update["messages"]
    assert updated.id == "a1"
    assert updated.content == ANSWER
    assert updated.additional_kwargs == {
        "keep": 1,
        "suggested_questions": ["How do I get time on Delta?", "Does Bridges-2 have H100s?"],
    }
    assert set(update) == {"messages"}
    # The state's own message is not mutated.
    assert state["messages"][-1].additional_kwargs == {"keep": 1}


async def test_the_call_is_small_unstreamed_and_sees_the_turn(fake_llm):
    long_answer = "A" * 5000
    catalog = {
        "servers": [{"server": "s", "tools": [{"name": "search_events", "description": "d"}]}]
    }

    await _node(
        _state(
            messages=[HumanMessage(QUESTION, id="h"), AIMessage(long_answer, id="a")],
            tool_catalog=catalog,
        )
    )

    assert fake_llm.kwargs == {"temperature": 0.2, "max_tokens": 200, "enable_thinking": False}
    ((messages, config),) = fake_llm.calls
    assert config == {"tags": ["nostream"]}
    prompt = messages[-1].content
    assert QUESTION in prompt
    assert "A" * 1500 + " ..." in prompt
    assert "A" * 1501 not in prompt
    assert "- search_events: d" in prompt
    assert "- search_access_documents:" in prompt
    assert fu.SUGGESTION_TIMEOUT_S == 4.0


async def test_list_content_answers_are_read_as_text(fake_llm):
    answer = AIMessage(
        content=[{"type": "text", "text": "Delta has "}, "A100s.", {"type": "image"}], id="a1"
    )

    update = await _node(_state(messages=[answer]))

    assert update["messages"][0].content == answer.content
    assert "Delta has A100s." in fake_llm.calls[0][0][-1].content


def test_content_that_is_neither_text_nor_blocks_reads_as_empty():
    assert fu._text(None) == ""


async def test_no_usable_suggestions_changes_nothing(fake_llm):
    fake_llm.reply = '["Register me for it", "Which resources have A100 GPUs?"]'
    assert await _node(_state()) == {}


async def test_garbage_output_changes_nothing(fake_llm):
    fake_llm.reply = "I can't think of any."
    assert await _node(_state()) == {}


async def test_an_error_changes_nothing_and_does_not_raise(fake_llm):
    fake_llm.error = RuntimeError("vLLM down")
    assert await _node(_state()) == {}


async def test_a_timeout_changes_nothing_and_does_not_raise(fake_llm, monkeypatch):
    monkeypatch.setattr(fu, "SUGGESTION_TIMEOUT_S", 0.05)
    monkeypatch.setattr(fu, "_MIN_CALL_S", 0.01)
    fake_llm.delay = 1.0

    started = time.monotonic()
    assert await _node(_state()) == {}
    assert time.monotonic() - started < 0.5


async def test_the_call_is_bounded_by_the_turn_deadline(fake_llm, monkeypatch):
    reset_turn_capture()
    monkeypatch.setattr(fu, "_MIN_CALL_S", 0.01)
    fake_llm.delay = 1.0
    record_turn_deadline(time.monotonic() + fu._DEADLINE_MARGIN_S + 0.05)

    started = time.monotonic()
    assert await _node(_state()) == {}
    assert time.monotonic() - started < 0.5
    reset_turn_capture()


async def test_too_little_time_left_skips_the_call(fake_llm):
    reset_turn_capture()
    record_turn_deadline(time.monotonic() + 1.0)

    assert await _node(_state()) == {}
    assert fake_llm.calls == []
    reset_turn_capture()


@pytest.mark.parametrize(
    "tool_results",
    [
        [SimpleNamespace(tool_name="register_for_event", data=None)],
        [{"tool_name": "cancel_registration", "data": None}],
        [{"tool_name": "some_tool", "data": {"status": "preview", "event": "x"}}],
    ],
)
async def test_turns_that_wrote_or_previewed_are_skipped(fake_llm, tool_results):
    assert await _node(_state(tool_results=tool_results)) == {}
    assert fake_llm.calls == []


async def test_read_only_tool_results_do_not_block_suggestions(fake_llm):
    results = [
        SimpleNamespace(tool_name="search_events", data={"status": "ok"}),
        {"tool_name": None, "data": "text"},
    ]
    assert (await _node(_state(tool_results=results)))["messages"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"answer_unavailable": True},
        {"tool_refusals": [{"tool_name": "get_my_registrations", "reason": "private_data"}]},
        {"messages": []},
        {"messages": [HumanMessage(QUESTION, id="h1")]},
        {"messages": [ToolMessage(content="x", tool_call_id="c1", id="t1")]},
        {"messages": [AIMessage(content="", id="a1")]},
        {"messages": [AIMessage(content=EMPTY_ANSWER_SENTINEL, id="a1")]},
        {
            "messages": [
                AIMessage(
                    content="calling",
                    id="a1",
                    tool_calls=[{"name": "search_events", "args": {}, "id": "c1"}],
                )
            ]
        },
        {"query": "  "},
    ],
)
async def test_answers_not_worth_following_are_skipped(fake_llm, overrides):
    assert await _node(_state(**overrides)) == {}
    assert fake_llm.calls == []


async def test_an_answer_without_an_id_is_skipped(fake_llm):
    answer = AIMessage(content=ANSWER)
    answer.id = None
    assert await _node(_state(messages=[answer])) == {}
    assert fake_llm.calls == []


async def test_widget_runs_are_untouched(fake_llm):
    assert await _node(_state(), WIDGET) == {}
    assert fake_llm.calls == []


def test_only_the_thread_route_marks_a_thread_run():
    assert AgentContext.anonymous().thread_run is False
    assert AgentContext.from_identity(None).thread_run is False
    assert AgentContext.from_identity(None, thread_run=True).thread_run is True
    identity = SimpleNamespace(sub="u@x", scopes=frozenset({"agent:read"}))
    signed_in = AgentContext.from_identity(identity, thread_run=True)  # type: ignore[arg-type]
    assert (signed_in.acting_user, signed_in.thread_run) == ("u@x", True)
    with pytest.raises(TypeError, match="thread_run"):
        AgentContext(acting_user=None, thread_run="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def test_graph_runs_suggestions_after_the_loop():
    graph = g.create_agent_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert ("tool_calling_loop", "suggest_follow_ups") in edges
    assert ("suggest_follow_ups", "__end__") in edges
    assert ("tool_calling_loop", "__end__") not in edges
    assert ("capability_discovery", "__end__") in edges
    assert ("capability_discovery", "suggest_follow_ups") not in edges


@pytest.fixture
def stub_loop(monkeypatch):
    async def _stub(state: dict[str, Any]) -> dict[str, Any]:
        n = sum(isinstance(m, AIMessage) for m in state["messages"]) + 1
        reply = AIMessage(content=ANSWER, id=f"answer-{n}")
        return {"final_answer": ANSWER, "messages": [*state["messages"], reply]}

    monkeypatch.setattr(g, "tool_calling_loop_node", _stub)


async def _stream(saver: InMemorySaver, thread_id: str, context: AgentContext) -> list[Any]:
    events = []
    async for event in g.stream_agent(
        query=QUESTION,
        session_id=thread_id,
        question_id="q",
        tool_catalog={},
        context=context,
        checkpointer=saver,
    ):
        events.append(event)
    return events


async def _thread(saver: InMemorySaver, thread_id: str) -> list[Any]:
    state = await g.create_checkpointed_graph(saver).aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    return state.values["messages"]


async def test_thread_run_checkpoints_the_answer_with_its_suggestions(stub_loop, fake_llm):
    saver = InMemorySaver()

    await _stream(saver, "t1", THREAD)
    await _stream(saver, "t1", THREAD)

    messages = await _thread(saver, "t1")
    assert [m.type for m in messages] == ["human", "ai", "human", "ai"]
    for answer in (messages[1], messages[3]):
        assert answer.content == ANSWER
        assert answer.additional_kwargs["suggested_questions"] == [
            "How do I get time on Delta?",
            "Does Bridges-2 have H100s?",
        ]


async def test_widget_run_never_calls_the_model_for_suggestions(stub_loop, fake_llm):
    saver = InMemorySaver()

    events = await _stream(saver, "t2", WIDGET)

    assert fake_llm.calls == []
    (answer,) = [m for m in await _thread(saver, "t2") if isinstance(m, AIMessage)]
    assert answer.additional_kwargs == {}
    suggestion_updates = [
        chunk["suggest_follow_ups"]
        for kind, chunk in events
        if kind == "updates" and "suggest_follow_ups" in chunk
    ]
    assert all(not update for update in suggestion_updates)


# ---------------------------------------------------------------------------
# The full-screen route
# ---------------------------------------------------------------------------


@pytest.fixture
async def thread_client(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.api.thread_runs as tr
    import src.thread_owners as owners
    import src.turn_reporter as reporter
    import src.turnstile as ts

    owners._store = None
    reporter._turn_reporter = None
    ts._guard = None
    tr._active_by_thread.clear()

    async def _registry() -> SimpleNamespace:
        return SimpleNamespace(catalog={})

    monkeypatch.setattr("src.api.routes.get_registry", _registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = InMemorySaver()
        yield c
        app.state.checkpointer = None
    reporter._turn_reporter = None
    owners._store = None
    ts._guard = None


def _values_event(body: str) -> dict[str, Any]:
    block = body.split("event: values\n", 1)[1]
    data: dict[str, Any] = json.loads(block.split("data: ", 1)[1].split("\n", 1)[0])
    return data


async def test_fullscreen_values_event_carries_the_whole_thread_with_suggestions(
    thread_client, stub_loop, fake_llm
):
    from src.turn_reporter import TurnReport, get_turn_reporter

    key = {CLAIM_KEY_HEADER: "k" * 43}
    r = await thread_client.post(
        "/api/v1/threads/t-fu/runs/stream",
        headers=key,
        json={"input": {"messages": [{"type": "human", "content": QUESTION}]}},
    )
    assert r.status_code == 200

    messages = _values_event(r.text)["messages"]
    assert [m["type"] for m in messages] == ["human", "ai"]
    answer = messages[1]
    assert answer["content"] == ANSWER
    assert answer["additional_kwargs"]["suggested_questions"] == [
        "How do I get time on Delta?",
        "Does Bridges-2 have H100s?",
    ]

    history = await thread_client.post("/api/v1/threads/t-fu/history", headers=key, json={})
    reloaded = history.json()[0]["values"]["messages"][1]
    assert reloaded["additional_kwargs"] == answer["additional_kwargs"]

    session = get_turn_reporter()._session_factory()
    try:
        row = session.query(TurnReport).filter_by(session_id="t-fu").one()
    finally:
        session.close()
    assert row.message_id == answer["id"]
