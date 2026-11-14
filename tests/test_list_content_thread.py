"""A thread whose questions carry content blocks runs through every middleware.

Questions with shared files have list content, not a string. Every registered
middleware and the follow-up node run over such a thread here, and the model
must never get a Python list rendered as text, nor lose the notes the
middlewares add to the latest question.
"""

import json
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from src.agent.context import AgentContext
from src.agent.graph import create_agent_graph
from src.agent.middleware import uploads as uploads_mod
from src.agent.middleware.earlier_results import EARLIER_TURN_NOTE, earlier_results_view
from src.agent.middleware.requires_auth import (
    SIGNED_IN_NOTE,
    refusal_artifact,
    refusal_text,
    signed_in_since,
)
from src.agent.nodes import follow_up_questions as fu
from src.agent.nodes import tool_calling_loop as loop_mod
from tests.agent_runtime import context_for
from tests.test_gated_turns import _ScriptedModel

THREAD = "thread-1"
USER = "user123"
LIVE = "get_chart_data"
PRIVATE = "get_my_rp_accounts"


def _ref(ref_id: str, kind: str = "text", name: str = "run_md.sh") -> dict[str, Any]:
    ref = {"type": "attachment", "id": ref_id, "kind": kind, "name": name}
    if kind == "image":
        return {**ref, "mime_type": "image/png", "width": 64, "height": 64}
    return {**ref, "size": 7, "est_tokens": 3}


def _question(message_id: str, text: str | None, *refs: dict[str, Any]) -> HumanMessage:
    typed = [{"type": "text", "text": text}] if text else []
    return HumanMessage(content=[*typed, *refs], id=message_id)


def _call(name: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": call_id, "type": "tool_call"}],
        id=f"ai-{call_id}",
    )


def _thread() -> list[Any]:
    """Two earlier turns, one refused while signed out and one with live data,
    then a question that shares an image and nothing typed."""
    return [
        _question("q1", "what are my accounts?", _ref("1" * 32)),
        _call(PRIVATE, "c1"),
        ToolMessage(
            content=refusal_text("private_data"),
            name=PRIVATE,
            tool_call_id="c1",
            status="error",
            artifact=refusal_artifact("private_data"),
            id="t1",
        ),
        AIMessage(content="Please sign in.", id="a1"),
        _question("q2", "wait times on Anvil?"),
        _call(LIVE, "c2"),
        ToolMessage(content=json.dumps({"wait": 3}), name=LIVE, tool_call_id="c2", id="t2"),
        AIMessage(content="About 3 hours.", id="a2"),
        _question("q3", None, _ref("2" * 32, "image", "Screenshot.png")),
    ]


class _Store:
    def get_payload(self, thread_id: str, attachment_id: str) -> dict[str, Any] | None:
        if attachment_id == "2" * 32:
            return {
                "kind": "image",
                "mime_type": "image/png",
                "name": "Screenshot.png",
                "base64": "iVBORw0KGgo=",
                "width": 64,
                "height": 64,
            }
        return {"kind": "text", "name": "run_md.sh", "text": "echo hi", "text_truncated": False}


def _no_rendered_lists(messages: list[Any]) -> None:
    for message in messages:
        blocks = message.content if isinstance(message.content, list) else [message.content]
        for block in blocks:
            text = block if isinstance(block, str) else block.get("text", "")
            assert "[{'type'" not in text and "{'type':" not in text, text
            if isinstance(block, dict):
                assert block.get("type") in ("text", "image_url"), block


@pytest.fixture
def stack(monkeypatch):
    monkeypatch.setattr(uploads_mod, "get_tool_attachment_store", _Store)
    model = _ScriptedModel(script=["That's a stack trace."])
    with (
        patch.object(loop_mod, "get_llm", return_value=model),
        patch.object(loop_mod, "_build_prompt_and_tools", return_value=("system", [], False)),
    ):
        yield model


async def test_every_middleware_and_the_follow_ups_read_list_content(stack, monkeypatch):
    asked: list[tuple[str, str]] = []

    async def suggest(question: str, answer: str, tools: list[str], timeout: float) -> list[str]:
        asked.append((question, answer))
        return ["What does line 3 mean?"]

    monkeypatch.setattr(fu, "_suggest", suggest)
    monkeypatch.setattr(fu, "_time_left", lambda: 30.0)
    state = {
        "messages": _thread(),
        "query": "what is this?",
        "tool_catalog": {},
        "session_id": THREAD,
    }

    result = await create_agent_graph().ainvoke(
        state, context=AgentContext(acting_user=USER, thread_run=True)
    )

    sent = stack.received[0]
    _no_rendered_lists(sent)
    latest = [m for m in sent if isinstance(m, HumanMessage)][-1]
    texts = [b["text"] for b in latest.content if b["type"] == "text"]
    # The notes are text, so the request has a user query without the placeholder.
    assert "(The user shared a file without a message.)" not in texts
    assert SIGNED_IN_NOTE in texts
    assert EARLIER_TURN_NOTE in texts
    assert [b["type"] for b in latest.content].count("image_url") == 1
    first = next(m for m in sent if isinstance(m, HumanMessage))
    assert '<file name="run_md.sh"' not in str(first.content)
    assert result["final_answer"].startswith("That's a stack trace.")
    assert asked == [("what is this?", result["final_answer"])]
    assert result["messages"][-1].additional_kwargs["suggested_questions"]
    # The thread in state is untouched.
    assert state["messages"][-1].content == [_ref("2" * 32, "image", "Screenshot.png")]


async def test_the_follow_up_node_reads_a_list_content_thread(monkeypatch):
    seen: list[str] = []

    async def suggest(question: str, answer: str, tools: list[str], timeout: float) -> list[str]:
        seen.append(answer)
        return ["What does line 3 mean?"]

    monkeypatch.setattr(fu, "_suggest", suggest)
    monkeypatch.setattr(fu, "_time_left", lambda: 30.0)
    answer = AIMessage(content=[{"type": "text", "text": "A stack trace."}], id="a3")
    state = {
        "messages": [*_thread(), answer],
        "query": "what is this?",
        "tool_catalog": {},
    }
    context = AgentContext(acting_user=USER, thread_run=True)

    result = await fu.suggest_follow_ups_node(state, Runtime(context=context))

    assert seen == ["A stack trace."]
    assert result["messages"][0].id == "a3"


def test_the_sign_in_note_is_a_block_on_list_content():
    view = signed_in_since(_thread(), context_for(USER))

    assert view[-1].content[-1] == {"type": "text", "text": SIGNED_IN_NOTE}
    assert view[-1].content[0]["type"] == "attachment"


def test_the_earlier_turn_note_is_a_block_on_list_content():
    view = earlier_results_view(_thread())

    assert view[-1].content[-1] == {"type": "text", "text": EARLIER_TURN_NOTE}
