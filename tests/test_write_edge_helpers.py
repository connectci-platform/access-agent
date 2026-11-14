"""Edge paths of the helpers that read Confirm ids and stopped previews."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.nodes.tool_calling_loop import _confirm_ids
from src.api.stopped_turns import _only_previewed


def test_confirm_ids_are_empty_without_a_question():
    assert _confirm_ids([AIMessage(content="hi")]) == []


def test_confirm_ids_come_from_the_latest_question():
    messages = [
        HumanMessage(content="Confirm", additional_kwargs={"confirm": ["a" * 32]}),
        AIMessage(content="Done."),
        HumanMessage(content="thanks"),
    ]

    assert _confirm_ids(messages) == []


def test_a_preview_in_content_blocks_counts_as_only_previewed():
    body = json.dumps({"status": "preview", "executed": False})
    result = ToolMessage(content=[{"type": "text", "text": body}], tool_call_id="c1")

    assert _only_previewed(result) is True


def test_content_that_isnt_text_isnt_taken_for_a_preview():
    result = ToolMessage(content="x", tool_call_id="c1").model_copy(update={"content": 5})

    assert _only_previewed(result) is False
