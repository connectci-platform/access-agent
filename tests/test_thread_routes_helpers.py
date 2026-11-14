"""Direct unit coverage for thread_routes.py's small pure helpers.

No HTTP/JWKS harness needed — these don't touch auth, DB, or the agent.
"""

from src.api.thread_routes import _last_user_message, _serialize_values


class TestLastUserMessage:
    """Which message is the turn's question. How its content becomes text is
    parse_message's, tested in test_upload_ingress.py."""

    def test_returns_latest_user_message(self):
        latest = {"role": "user", "content": "second"}
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            latest,
        ]
        assert _last_user_message(messages) is latest

    def test_both_the_widget_and_langchain_shapes_are_user_turns(self):
        """The widget sends role:"user"; the full-screen client sends
        type:"human"."""
        human = {"type": "human", "content": [{"type": "text", "text": "Q"}]}
        assert _last_user_message([{"role": "user", "content": "a"}, human]) is human
        user = {"role": "user", "content": "b"}
        assert _last_user_message([human, {"type": "ai", "content": "r"}, user]) is user

    def test_no_user_message_is_empty(self):
        assert _last_user_message([{"role": "assistant", "content": "hi"}]) == {}
        assert _last_user_message([]) == {}


class TestSerializeValues:
    def test_empty_values_returns_empty_dict(self):
        assert _serialize_values(None) == {}
        assert _serialize_values({}) == {}

    def test_model_dump_messages_are_dumped(self):
        class _FakeModel:
            def model_dump(self):
                return {"type": "ai", "content": "hi"}

        result = _serialize_values({"messages": [_FakeModel()]})
        assert result == {"messages": [{"type": "ai", "content": "hi"}]}

    def test_plain_dict_message_passed_through_unchanged(self):
        """A message with no model_dump (already a plain dict, e.g. rebuilt
        from a checkpoint) must pass through as-is rather than erroring."""
        plain = {"type": "human", "content": "hi"}
        result = _serialize_values({"messages": [plain]})
        assert result == {"messages": [plain]}
        assert result["messages"][0] is plain
