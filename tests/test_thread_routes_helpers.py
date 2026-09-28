"""Direct unit coverage for thread_routes.py's small pure helpers.

No HTTP/JWKS harness needed — these don't touch auth, DB, or the agent.
"""

from src.api.thread_routes import _last_user_text, _serialize_values


class TestLastUserText:
    def test_returns_latest_user_message(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _last_user_text(messages) == "second"

    def test_non_string_content_stringified(self):
        """A run body whose latest user message's content is a list/dict
        (some SDK clients send structured content) must not raise — it's
        coerced to str rather than returned as-is."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = _last_user_text(messages)
        assert isinstance(result, str)
        assert result == str([{"type": "text", "text": "hi"}])

    def test_no_user_message_returns_empty_string(self):
        assert _last_user_text([{"role": "assistant", "content": "hi"}]) == ""
        assert _last_user_text([]) == ""


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
