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

    def test_role_user_string_content(self):
        """The widget shape: role:"user" + plain string content. Must keep
        working unchanged."""
        messages = [{"role": "user", "content": "hello there"}]
        assert _last_user_text(messages) == "hello there"

    def test_type_human_string_content(self):
        """LangChain message format with plain string content."""
        messages = [{"type": "human", "content": "hello there"}]
        assert _last_user_text(messages) == "hello there"

    def test_type_human_block_array_content(self):
        """The full-screen client (agent-chat-ui) shape: type:"human" with
        content as a block array. This is the reproduced bug — the old code
        matched only role:"user" and str()'d list content into a stringified
        Python list instead of the actual text."""
        messages = [{"type": "human", "content": [{"type": "text", "text": "Q"}]}]
        assert _last_user_text(messages) == "Q"

    def test_multiple_messages_returns_latest_user_turn(self):
        messages = [
            {"type": "human", "content": [{"type": "text", "text": "first"}]},
            {"type": "ai", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _last_user_text(messages) == "second"

    def test_no_user_message_returns_empty_string(self):
        assert _last_user_text([{"role": "assistant", "content": "hi"}]) == ""
        assert _last_user_text([]) == ""

    def test_non_string_non_list_content_returns_empty_string(self):
        """content that is neither a string nor a block-array (e.g. None, or
        a bare dict) must not be str()'d into a repr — it's "" instead."""
        messages = [{"role": "user", "content": None}]
        assert _last_user_text(messages) == ""
        messages = [{"type": "human", "content": {"unexpected": "shape"}}]
        assert _last_user_text(messages) == ""

    def test_block_array_ignores_non_text_blocks(self):
        messages = [
            {
                "type": "human",
                "content": [
                    {"type": "image", "source": "data:..."},
                    {"type": "text", "text": "describe this"},
                ],
            }
        ]
        assert _last_user_text(messages) == "describe this"


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
