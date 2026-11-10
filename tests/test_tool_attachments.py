"""A large tool attachment is stored once per thread and the message keeps a reference (D8-3054)."""

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import text

from src.thread_owners import get_thread_owner_store
from src.tool_attachments import (
    NOT_ATTACHED_NOTE,
    externalize,
    get_tool_attachment_store,
    referenced_ids,
)

TABLE = {
    "version": 1,
    "source": "XDMoD",
    "realm": "Jobs",
    "metric": "total_cpu_hours",
    "dimension": "person",
    "columns": [{"name": "User", "kind": "label"}, {"name": "CPU Hours", "kind": "number"}],
    "rows": [["a", 1.0], ["b", None]],
    "row_count": 2,
}


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as owners
    import src.tool_attachments as attachments

    owners._store = None
    attachments._store = None


def _store_one(thread_id: str) -> str:
    out, dropped = externalize({"table": TABLE}, thread_id)
    assert not dropped
    return str(out["table_ref"]["id"])


def test_a_table_is_stored_and_the_message_keeps_a_reference():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    chart = {"version": 1, "series": []}

    out, dropped = externalize({"table": TABLE, "chart": chart}, "t1")

    assert dropped is False
    assert "table" not in out
    assert out["chart"] is chart  # charts stay inline
    ref = out["table_ref"]
    assert ref["kind"] == "table"
    assert ref["row_count"] == 2
    assert ref["columns"] == TABLE["columns"]
    assert "rows" not in ref
    assert json.loads(get_tool_attachment_store().get("t1", ref["id"])) == TABLE


def test_the_reference_is_small_whatever_the_table_holds():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    big = {**TABLE, "rows": [[f"user {i}", float(i)] for i in range(50_000)], "row_count": 50_000}

    out, _ = externalize({"table": big}, "t1")

    assert len(json.dumps(out)) < 1000


def test_another_threads_id_finds_nothing():
    store = get_thread_owner_store()
    store.claim_thread("t1", "alice@x")
    store.claim_thread("t2", "bob@x")
    attachment = _store_one("t1")

    assert get_tool_attachment_store().get("t2", attachment) is None
    assert get_tool_attachment_store().get("t1", "0" * 32) is None


@pytest.mark.parametrize("thread_id", [None, "never-claimed"])
def test_a_table_that_cant_be_stored_is_dropped_not_left_inline(thread_id):
    out, dropped = externalize({"table": TABLE, "chart": {"a": 1}}, thread_id)

    assert dropped is True
    assert out == {"chart": {"a": 1}}


def test_without_a_database_the_table_is_dropped(monkeypatch):
    from src.config import settings
    from src.tool_attachments import ToolAttachmentStore

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    store = ToolAttachmentStore()

    assert externalize({"table": TABLE}, "t1", store) == ({}, True)
    assert store.get("t1", "x") is None
    assert store.ids_for("t1") == set()
    assert store.keep_only("t1", set()) == 0


def test_a_database_error_on_the_first_write_drops_the_table_without_raising(monkeypatch):
    import src.tool_attachments as attachments

    get_thread_owner_store().claim_thread("t1", "alice@x")

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database is down")

    monkeypatch.setattr(attachments.ToolAttachmentRow.__table__, "create", refuse)

    assert externalize({"table": TABLE}, "t1") == ({}, True)


def test_a_failed_create_all_keeps_no_engine(monkeypatch):
    import src.thread_owners as owners

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database is down")

    monkeypatch.setattr(owners.ThreadOwnerBase.metadata, "create_all", refuse)
    store = owners.ThreadOwnerStore()

    with pytest.raises(RuntimeError):
        store.session_factory()
    assert store._engine is None
    assert store._session_factory is None


def test_the_store_shares_the_thread_owner_stores_engine():
    sessions = get_thread_owner_store().session_factory()
    assert get_tool_attachment_store()._sessions() is sessions


def test_a_table_that_is_not_an_object_is_dropped():
    get_thread_owner_store().claim_thread("t1", "alice@x")

    assert externalize({"table": "not a table"}, "t1") == ({}, True)


def test_deleting_a_threads_owner_row_deletes_its_attachments():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    get_thread_owner_store().claim_thread("t2", "bob@x")
    kept = _store_one("t2")
    gone = _store_one("t1")
    sessions = get_thread_owner_store().session_factory()
    assert sessions is not None
    with sessions.kw["bind"].connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))  # Postgres always enforces it
        conn.execute(text("DELETE FROM thread_owners WHERE thread_id = 't1'"))
        conn.commit()

    assert get_tool_attachment_store().get("t1", gone) is None
    assert get_tool_attachment_store().get("t2", kept) is not None


def test_keep_only_deletes_what_the_thread_no_longer_refers_to():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    get_thread_owner_store().claim_thread("t2", "bob@x")
    kept = _store_one("t1")
    replaced = _store_one("t1")
    other = _store_one("t2")
    store = get_tool_attachment_store()

    assert store.ids_for("t1") == {kept, replaced}
    assert store.keep_only("t1", {kept}) == 1
    assert store.ids_for("t1") == {kept}
    assert store.keep_only("t1", set()) == 1
    assert store.ids_for("t2") == {other}


def test_referenced_ids_reads_tool_message_references():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    ref = externalize({"table": TABLE}, "t1")[0]
    messages = [
        AIMessage(content="hi"),
        ToolMessage(content="x", tool_call_id="a", artifact={"structured_content": ref}),
        ToolMessage(content="x", tool_call_id="b", artifact={"refusal": {"reason": "r"}}),
        ToolMessage(content="x", tool_call_id="c", artifact={"structured_content": "odd"}),
        ToolMessage(content="x", tool_call_id="d"),
    ]

    assert referenced_ids(messages) == {ref["table_ref"]["id"]}


def _wrapper(structured: Any, thread_id: str | None, data: Any = "summary"):
    from src.agent.domains.tools import MCPToolWrapper
    from src.tools.mcp_client import MCPClient, MCPToolResult

    client = MCPClient()
    calls: list[dict[str, Any]] = []

    async def fake_call_tool(**kwargs: Any) -> MCPToolResult:
        calls.append(kwargs)
        return MCPToolResult(success=True, data=data, structured=structured)

    client.call_tool = fake_call_tool  # type: ignore[method-assign]
    wrapper = MCPToolWrapper(
        name="get_raw_data",
        description="d",
        tool_server="xdmod-data",
        mcp_client=client,
        attachment_thread_id=thread_id,
    )
    content, artifact = asyncio.run(wrapper._arun())
    return content, artifact, calls


def test_a_thread_runs_table_reaches_the_message_as_a_reference():
    get_thread_owner_store().claim_thread("t1", "alice@x")

    content, artifact, calls = _wrapper({"table": TABLE}, "t1")

    assert calls[0]["include_structured"] is True
    assert content == json.dumps("summary")
    ref = artifact["structured_content"]["table_ref"]
    assert json.loads(get_tool_attachment_store().get("t1", ref["id"])) == TABLE


def test_a_run_with_no_thread_doesnt_ask_for_structured_content():
    _, _, calls = _wrapper(None, None)

    assert calls[0]["include_structured"] is False


def test_a_table_that_cant_be_stored_tells_the_model():
    """No owner row, so nothing is stored: the summary's promise is corrected."""
    content, artifact, _ = _wrapper({"table": TABLE}, "never-claimed")

    assert artifact is None
    assert json.loads(content) == f"summary\n\n{NOT_ATTACHED_NOTE}"


def test_a_table_that_cant_be_stored_beside_a_dict_result_tells_the_model():
    content, artifact, _ = _wrapper({"table": TABLE, "chart": {"a": 1}}, None, data={"x": 1})

    assert artifact == {"structured_content": {"chart": {"a": 1}}}
    assert json.loads(content) == {"result": {"x": 1}, "note": NOT_ATTACHED_NOTE}


def test_the_loop_gives_tools_the_thread_only_on_a_thread_run():
    from src.agent.domains.tools import create_mcp_tools_from_catalog

    catalog = {"servers": [{"server": "xdmod-data", "tools": [{"name": "get_raw_data"}]}]}
    (tool,) = create_mcp_tools_from_catalog(catalog, attachment_thread_id="t1")
    assert tool.attachment_thread_id == "t1"
    (tool,) = create_mcp_tools_from_catalog(catalog)
    assert tool.attachment_thread_id is None


def test_the_model_and_its_token_count_never_see_the_rows():
    from src.agent.token_estimate import estimate_messages

    big = {**TABLE, "rows": [[f"user {i}", float(i)] for i in range(20_000)]}
    bare = ToolMessage(content="summary", tool_call_id="c", name="get_raw_data")
    with_table = ToolMessage(
        content="summary",
        tool_call_id="c",
        name="get_raw_data",
        artifact={"structured_content": {"table": big}},
    )

    assert estimate_messages([with_table]) == estimate_messages([bare])
