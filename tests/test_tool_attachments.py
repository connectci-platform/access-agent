"""A large tool attachment is stored once per thread and the message keeps a reference (D8-3054)."""

import json

import pytest
from sqlalchemy import text

from src.thread_owners import get_thread_owner_store
from src.tool_attachments import externalize, get_tool_attachment_store

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


def test_a_table_is_stored_and_the_message_keeps_a_reference():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    chart = {"version": 1, "series": []}

    out = externalize({"table": TABLE, "chart": chart}, "t1")

    assert "table" not in out
    assert out["chart"] is chart  # charts stay inline
    ref = out["table_ref"]
    assert ref["kind"] == "table"
    assert ref["row_count"] == 2
    assert ref["columns"] == TABLE["columns"]
    assert "rows" not in ref
    stored = get_tool_attachment_store().get("t1", ref["id"])
    assert json.loads(stored) == TABLE


def test_the_reference_is_small_whatever_the_table_holds():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    big = {**TABLE, "rows": [[f"user {i}", float(i)] for i in range(50_000)], "row_count": 50_000}

    out = externalize({"table": big}, "t1")

    assert len(json.dumps(out)) < 1000


def test_another_threads_id_finds_nothing():
    store = get_thread_owner_store()
    store.claim_thread("t1", "alice@x")
    store.claim_thread("t2", "bob@x")
    ref = externalize({"table": TABLE}, "t1")["table_ref"]

    assert get_tool_attachment_store().get("t2", ref["id"]) is None
    assert get_tool_attachment_store().get("t1", "0" * 32) is None


def test_without_a_thread_the_table_stays_inline():
    assert externalize({"table": TABLE}, None) == {"table": TABLE}


def test_a_thread_with_no_owner_row_keeps_the_table_inline():
    assert externalize({"table": TABLE}, "never-claimed") == {"table": TABLE}


def test_deleting_a_threads_owner_row_deletes_its_attachments():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    get_thread_owner_store().claim_thread("t2", "bob@x")
    kept = externalize({"table": TABLE}, "t2")["table_ref"]["id"]
    gone = externalize({"table": TABLE}, "t1")["table_ref"]["id"]
    store = get_tool_attachment_store()
    store._ensure()
    with store._engine.begin() as conn:  # type: ignore[union-attr]
        conn.execute(text("DELETE FROM thread_owners WHERE thread_id = 't1'"))

    assert store.get("t1", gone) is None
    assert store.get("t2", kept) is not None


def test_delete_thread_removes_only_that_threads_attachments():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    get_thread_owner_store().claim_thread("t2", "bob@x")
    externalize({"table": TABLE}, "t1")
    externalize({"table": TABLE}, "t1")
    kept = externalize({"table": TABLE}, "t2")["table_ref"]["id"]

    assert get_tool_attachment_store().delete_thread("t1") == 2
    assert get_tool_attachment_store().get("t2", kept) is not None


def _wrapper(structured, thread_id):
    import asyncio

    from src.agent.domains.tools import MCPToolWrapper
    from src.tools.mcp_client import MCPClient, MCPToolResult

    client = MCPClient()

    async def fake_call_tool(**kwargs):
        return MCPToolResult(success=True, data="summary", structured=structured)

    client.call_tool = fake_call_tool  # type: ignore[method-assign]
    wrapper = MCPToolWrapper(
        name="get_raw_data",
        description="d",
        tool_server="xdmod-data",
        mcp_client=client,
        attachment_thread_id=thread_id,
    )
    return asyncio.run(wrapper._arun())


def test_a_thread_runs_table_reaches_the_message_as_a_reference():
    get_thread_owner_store().claim_thread("t1", "alice@x")

    content, artifact = _wrapper({"table": TABLE}, "t1")

    assert content == json.dumps("summary")
    ref = artifact["structured_content"]["table_ref"]
    assert json.loads(get_tool_attachment_store().get("t1", ref["id"])) == TABLE


def test_a_run_with_no_thread_keeps_the_table_on_the_message():
    _, artifact = _wrapper({"table": TABLE}, None)

    assert artifact == {"structured_content": {"table": TABLE}}


def test_the_loop_gives_tools_the_thread_only_on_a_thread_run():
    from src.agent.domains.tools import create_mcp_tools_from_catalog

    catalog = {"servers": [{"server": "xdmod-data", "tools": [{"name": "get_raw_data"}]}]}
    (tool,) = create_mcp_tools_from_catalog(catalog, attachment_thread_id="t1")
    assert tool.attachment_thread_id == "t1"
    (tool,) = create_mcp_tools_from_catalog(catalog)
    assert tool.attachment_thread_id is None


def test_without_a_database_nothing_is_stored(monkeypatch):
    from src.config import settings
    from src.tool_attachments import ToolAttachmentStore

    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    store = ToolAttachmentStore()

    assert externalize({"table": TABLE}, "t1", store) == {"table": TABLE}
    assert store.get("t1", "x") is None
    assert store.delete_thread("t1") == 0


def test_a_table_that_is_not_an_object_is_left_alone():
    get_thread_owner_store().claim_thread("t1", "alice@x")

    assert externalize({"table": "not a table"}, "t1") == {"table": "not a table"}


def test_the_model_and_its_token_count_never_see_the_rows():
    from langchain_core.messages import ToolMessage

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
