"""Uploads are stored once per thread and the question keeps a reference (D8-3086)."""

import base64
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine, insert, update
from sqlalchemy.orm import sessionmaker

import src.agent.graph as graph_mod
from src.agent.context import AgentContext
from src.agent.state import create_initial_state, human_turn
from src.agent.token_estimate import estimate_text
from src.thread_owners import get_thread_owner_store
from src.tool_attachments import (
    ToolAttachmentRow,
    get_tool_attachment_store,
    referenced_ids,
    upload_reference,
)
from src.turn_reporter import (
    TurnReport,
    TurnReportBase,
    TurnReporter,
    _assemble_turn_report,
    _hash_user,
)

IMAGE = {
    "kind": "image",
    "mime_type": "image/png",
    "name": "Screenshot.png",
    "base64": base64.b64encode(b"png-bytes").decode(),
    "width": 1600,
    "height": 1000,
}
PDF = {
    "kind": "pdf",
    "name": "guide.pdf",
    "base64": base64.b64encode(b"%PDF-1.7 twelve").decode(),
    "pages": 3,
    "text": "Section one. Section two.",
    "text_truncated": False,
    "no_text": False,
}
TEXT = {
    "kind": "text",
    "name": "run.slurm",
    "base64": base64.b64encode(b"#!/bin/bash\nsrun a.out\n").decode(),
    "text": "#!/bin/bash\nsrun a.out\n",
    "text_truncated": False,
}


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as owners
    import src.tool_attachments as attachments

    owners._store = None
    attachments._store = None


# ---------------------------------------------------------------------------
# The reference block
# ---------------------------------------------------------------------------


def test_an_image_reference_carries_its_type_and_size():
    assert upload_reference("a" * 32, IMAGE) == {
        "type": "attachment",
        "id": "a" * 32,
        "kind": "image",
        "name": "Screenshot.png",
        "mime_type": "image/png",
        "width": 1600,
        "height": 1000,
    }


def test_a_pdf_reference_carries_size_pages_and_estimate():
    assert upload_reference("b" * 32, PDF) == {
        "type": "attachment",
        "id": "b" * 32,
        "kind": "pdf",
        "name": "guide.pdf",
        "size": len(b"%PDF-1.7 twelve"),
        "pages": 3,
        "est_tokens": estimate_text(PDF["text"]),
    }


def test_a_text_reference_carries_size_and_estimate():
    assert upload_reference("c" * 32, TEXT) == {
        "type": "attachment",
        "id": "c" * 32,
        "kind": "text",
        "name": "run.slurm",
        "size": len(b"#!/bin/bash\nsrun a.out\n"),
        "est_tokens": estimate_text(TEXT["text"]),
    }


@pytest.mark.parametrize("raw", [b"", b"a", b"ab", b"abc", b"abcd"])
def test_the_size_is_the_original_byte_length(raw):
    payload = {**TEXT, "base64": base64.b64encode(raw).decode()}
    assert upload_reference("c" * 32, payload)["size"] == len(raw)


def test_referenced_ids_finds_references_in_message_content():
    refs = [{"type": "attachment", "id": "a" * 32}, {"type": "attachment", "id": "b" * 32}]
    messages = [
        HumanMessage(content="plain text"),
        HumanMessage(content=[{"type": "text", "text": "look"}, *refs]),
        HumanMessage(content=["a bare string", {"type": "text", "text": "x"}]),
        HumanMessage(content=[{"type": "attachment", "id": 5}, {"type": "attachment"}]),
        AIMessage(content="answer"),
    ]
    assert referenced_ids(messages) == {"a" * 32, "b" * 32}


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_get_payload_reads_an_upload_of_this_thread_only():
    owners = get_thread_owner_store()
    owners.claim_thread("t1", "alice@x")
    owners.claim_thread("t2", "alice@x")
    store = get_tool_attachment_store()
    upload = store.put("t1", "upload", IMAGE)
    table = store.put("t1", "table", {"rows": []})
    assert upload and table

    assert store.get_payload("t1", upload) == IMAGE
    assert store.get_payload("t2", upload) is None
    assert store.get_payload("t1", table) is None
    assert store.get_payload("t1", "0" * 32) is None


def test_get_payload_reads_a_database_error_as_not_found(monkeypatch):
    from src.tool_attachments import ToolAttachmentStore

    def down(self):
        raise RuntimeError("database is down")

    monkeypatch.setattr(ToolAttachmentStore, "_sessions", down)
    assert get_tool_attachment_store().get_payload("t1", "a" * 32) is None


def test_get_payload_without_a_database_is_none(monkeypatch):
    from src.tool_attachments import ToolAttachmentStore

    monkeypatch.setattr(ToolAttachmentStore, "_sessions", lambda self: None)
    store = get_tool_attachment_store()
    assert store.get_payload("t1", "a" * 32) is None
    assert store.delete_expired_uploads(datetime.now(UTC)) == 0
    assert store.delete_ids("t1", {"a" * 32}) == 0


def _age(attachment_id: str, days: int) -> None:
    store = get_tool_attachment_store()
    sessions = store._sessions()
    assert sessions is not None
    with sessions() as session, session.begin():
        session.execute(
            update(ToolAttachmentRow)
            .where(ToolAttachmentRow.id == attachment_id)
            .values(created_at=datetime.now(UTC) - timedelta(days=days))
        )


def test_expiry_deletes_only_old_uploads():
    get_thread_owner_store().claim_thread("t1", "alice@x")
    store = get_tool_attachment_store()
    old_upload = store.put("t1", "upload", IMAGE)
    new_upload = store.put("t1", "upload", IMAGE)
    old_table = store.put("t1", "table", {"rows": []})
    assert old_upload and new_upload and old_table
    _age(old_upload, 31)
    _age(old_table, 31)

    deleted = store.delete_expired_uploads(datetime.now(UTC) - timedelta(days=30))

    assert deleted == 1
    assert store.ids_for("t1") == {new_upload, old_table}


def test_delete_ids_removes_only_this_threads_rows():
    owners = get_thread_owner_store()
    owners.claim_thread("t1", "alice@x")
    owners.claim_thread("t2", "alice@x")
    store = get_tool_attachment_store()
    mine, kept = store.put("t1", "upload", IMAGE), store.put("t1", "upload", IMAGE)
    theirs = store.put("t2", "upload", IMAGE)
    assert mine and kept and theirs

    assert store.delete_ids("t1", {mine, theirs}) == 1
    assert store.delete_ids("t1", set()) == 0
    assert store.ids_for("t1") == {kept}
    assert store.ids_for("t2") == {theirs}


# ---------------------------------------------------------------------------
# Expiry runs at the start of a run, at most once an hour
# ---------------------------------------------------------------------------


@pytest.fixture
def _expiry_clock(monkeypatch):
    import src.api.thread_routes as routes

    monkeypatch.setattr(routes, "_last_expiry", None)
    now = [1000.0]
    monkeypatch.setattr(routes, "_monotonic", lambda: now[0])
    return now


async def test_the_sweep_expires_old_uploads_once_an_hour(monkeypatch, _expiry_clock):
    from src.api import thread_routes

    calls: list[datetime] = []
    store = get_tool_attachment_store()
    monkeypatch.setattr(store, "delete_expired_uploads", lambda when: calls.append(when) or 2)

    # A thread with nothing stored still runs the expiry, before its early return.
    await thread_routes._sweep_attachments("t-empty", object())
    await thread_routes._sweep_attachments("t-empty", None)
    _expiry_clock[0] += 3599
    await thread_routes._sweep_attachments("t-empty", None)
    assert len(calls) == 1
    age = datetime.now(UTC) - calls[0]
    assert timedelta(days=29, hours=23) < age < timedelta(days=30, minutes=1)

    _expiry_clock[0] += 2
    await thread_routes._sweep_attachments("t-empty", None)
    assert len(calls) == 2


async def test_an_expiry_error_is_logged_not_raised(monkeypatch, caplog, _expiry_clock):
    from src.api import thread_routes

    def broken(when):
        raise RuntimeError("db down")

    monkeypatch.setattr(get_tool_attachment_store(), "delete_expired_uploads", broken)
    await thread_routes._sweep_attachments("t-empty", None)
    assert "Couldn't delete expired uploads" in caplog.text


async def test_the_sweep_keeps_references_the_request_resends(monkeypatch):
    """A Retry resends a reference the saved thread no longer holds."""
    from src.api import thread_routes

    get_thread_owner_store().claim_thread("t1", "alice@x")
    store = get_tool_attachment_store()
    resent, stale = store.put("t1", "upload", IMAGE), store.put("t1", "upload", IMAGE)
    assert resent and stale

    class _Graph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"messages": []})

    monkeypatch.setattr(thread_routes, "create_checkpointed_graph", lambda cp: _Graph())
    await thread_routes._sweep_attachments("t1", object(), keep={resent})

    assert store.ids_for("t1") == {resent}


# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------


def test_a_text_question_is_built_as_before():
    msg = human_turn("hello")
    assert msg.content == "hello"
    assert msg.id is None
    assert msg.additional_kwargs == {}


def test_a_question_with_files_carries_its_blocks_and_id():
    blocks = [{"type": "text", "text": "what is this?"}, upload_reference("a" * 32, IMAGE)]
    msg = human_turn("what is this?", ["c" * 32], content=blocks, message_id="m-1")
    assert msg.content == blocks
    assert msg.id == "m-1"
    assert msg.additional_kwargs == {"confirm": ["c" * 32]}


def test_the_initial_state_carries_the_blocks_id_and_label():
    blocks = [upload_reference("a" * 32, IMAGE)]
    state = create_initial_state(
        query="",
        session_id="s",
        question_id="q",
        tool_catalog={},
        content_blocks=blocks,
        message_id="m-1",
        upload_label="Shared Screenshot.png",
    )
    assert state["messages"][0].content == blocks
    assert state["messages"][0].id == "m-1"
    assert state["upload_label"] == "Shared Screenshot.png"
    plain = create_initial_state(query="q", session_id="s", question_id="q", tool_catalog={})
    assert plain["upload_label"] is None
    assert plain["messages"][0].content == "q"


class _Graph:
    def __init__(self, prior: list[Any] | None = None):
        self.prior = prior
        self.state: dict[str, Any] = {}

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.prior} if self.prior else {})

    async def astream(self, state, config=None, **kwargs):
        self.state = state
        if False:
            yield


class _Saver:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def setup(self):
        return None


@pytest.mark.parametrize("path", ["fresh", "checkpointer", "db_uri"])
async def test_every_stream_agent_path_asks_the_question_with_its_blocks(monkeypatch, path):
    prior = None if path == "fresh" else [HumanMessage("q0", id="h0"), AIMessage("a0", id="a0")]
    g = _Graph(prior)
    monkeypatch.setattr(graph_mod, "create_agent_graph", lambda: g)
    monkeypatch.setattr(graph_mod, "create_checkpointed_graph", lambda cp: g)
    monkeypatch.setattr(graph_mod, "create_async_checkpointer", lambda uri: _Saver())
    blocks = [upload_reference("a" * 32, IMAGE)]
    kwargs: dict[str, Any] = {
        "checkpointer": object() if path == "checkpointer" else None,
        "use_checkpointing": path == "db_uri",
        "db_uri": "postgresql://x" if path == "db_uri" else None,
    }
    async for _ in graph_mod.stream_agent(
        query="",
        session_id="s",
        question_id="q",
        tool_catalog={},
        context=AgentContext.anonymous(),
        confirm_ids=["c" * 32],
        content_blocks=blocks,
        message_id="m-1",
        upload_label="Shared Screenshot.png",
        **kwargs,
    ):
        pass
    question = g.state["messages"][-1]
    assert question.content == blocks
    assert question.id == "m-1"
    assert question.additional_kwargs == {"confirm": ["c" * 32]}
    assert [m.id for m in g.state["messages"][:-1]] == [m.id for m in prior or []]
    assert g.state["upload_label"] == "Shared Screenshot.png"


def test_a_capability_question_with_a_file_goes_to_the_model():
    blocks = [{"type": "text", "text": "what can you do?"}, upload_reference("a" * 32, IMAGE)]
    with_file = create_initial_state(
        query="what can you do?",
        session_id="s",
        question_id="q",
        tool_catalog={},
        content_blocks=blocks,
    )
    plain = create_initial_state(
        query="what can you do?", session_id="s", question_id="q", tool_catalog={}
    )
    assert graph_mod._route_turn(with_file) == "tool_calling_loop"
    assert graph_mod._route_turn(plain) == "capability_discovery"


# ---------------------------------------------------------------------------
# Turn reports and the sidebar
# ---------------------------------------------------------------------------


@pytest.fixture
def reporter():
    r = TurnReporter()
    r._engine = create_engine("sqlite:///:memory:")
    TurnReportBase.metadata.create_all(r._engine)
    r._session_factory = sessionmaker(bind=r._engine)
    r._initialized = True
    return r


def test_the_report_keeps_the_upload_label():
    report, _ = _assemble_turn_report(
        final_state={},
        session_id="s",
        turn_index=1,
        question_id="q",
        query_text="",
        duration_ms=1.0,
        acting_user="me@x",
        success=True,
        capabilities=[],
        upload_label="Shared Screenshot.png",
    )
    assert report["query_text"] == ""
    assert report["upload_label"] == "Shared Screenshot.png"


def test_the_upload_label_column_is_added_to_an_old_table(tmp_path):
    from sqlalchemy import inspect, text

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    TurnReportBase.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE turn_reports DROP COLUMN upload_label"))
    r = TurnReporter()
    r._engine = engine
    r._migrate_columns()
    assert "upload_label" in {c["name"] for c in inspect(engine).get_columns("turn_reports")}


def _seed(reporter: TurnReporter, session_id: str, turn: int, query: str, label: str | None):
    with reporter._engine.begin() as conn:
        conn.execute(
            insert(TurnReport).values(
                session_id=session_id,
                turn_index=turn,
                question_id=f"{session_id}-{turn}",
                query_text=query,
                upload_label=label,
                user_hash=_hash_user("me@x"),
                created_at=datetime.now(UTC) + timedelta(seconds=turn),
            )
        )


def test_an_image_only_thread_is_titled_by_its_upload(reporter):
    _seed(reporter, "image-only", 1, "", "Shared Screenshot.png")
    _seed(reporter, "image-only", 2, "and now?", None)
    _seed(reporter, "typed", 1, "why did my job fail?", "Shared 2 files")
    labels = {
        row["session_id"]: row["label"]
        for row in reporter.list_threads_for_user(_hash_user("me@x") or "", 10)
    }
    assert labels == {"image-only": "Shared Screenshot.png", "typed": "why did my job fail?"}
