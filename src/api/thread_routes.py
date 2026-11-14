"""LangGraph thread/run protocol endpoints: create + history (resume) + run.

The Bearer agent token is optional, not required: a caller without one is
anonymous, not rejected. Every route goes through require_thread_access in
src/api/thread_access.py, which is where access is decided.
"""

import asyncio
import base64
import gzip
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ..agent.context import AgentContext
from ..agent.graph import create_checkpointed_graph, stream_agent
from ..agent.middleware.write_gate import MAX_CONFIRM_IDS, clean_confirm_ids
from ..agent.nodes.capability_discovery import is_capability_question
from ..agent.profile import UserProfile
from ..agent.profile_source import allocated_profile_for
from ..agent.state import human_turn
from ..agent.turn_capture import get_turn_capture, reset_turn_capture
from ..config import settings
from ..tool_attachments import (
    UPLOAD_KIND,
    get_tool_attachment_store,
    referenced_ids,
    upload_reference,
)
from ..turn_reporter import _hash_user, get_turn_reporter
from ..turnstile import NOT_CHARGED, charge_query, check_turnstile
from ..uploads.process import Processed, Rejected, clean_filename, process_upload
from .answer_tokens import ANSWER_MODEL_NODE, is_answer_token
from .handoff_tokens import (
    DEFAULT_TTL_SECONDS,
    exchange_handoff_token,
    mint_handoff_token,
)
from .identity import ClaimKey, Identity, require_identity
from .sse import format_sse_event
from .stopped_turns import answer_delivered as _answer_delivered
from .stopped_turns import (
    keeping_cut_off_turn,
    stopped_turn_saved,
    stopped_turn_saving,
    without_tool_messages,
)
from .thread_access import refuse_thread, require_thread_access
from .thread_runs import (
    acquire_thread_run,
    cancel_run,
    held_run,
    release_thread_run,
    run_with_timeout,
)

logger = logging.getLogger(__name__)

thread_router = APIRouter()


class HistoryRequest(BaseModel):
    limit: int = 10
    before: str | None = None


class ExchangeRequest(BaseModel):
    token: str


class SearchRequest(BaseModel):
    """SDK thread-search body. ``metadata`` (e.g. graph_id/assistant_id) is
    accepted for protocol compatibility and ignored — single-agent deployment,
    scope is already per-user via the caller's identity."""

    metadata: dict[str, Any] | None = None
    limit: int = 10


class RunInput(BaseModel):
    messages: list[dict[str, Any]] = []


class RunRequest(BaseModel):
    """SDK run body. The route reads ``input.messages``, ``turnstile_token`` (top
    level, or in ``context`` because the SDK's submit can't add a top-level field),
    ``resource_context`` from ``config.configurable`` or ``context``, and
    ``time_zone`` (the browser's IANA zone, e.g. "America/New_York") and
    ``replace_stopped_turn`` (true from Retry or Edit on a stopped turn) from
    ``context``; everything else is accepted for protocol compatibility and
    ignored. The body's ``context`` never reaches the agent's runtime context,
    which comes from the verified token only, so a body claiming a user or
    scopes has no effect. A
    run always creates-and-owns a non-existent thread, so ``if_not_exists`` is
    accepted and ignored rather than honored."""

    input: RunInput = RunInput()
    assistant_id: str | None = None
    stream_mode: Any = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    if_not_exists: str | None = None
    turnstile_token: str | None = None


# Longest IANA names are ~30 characters; anything far past that isn't one.
_MAX_TIME_ZONE_LEN = 64


def _time_zone_from(context: dict[str, Any] | None) -> str | None:
    """The run body's ``context.time_zone`` if it names a real IANA zone, else None.

    A display preference, not identity: it only changes how times are written
    for the user, so taking it from the body is fine. A bad value is dropped,
    never an error, and the turn falls back to UTC.
    """
    value = (context or {}).get("time_zone")
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > _MAX_TIME_ZONE_LEN:
        logger.debug("Ignoring time_zone that isn't a short string: %r", value)
        return None
    try:
        ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError):
        logger.debug("Ignoring unknown time_zone %r", value)
        return None
    return value


async def _write_turn_report(
    *,
    final_state: dict[str, Any],
    thread_id: str,
    run_id: str,
    query: str,
    start_time: float,
    caller: str | None,
    success: bool,
) -> None:
    """Turn report (denormalized read model). Off the response path; never
    raises into the stream. Mirrors routes.py:454-483 (success) /
    routes.py:500-527 (failure). acting_user == `caller` here; session_id ==
    thread_id (the thread/run path keys the reporter by thread_id, exactly as
    stream_agent is called with session_id=thread_id). Called OUTSIDE
    run_with_timeout/held_run — see thread_run_stream's placement note.
    """
    try:
        from ..agent.domains.capabilities import get_capability_registry as _cap_reg
        from ..services.resource_matcher import resources_for_turn
        from ..turn_reporter import get_turn_reporter

        reporter = get_turn_reporter()
        prior_turns = await asyncio.to_thread(reporter.count_turns_for_session, thread_id)
        turn_index = prior_turns + 1 if prior_turns is not None else None  # None on DB error → NULL
        resources = await resources_for_turn(query, str(final_state.get("final_answer") or ""))
        await asyncio.to_thread(
            reporter.log_turn_report,
            final_state=final_state,
            session_id=thread_id,
            turn_index=turn_index,
            question_id=run_id,
            query_text=query,
            duration_ms=(time.time() - start_time) * 1000,
            acting_user=caller,
            success=success,
            capabilities=_cap_reg().infer_capability_ids(final_state.get("tool_results", [])),
            resources=resources,
            turn_capture=get_turn_capture(),
            judge=None,
            upload_label=final_state.get("upload_label"),
        )
    except Exception:
        logger.exception("Turn report write failed")


# Reports written after a cancel, held so the loop doesn't drop them mid-write.
_detached_reports: set[asyncio.Task[None]] = set()


def _report_if_delivered(
    final_state: dict[str, Any], report: Callable[..., Coroutine[Any, Any, None]]
) -> bool:
    """On a cancel, report the turn as a success if its answer was delivered.

    The write runs in its own task, so the cancel that ended the run can't cut
    it short. Returns whether it was started.
    """
    if not _answer_delivered(final_state):
        return False
    task = asyncio.ensure_future(report(state=final_state, ok=True))
    _detached_reports.add(task)
    task.add_done_callback(_detached_reports.discard)
    return True


async def _stream_run_events(
    *,
    thread_id: str,
    parsed: "ParsedMessage",
    message_id: str,
    checkpointer: Any,
    final_state: dict[str, Any],
    replace_stopped_turn: bool = False,
    **run: Any,
) -> AsyncGenerator[str, None]:
    """``_run_events`` for one run, keeping the turn in the thread when it ends
    before its answer was delivered (see stopped_turns.keeping_cut_off_turn).

    The sweep and the message's files come first, before the turn's timer
    starts, so a slow PDF doesn't eat the turn's time. What was left out goes
    to the client as one ``upload_dropped`` event before the answer. The
    turn's ``upload_label`` goes into ``final_state`` for its report.
    """
    shared = await _sweep_then_share(thread_id, checkpointer, parsed)
    final_state["upload_label"] = shared.label()
    if shared.notes:
        yield format_sse_event("custom", {"type": "upload_dropped", "files": shared.notes})
    streamed: dict[str, AIMessageChunk] = {}
    tool_messages: list[Any] = []
    content_blocks = shared.content(parsed.text)
    # The question as stream_agent builds it, which a cut-off turn saves.
    question = human_turn(
        parsed.text, parsed.confirm_ids, content=content_blocks, message_id=message_id
    )
    events = _run_events(
        thread_id=thread_id,
        query=parsed.text,
        confirm_ids=parsed.confirm_ids,
        content_blocks=content_blocks,
        question=question,
        upload_label=final_state["upload_label"],
        checkpointer=checkpointer,
        final_state=final_state,
        streamed=streamed,
        tool_messages=tool_messages,
        **run,
    )
    async with run_with_timeout(settings.AGENT_TURN_TIMEOUT_S):
        async for event in keeping_cut_off_turn(
            events,
            checkpointer=checkpointer,
            thread_id=thread_id,
            question=question,
            final_state=final_state,
            streamed=streamed,
            tool_messages=tool_messages,
            replace_stopped_turn=replace_stopped_turn,
        ):
            yield event


async def _run_events(
    *,
    thread_id: str,
    run_id: str,
    query: str,
    context: AgentContext,
    resource_context: str | None,
    profile: UserProfile | None,
    time_zone: str | None,
    checkpointer: Any,
    final_state: dict[str, Any],
    streamed: dict[str, AIMessageChunk],
    tool_messages: list[Any],
    confirm_ids: list[str],
    content_blocks: list[dict[str, Any]] | None,
    question: HumanMessage,
    upload_label: str | None,
) -> AsyncGenerator[str, None]:
    """The messages/custom/values SSE sequence for one run, mutating
    ``final_state`` in place (mirrors turn_capture's in-place-mutation pattern)
    so the caller can read it after this generator is exhausted. Extracted
    from thread_run_stream's ``_gen`` to keep both under ruff's branch/statement
    caps once the turn-report write was added. The caller sends the metadata
    event first.
    """
    from .routes import get_registry

    registry = await get_registry()
    # ``streamed``: answer chunks by message id, in arrival order, for the
    # values fallback below and for the stopped answer a stop saves.
    final_messages: list[dict[str, Any]] = []
    # The thread as the latest node update leaves it, for the values event.
    thread: list[Any] = []

    # Tool calls are kept aside for a cut-off turn's write record.
    async for stream_type, chunk in without_tool_messages(
        tool_messages,
        stream_agent(
            query=query,
            session_id=thread_id,
            question_id=run_id,
            tool_catalog=registry.catalog,
            context=context,
            resource_context=resource_context,
            profile=profile,
            time_zone=time_zone,
            checkpointer=checkpointer,
            confirm_ids=confirm_ids,
            content_blocks=content_blocks,
            # Read as the run starts, after keeping_cut_off_turn has made sure
            # the thread doesn't already hold it.
            message_id=question.id,
            upload_label=upload_label,
        ),
    ):
        if stream_type == "custom":
            if (
                isinstance(chunk, dict)
                and chunk.get("type") == "status"
                and isinstance(chunk.get("message"), str)
            ):
                yield format_sse_event("custom", {"type": "status", "message": chunk["message"]})
        elif stream_type == "messages":
            msg, metadata = chunk
            if is_answer_token(msg, metadata):
                key = msg.id or ""
                streamed[key] = streamed[key] + msg if key in streamed else msg
                if not streamed[key].text.strip():
                    # Whitespace before any visible text (such as the "\n\n" after
                    # a think block) would open a blank bubble. Held back, and not
                    # resent: the final values event replaces the text.
                    continue
                # The SDK reads event `messages` with data [message, metadata].
                yield format_sse_event(
                    "messages", [msg.model_dump(), {"langgraph_node": ANSWER_MODEL_NODE}]
                )
        elif stream_type == "updates" and isinstance(chunk, dict):
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    final_state.update(node_output)
                    msgs = node_output.get("messages")
                    if msgs:
                        # As the checkpoint applies it: a whole thread replaces by
                        # id, and suggest_follow_ups' single answer replaces its own.
                        thread = cast("list[Any]", add_messages(thread, msgs))
                        final_state["messages"] = thread
                        final_messages = [
                            m.model_dump() if hasattr(m, "model_dump") else m for m in thread
                        ]

    refusals = final_state.get("tool_refusals") or []
    if refusals:
        # The caller asked for something they may not do here. Sent as a
        # LangGraph `custom` event because useStream drops event names it doesn't
        # know; `type` keeps it apart from the ui events on the same channel. The
        # turn still completes below with the model's reply.
        yield format_sse_event("custom", {"type": "requires_auth", "reason": refusals[0]["reason"]})

    if streamed and not final_messages:
        # The model produced tokens but no node update carried messages. One AI
        # message per streamed id, so separate rounds are never merged.
        final_messages = [
            AIMessage(content=c.content, id=c.id or None).model_dump() for c in streamed.values()
        ]

    yield format_sse_event("values", {"messages": final_messages})


_REF_ID = re.compile(r"[0-9a-f]{32}")
_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


@dataclass
class IncomingUpload:
    kind: Literal["image", "pdf", "text"]
    mime_type: str
    name: str
    data: bytes


@dataclass
class ParsedMessage:
    text: str = ""
    uploads: list[IncomingUpload] = field(default_factory=list)
    ref_ids: list[str] = field(default_factory=list)
    confirm_ids: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _size_label(nbytes: int) -> str:
    if nbytes >= 1024 * 1024:
        return f"{nbytes // (1024 * 1024)} MB"
    return f"{nbytes // 1024} KB"


def _block_name(block: dict[str, Any]) -> str:
    meta = block.get("metadata")
    if isinstance(meta, dict):
        for key in ("name", "filename"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return clean_filename(value)
    return "a file"


def _upload_kind(block: dict[str, Any]) -> tuple[Literal["image", "pdf", "text"], str] | None:
    """The upload kind and declared mime type for an allow-listed block."""
    mime = block.get("mimeType")
    if not isinstance(mime, str):
        return None
    if block.get("type") == "image" and mime in _IMAGE_TYPES:
        return "image", mime
    if block.get("type") == "file" and mime == "application/pdf":
        return "pdf", mime
    if block.get("type") == "file" and mime == "text/plain":
        return "text", mime
    return None


_KIND_LABEL = {"image": "images", "pdf": "PDFs", "text": "text files"}
REF_NOT_FOUND = "A file shared earlier couldn't be found."


def _read_upload(block: dict[str, Any], kind_max: dict[str, int]) -> IncomingUpload | str:
    """Decode one image or file block, or return the sentence for why it was left out."""
    name = _block_name(block)
    found = _upload_kind(block)
    data = block.get("data")
    if found is None or not isinstance(data, str):
        return f"{name} was left out: that kind of file isn't supported."
    kind, mime = found
    limit = kind_max[kind]
    too_big = f"{name} was left out: {_KIND_LABEL[kind]} can be at most {_size_label(limit)}."
    # Base64 is 4 characters per 3 bytes, so an oversized claim is refused before decoding.
    if len(data) // 4 * 3 > limit + 3:
        return too_big
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError:
        return f"{name} was left out: it couldn't be read."
    if len(raw) > limit:
        return too_big
    return IncomingUpload(kind, mime, name, raw)


def _upload_limits(anonymous: bool) -> tuple[int, int, dict[str, int]]:
    """Files per message, total bytes per message, and bytes per file by kind."""
    sfx = "_ANON" if anonymous else ""
    kind_max = {
        "image": getattr(settings, f"UPLOAD_MAX_IMAGE_BYTES{sfx}"),
        "pdf": getattr(settings, f"UPLOAD_MAX_PDF_BYTES{sfx}"),
        "text": getattr(settings, f"UPLOAD_MAX_TEXT_BYTES{sfx}"),
    }
    return (
        getattr(settings, f"UPLOAD_MAX_FILES{sfx}"),
        getattr(settings, f"UPLOAD_MAX_TOTAL_BYTES{sfx}"),
        kind_max,
    )


def _over_message_limit(
    parsed: ParsedMessage, upload: IncomingUpload, max_files: int, max_total: int
) -> str | None:
    """The sentence for why one more upload doesn't fit in this message, if it doesn't."""
    if len(parsed.uploads) + len(parsed.ref_ids) >= max_files:
        return f"{upload.name} was left out: at most {max_files} files per message."
    if sum(len(u.data) for u in parsed.uploads) + len(upload.data) > max_total:
        return (
            f"{upload.name} was left out: files can total at most "
            f"{_size_label(max_total)} per message."
        )
    return None


def _take_ref(parsed: ParsedMessage, ref: Any, max_files: int) -> None:
    """Add a stored-attachment reference to the message, or note why it was left out."""
    if not isinstance(ref, str) or not _REF_ID.fullmatch(ref):
        parsed.dropped.append(REF_NOT_FOUND)
    elif ref in parsed.ref_ids:
        return
    elif len(parsed.uploads) + len(parsed.ref_ids) >= max_files:
        parsed.dropped.append(
            f"A file shared earlier was left out: at most {max_files} files per message."
        )
    else:
        parsed.ref_ids.append(ref)


def parse_message(content: Any, *, anonymous: bool) -> ParsedMessage:
    """Split a run request's message content into typed parts.

    Client input: nothing here raises. Blocks that are off the allow-list or
    over a limit are left out and named in ``dropped``. Limits are checked on
    the decoded byte length.
    """
    if isinstance(content, str):
        return ParsedMessage(text=content)
    if not isinstance(content, list):
        return ParsedMessage()

    max_files, max_total, kind_max = _upload_limits(anonymous)
    parsed = ParsedMessage(confirm_ids=extract_confirm_ids(content))
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parsed.dropped.append("Part of the message couldn't be read and was left out.")
            continue
        btype = block.get("type")
        if btype == "text":
            texts.append(block["text"] if isinstance(block.get("text"), str) else "")
        elif btype == "attachment":
            _take_ref(parsed, block.get("id"), max_files)
        elif btype != "confirm":
            result = _read_upload(block, kind_max)
            if isinstance(result, str):
                parsed.dropped.append(result)
            else:
                refusal = _over_message_limit(parsed, result, max_files, max_total)
                if refusal:
                    parsed.dropped.append(refusal)
                else:
                    parsed.uploads.append(result)
    parsed.text = " ".join(t for t in texts if t)
    return parsed


def _extract_text_content(content: Any) -> str:
    """Render a message's ``content`` field as plain text.

    Callers send a plain string or a LangChain-style list of content blocks.
    Only the text blocks count; anything else yields "".
    """
    return parse_message(content, anonymous=True).text


def extract_confirm_ids(content: Any) -> list[str]:
    """Preview ids from a Confirm click's ``confirm`` block.

    Client input, so anything malformed is dropped rather than raised: only
    exactly-32-lowercase-hex strings survive, deduplicated, at most 10.
    """
    if not isinstance(content, list):
        return []
    ids: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "confirm":
            for preview_id in clean_confirm_ids(block.get("preview_ids")):
                if preview_id not in ids:
                    ids.append(preview_id)
    return ids[:MAX_CONFIRM_IDS]


def _last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    users = (m for m in reversed(messages) if m.get("type") == "human" or m.get("role") == "user")
    return next(users, {})


def _last_user_content(messages: list[dict[str, Any]]) -> Any:
    return _last_user_message(messages).get("content", "")


def _message_id(message: dict[str, Any]) -> str:
    """The id the turn's question is saved under: the client's, when it is a
    UUID in canonical form, otherwise a new one. A stopped turn finds its
    question by this id. One the thread already holds is replaced once the run
    reads the thread (stopped_turns.keeping_cut_off_turn)."""
    sent = message.get("id")
    if isinstance(sent, str):
        try:
            if str(uuid.UUID(sent)) == sent:
                return sent
        except ValueError:
            pass
    return str(uuid.uuid4())


def _shares_files(content: Any) -> bool:
    """Whether the message has any block but typed text and a Confirm click,
    including ones parse_message will leave out."""
    return isinstance(content, list) and any(
        not (isinstance(block, dict) and block.get("type") in ("text", "confirm"))
        for block in content
    )


@dataclass
class SharedFiles:
    """What a run's message shares: a reference per file now stored for the
    thread, in the order the question lists them, and a sentence for each
    file that was left out."""

    references: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def content(self, text: str) -> list[dict[str, Any]] | None:
        """The question's blocks: the typed text, the references, then the
        notes, so the model can tell the user what it didn't get. None when
        nothing was shared, and the question is the text alone."""
        if not self.references and not self.notes:
            return None
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        blocks.extend(self.references)
        if self.notes:
            notes = "\n".join(f"[Not shared: {note}]" for note in self.notes)
            blocks.append({"type": "text", "text": notes})
        return blocks

    def label(self) -> str | None:
        """The question's title when it shares files, such as "Shared
        Screenshot.png" or "Shared 3 files". None when it shares none."""
        if not self.references:
            return None
        if len(self.references) == 1:
            return f"Shared {self.references[0]['name']}"
        return f"Shared {len(self.references)} files"


def _upload_payload(upload: Processed) -> dict[str, Any]:
    """The stored row for a processed upload. PDFs and text files keep their
    original bytes, for attaching to a ticket later."""
    b64 = base64.b64encode(upload.data).decode()
    if upload.kind == "image":
        return {
            "kind": "image",
            "mime_type": upload.mime_type,
            "name": upload.name,
            "base64": b64,
            "width": upload.width,
            "height": upload.height,
        }
    if upload.kind == "pdf":
        return {
            "kind": "pdf",
            "name": upload.name,
            "base64": b64,
            "pages": upload.pages,
            "text": upload.text,
            "text_truncated": upload.text_truncated,
            "no_text": upload.no_text,
        }
    return {
        "kind": "text",
        "name": upload.name,
        "base64": b64,
        "text": upload.text,
        "text_truncated": upload.text_truncated,
    }


async def _share_files(thread_id: str, parsed: ParsedMessage) -> SharedFiles:
    """Check the message's references and store its new uploads.

    A reference counts only when it names an upload stored for this thread,
    and its block is rebuilt from that row, never taken from the client. A
    failure is a note, never an error. Cancelled, as a Stop during a slow PDF
    is, it deletes what it had stored for the message before re-raising.
    """
    store = get_tool_attachment_store()
    shared = SharedFiles(notes=list(parsed.dropped))
    for ref_id in parsed.ref_ids:
        payload = await asyncio.to_thread(store.get_payload, thread_id, ref_id)
        if payload is None:
            shared.notes.append(REF_NOT_FOUND)
        else:
            shared.references.append(upload_reference(ref_id, payload))

    # Files are processed side by side but written one at a time. Each write
    # runs to the end even when the run is cancelled, so its id is known and
    # the row can be deleted.
    puts: list[asyncio.Future[str | None]] = []
    writing = asyncio.Lock()

    async def _share(upload: IncomingUpload) -> dict[str, Any] | str:
        result = await process_upload(upload.kind, upload.mime_type, upload.name, upload.data)
        if isinstance(result, Rejected):
            return f"{result.name} couldn't be shared: it {result.reason}."
        payload = _upload_payload(result)
        async with writing:
            put = asyncio.ensure_future(
                asyncio.to_thread(store.put, thread_id, UPLOAD_KIND, payload)
            )
            puts.append(put)
            attachment_id = await asyncio.shield(put)
        if attachment_id is None:
            return f"{result.name} couldn't be shared: it couldn't be saved."
        return upload_reference(attachment_id, payload)

    try:
        results = await asyncio.gather(*(_share(u) for u in parsed.uploads))
    except BaseException:
        await asyncio.shield(asyncio.ensure_future(_discard(thread_id, puts)))
        raise
    for result in results:
        if isinstance(result, str):
            shared.notes.append(result)
        else:
            shared.references.append(result)
    return shared


async def _discard(thread_id: str, puts: list[asyncio.Future[str | None]]) -> None:
    """Delete the rows these writes stored. Never raises."""
    try:
        ids = {i for i in await asyncio.gather(*puts) if i}
        if ids:
            await asyncio.to_thread(get_tool_attachment_store().delete_ids, thread_id, ids)
    except Exception:
        logger.warning("Couldn't delete the uploads of a stopped run on %s", thread_id)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """The query the agent runs is the latest user turn in the run input.

    Checkpoint resume rebuilds prior context from the thread; the run body only
    needs to carry the new turn.

    A message is a user turn if it has ``role: "user"`` (the widget's shape)
    or ``type: "human"`` (the LangChain message shape the full-screen client
    sends). Its ``content`` may be a plain string or a block-array (the
    full-screen client sends ``[{"type": "text", "text": "..."}]``); only the
    text blocks are extracted.
    """
    return _extract_text_content(_last_user_content(messages))


@thread_router.post("/threads")
async def create_thread() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "thread_id": str(uuid.uuid4()),
        "created_at": now,
        "metadata": {},
        "status": "idle",
        "values": {},
    }


@thread_router.post("/threads/search")
async def search_threads(body: SearchRequest, identity: Identity) -> list[dict[str, Any]]:
    """The conversation sidebar: the caller's own threads, most recent first.

    Authed-only IN EFFECT, not by a 401 gate: a caller with no verified
    identity has no user_hash to match, so there is nothing to enumerate and
    the correct response is an empty list, not a 401 (unlike every other
    route in this module).
    """
    user_hash = _hash_user(identity.sub if identity else None)
    if not user_hash:
        return []
    rows = get_turn_reporter().list_threads_for_user(user_hash, body.limit)
    return [
        {
            "thread_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": {},
            "status": "idle",
            "interrupts": {},
            "values": {"messages": [{"type": "human", "content": row["label"]}]},
        }
        for row in rows
    ]


@thread_router.post("/threads/{thread_id}/history")
async def thread_history(
    thread_id: str,
    body: HistoryRequest,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> list[dict[str, Any]]:
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
    checkpointer = getattr(raw_request.app.state, "checkpointer", None)
    if checkpointer is None:
        return []
    # The read after a Stop shows the stopped turn, not the question alone.
    await stopped_turn_saved(thread_id)
    graph = create_checkpointed_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    states = []
    async for snap in graph.aget_state_history(config, limit=body.limit):
        states.append(
            {
                "values": _serialize_values(snap.values),
                "checkpoint": snap.config.get("configurable", {}),
                "parent_checkpoint": (snap.parent_config or {}).get("configurable", {}),
                "metadata": snap.metadata or {},
                "created_at": snap.created_at,
            }
        )
    return states


def _serialize_values(values: dict[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    msgs = values.get("messages", [])
    return {"messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in msgs]}


@thread_router.get("/threads/{thread_id}/attachments/{attachment_id}")
async def thread_attachment(
    thread_id: str,
    attachment_id: str,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> Response:
    """A stored tool attachment (src/tool_attachments.py), on the same access
    as reading the thread's history. An id stored under another thread is the
    same 404 as an unknown thread."""
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
    payload = await asyncio.to_thread(get_tool_attachment_store().get, thread_id, attachment_id)
    if payload is None:
        raise refuse_thread()
    # The thread's owner may reload it; nobody between may keep a copy.
    headers = {"Cache-Control": "private, no-store", "Vary": "Accept-Encoding"}
    if "gzip" in raw_request.headers.get("accept-encoding", ""):
        # A table's JSON is mostly repeated names and digits, so it shrinks to
        # about a quarter. The app has no compression middleware, which would
        # hold back the SSE run stream.
        body = await asyncio.to_thread(gzip.compress, payload.encode(), 6)
        headers["Content-Encoding"] = "gzip"
        return Response(body, media_type="application/json", headers=headers)
    return Response(payload, media_type="application/json", headers=headers)


@thread_router.post("/threads/{thread_id}/claim")
async def claim_thread_on_sign_in(
    thread_id: str, identity: Identity, claim_key: ClaimKey = None
) -> dict[str, str]:
    """Link a conversation started signed out to the caller who just signed in.

    Repeating this as the owner succeeds, which also retries a failed backfill.
    """
    require_thread_access(thread_id, require_identity(identity), claim_key, write=True)
    return {"thread_id": thread_id}


@thread_router.post("/threads/{thread_id}/handoff")
async def mint_handoff(thread_id: str, identity: Identity) -> dict[str, Any]:
    """Mint a single-use cross-surface handoff token for an authed-owned thread.

    Owners only: a handoff carries the thread into a URL, and an unclaimed
    thread has no owner for the exchange to bind to.
    """
    caller = require_identity(identity)
    require_thread_access(thread_id, caller, None)  # no key, so only the owner passes
    token = mint_handoff_token(thread_id, _hash_user(caller))  # type: ignore[arg-type]
    return {"token": token, "expires_in": DEFAULT_TTL_SECONDS}


@thread_router.post("/handoff/exchange")
async def exchange_handoff(body: ExchangeRequest, identity: Identity) -> dict[str, str]:
    """Redeem a single-use handoff token for a thread_id, identity-bound.

    The token is consumed atomically on exchange regardless of outcome; a
    mismatching caller burning someone else's token only harms an attacker.
    Returns ONLY thread_id — never message content (that comes from the
    separately-gated /history).
    """
    caller = require_identity(identity)
    result = exchange_handoff_token(body.token)
    if result is None:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    thread_id, owner_hash = result
    if owner_hash != _hash_user(caller):
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    return {"thread_id": thread_id}


# When this process last deleted expired uploads, on the monotonic clock.
_last_expiry: float | None = None
_EXPIRY_INTERVAL_S = 3600
_monotonic = time.monotonic


async def _expire_uploads() -> None:
    """Delete every thread's uploads past their retention, at most once an
    hour per process. Never raises; a failure waits for the next hour."""
    global _last_expiry
    now = _monotonic()
    if _last_expiry is not None and now - _last_expiry < _EXPIRY_INTERVAL_S:
        return
    _last_expiry = now
    older_than = datetime.now(UTC) - timedelta(days=settings.UPLOAD_RETENTION_DAYS)
    try:
        deleted = await asyncio.to_thread(
            get_tool_attachment_store().delete_expired_uploads, older_than
        )
        if deleted:
            logger.info("Deleted %d expired uploads", deleted)
    except Exception:
        logger.warning("Couldn't delete expired uploads", exc_info=True)


async def _sweep_attachments(thread_id: str, checkpointer: Any, keep: Iterable[str] = ()) -> None:
    """Delete this thread's stored attachments that its messages no longer
    refer to: a turn stopped and replaced, retried or edited, or folded into a
    summary by compaction. ``keep`` adds the references the run's own message
    resends, as a Retry does after its stopped turn is dropped. Never raises;
    whatever is missed goes next run.

    Runs at the start of a run, holding its thread lock, before the run stores
    anything: a run's own tables are never swept before their messages save,
    and no run is cut short by a cleanup after it. Expired uploads go first,
    from every thread."""
    await _expire_uploads()
    if checkpointer is None:
        return
    store = get_tool_attachment_store()
    try:
        if not await asyncio.to_thread(store.ids_for, thread_id):
            return
        await stopped_turn_saved(thread_id)
        if stopped_turn_saving(thread_id):
            return  # its turn may yet refer to one
        graph = create_checkpointed_graph(checkpointer)
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        kept = referenced_ids((state.values or {}).get("messages", [])) | set(keep)
        deleted = await asyncio.to_thread(store.keep_only, thread_id, kept)
        if deleted:
            logger.info("Deleted %d attachments no message refers to on %s", deleted, thread_id)
    except Exception:
        logger.warning("Couldn't sweep attachments on %s", thread_id, exc_info=True)


async def _sweep_then_share(
    thread_id: str, checkpointer: Any, parsed: ParsedMessage
) -> SharedFiles:
    """The sweep, then the message's files. In this order the sweep can't
    delete what this run stores, and it keeps the references the run resends."""
    await _sweep_attachments(thread_id, checkpointer, keep=parsed.ref_ids)
    return await _share_files(thread_id, parsed)


@dataclass
class _Turn:
    parsed: ParsedMessage
    message_id: str
    shares_files: bool


async def _read_turn(body: RunRequest, thread_id: str, caller: str | None) -> _Turn | JSONResponse:
    """The run's question, or the Turnstile challenge that has to come first.

    A message that shares files never rides the free-query allowance: an
    anonymous caller must have passed the challenge, which is the same
    response the allowance running out gets, so the client's restore and
    resend handles it. Then the message is decoded, once and off the event
    loop. Its files are only checked here; the run processes and stores them.
    """
    message = _last_user_message(body.input.messages)
    content = message.get("content", "")
    shares_files = _shares_files(content)
    challenge = await check_turnstile(
        caller,
        thread_id,
        body.turnstile_token or (body.context or {}).get("turnstile_token"),
        require_verified=shares_files,
    )
    if challenge is not None:
        return challenge
    parsed = await asyncio.to_thread(parse_message, content, anonymous=caller is None)
    return _Turn(parsed, _message_id(message), shares_files)


@thread_router.post("/threads/{thread_id}/runs/stream", response_model=None)
async def thread_run_stream(
    thread_id: str,
    body: RunRequest,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> StreamingResponse | JSONResponse:
    """Stream one agent turn on a thread as the LangGraph run protocol SSE.

    Ownership is atomic on the first run: ``require_thread_access`` writes the
    ownership row transactionally (UNIQUE(thread_id)), so one caller wins a new
    thread and an existing thread must pass the check. All status-code-bearing
    checks run before StreamingResponse is returned: the token's 401/503 in
    ``request_identity``, and 404/409 HERE. A status raised inside the generator
    would land after Starlette flushed 200.
    """
    # 1. Anon-accept: caller may be None; ownership below still fails closed.
    caller = identity.sub if identity else None

    # 2. Check access. For a new thread this also writes the ownership row.
    require_thread_access(thread_id, caller, claim_key, write=True, create=True)

    # Turnstile gate — ported from routes.py's /query (see src/turnstile.py).
    # Authed callers skip immediately inside check_turnstile. The thread_id is
    # the key the guard tracks free-query/verification state under, as
    # session_id is on /query.
    # Unlike /query, there is no "session_id required" 400 to mirror: thread_id
    # always exists (it's in the URL path), so the guard always has a key to
    # track. Placed AFTER the ownership gate and BEFORE acquire_thread_run so a
    # challenged request never acquires the run lock.
    turn = await _read_turn(body, thread_id, caller)
    if isinstance(turn, JSONResponse):
        return turn
    query = turn.parsed.text

    # Client-supplied UI hint, not server state: LangGraph SDK convention puts
    # per-run values under config.configurable; context is the newer top-level
    # alternative. Absent on both → None (accepted context-loss, per spec).
    cfg = (body.config or {}).get("configurable", {}) if body.config else {}
    resource_context = cfg.get("resource_context") or (body.context or {}).get("resource_context")

    # 3. + 4. Acquire the per-thread run lock in the handler so a busy thread is a
    #    real 409 status, not an event flushed after a 200.
    run_id = str(uuid.uuid4())
    acquire_thread_run(thread_id, run_id)
    # Same guard and key as /query, where session_id == thread_id, so a handed-off
    # conversation keeps its count. A capability question is answered from the
    # registry, not the model, so it is free here as it is on /query, unless
    # it shares files, which only the model can read.
    free = not turn.shares_files and is_capability_question(query)
    charge = NOT_CHARGED if free else charge_query(caller, thread_id)

    async def _gen() -> AsyncGenerator[str, None]:
        start_time = time.time()
        reset_turn_capture()
        final_state: dict[str, Any] = {}
        success = False

        async def _report(*, state: dict[str, Any], ok: bool) -> None:
            await _write_turn_report(
                final_state=state,
                thread_id=thread_id,
                run_id=run_id,
                query=query,
                start_time=start_time,
                caller=caller,
                success=ok,
            )

        # Built from the verified identity, never from the request body. Runs
        # while the metadata event (the client's run_id) goes out, and is
        # awaited before the turn's timer starts so it can't eat the turn
        # budget. Bounded and cached; None for anonymous runs and any failure.
        # ``caller`` is the token's subject, the same user AgentContext acts for.
        lookup = asyncio.ensure_future(allocated_profile_for(caller))
        async with held_run(thread_id, run_id):
            try:
                yield format_sse_event("metadata", {"run_id": run_id, "attempt": 1})
                profile = await lookup
                # The turn's timer starts inside, after the files are stored.
                async for event in _stream_run_events(
                    thread_id=thread_id,
                    run_id=run_id,
                    parsed=turn.parsed,
                    message_id=turn.message_id,
                    context=AgentContext.from_identity(identity, thread_run=True),
                    resource_context=resource_context,
                    profile=profile,
                    # The browser's zone, for event times. See _time_zone_from.
                    time_zone=_time_zone_from(body.context),
                    checkpointer=getattr(raw_request.app.state, "checkpointer", None),
                    final_state=final_state,
                    replace_stopped_turn=(body.context or {}).get("replace_stopped_turn") is True,
                ):
                    yield event
                success = True
            except asyncio.CancelledError:
                # Real cancel (client disconnect / cancel_run). Propagate so the
                # task truly cancels; held_run's finally still releases.
                # Cancelled after the answer, while follow-ups were being
                # suggested, the turn succeeded: it is reported as one and no
                # error is sent. Before the answer there is no report, as on
                # /query, which also re-raises on cancel without a failed-turn row;
                # _stream_run_events saves what had streamed as a stopped answer.
                if not _report_if_delivered(final_state, _report):
                    yield format_sse_event("error", {"error": "cancelled"})
                raise
            except TimeoutError:
                # asyncio.timeout expiry (a TimeoutError in 3.11, NOT CancelledError).
                # Emit and return; do not synthesize a CancelledError.
                charge.refund()
                yield format_sse_event("error", {"error": "timeout"})
                await _report(state=final_state, ok=False)
                return
            except Exception as exc:
                logger.exception("Run stream failed: %s", exc)
                charge.refund()
                yield format_sse_event("error", {"error": "agent_error"})
                await _report(state=final_state, ok=False)
                return
            finally:
                # A disconnect before the lookup finished. A no-op once it has.
                lookup.cancel()

        # Turn report write lives OUTSIDE both run_with_timeout and held_run — a
        # slow reporter write must not trip a spurious TimeoutError into an
        # already-successful stream, as on /query.
        if success:
            await _report(state=final_state, ok=True)

    # The lock was acquired synchronously above; it is released by held_run's
    # finally, but ONLY once the generator is actually started. An async
    # generator that is never iterated never runs its finally (verified against
    # CPython/Starlette 1.0: GC of an un-started async gen does not run finally),
    # so any failure on the synchronous path between acquire and returning a
    # started response would leak the lock and 409-wedge the thread until
    # restart. Guard that gap: release on any exception before hand-off.
    try:
        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except BaseException:
        release_thread_run(thread_id, run_id)
        charge.refund()
        raise


@thread_router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_thread_run(
    thread_id: str, run_id: str, identity: Identity, claim_key: ClaimKey = None
) -> dict[str, str]:
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
    if not cancel_run(thread_id, run_id):  # no such in-flight run ON THIS THREAD
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "cancelled"}
