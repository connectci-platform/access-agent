"""Answer ratings from the full-screen client (D8-3022).

A rating names its turn by the thread and the answer's message id, the one id
the client sees for it. Access is require_thread_access, as for reading the
thread, so rating a turn takes the same standing as seeing it. The answer has
to be a turn's final answer in the thread's checkpoint, which is what the
client shows and what the turn report records.

Ratings live in turn_ratings and the turn report, never in the thread's state,
so the model never reads them. Feedback text is never logged.
"""

import asyncio
import unicodedata
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agent.graph import create_checkpointed_graph
from ..turn_reporter import get_turn_reporter, turn_answer_ids
from .identity import ClaimKey, Identity
from .thread_access import refuse_thread, require_thread_access

rating_router = APIRouter()

# Kept, after cleaning. The client stops at the same length.
MAX_FEEDBACK_CHARS = 500
# Refused outright: no real comment is this long.
MAX_FEEDBACK_BODY_CHARS = 2000

MessageId = Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


def clean_feedback(raw: str | None) -> str | None:
    """Plain text a reviewer can read safely, capped, or None for nothing.

    Drops control and formatting characters (NULs, bidi overrides, zero-width
    marks) but keeps newlines and tabs.
    """
    if not raw:
        return None
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    kept = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    return kept.strip()[:MAX_FEEDBACK_CHARS].strip() or None


class AnswerRatingRequest(BaseModel):
    rating: Literal["helpful", "not_helpful"]
    feedback: str | None = Field(None, max_length=MAX_FEEDBACK_BODY_CHARS)


async def _is_turn_answer(request: Request, thread_id: str, message_id: str) -> bool:
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None:
        return False
    graph = create_checkpointed_graph(checkpointer)
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = (snapshot.values or {}).get("messages")
    return message_id in turn_answer_ids(messages)


@rating_router.put("/threads/{thread_id}/messages/{message_id}/rating")
async def rate_answer(
    thread_id: str,
    message_id: MessageId,
    body: AnswerRatingRequest,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> dict[str, Any]:
    """Set the caller's rating of one answer, replacing any earlier one.

    No write flag: rating doesn't link the thread to a signed-in caller.
    """
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
    if not await _is_turn_answer(raw_request, thread_id, message_id):
        raise refuse_thread()
    feedback = clean_feedback(body.feedback) if body.rating == "not_helpful" else None
    saved = await asyncio.to_thread(
        get_turn_reporter().rate_answer, thread_id, message_id, body.rating, feedback
    )
    if not saved:
        raise HTTPException(status_code=503, detail="Rating is unavailable")
    return {"message_id": message_id, "rating": body.rating}


@rating_router.get("/threads/{thread_id}/ratings")
async def list_ratings(
    thread_id: str, identity: Identity, claim_key: ClaimKey = None
) -> JSONResponse:
    """The thread's ratings by answer id, so a reload shows what was chosen.

    Ratings only: feedback stays with the team.
    """
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
    ratings = await asyncio.to_thread(get_turn_reporter().ratings_for_thread, thread_id)
    if ratings is None:
        raise HTTPException(status_code=503, detail="Rating is unavailable")
    return JSONResponse(
        content={"ratings": ratings},
        headers={"Cache-Control": "private, no-store"},
    )
