"""LangGraph thread/run protocol endpoints: create + history (resume).

Both routes sit behind auth (SESSaccess_auth cookie) and the thread-ownership
gate from src/thread_owners.py. Non-owner and unknown-thread requests return
an identical 404 body so thread existence is not observable to a caller who
doesn't own it.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..agent.graph import create_checkpointed_graph
from ..auth import get_acting_user_from_cookie
from ..thread_owners import get_thread_owner_store

thread_router = APIRouter()


def _require_user(raw_request: Request) -> str:
    user, _ = get_acting_user_from_cookie(raw_request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _require_access(thread_id: str, caller: str) -> None:
    if not get_thread_owner_store().check_access(thread_id, caller):
        raise HTTPException(status_code=404, detail="Thread not found")  # never 403


class HistoryRequest(BaseModel):
    limit: int = 10
    before: str | None = None


@thread_router.post("/threads")
async def create_thread(raw_request: Request) -> dict[str, Any]:
    _require_user(raw_request)
    now = datetime.now(UTC).isoformat()
    return {
        "thread_id": str(uuid.uuid4()),
        "created_at": now,
        "metadata": {},
        "status": "idle",
        "values": {},
    }


@thread_router.post("/threads/{thread_id}/history")
async def thread_history(
    thread_id: str, body: HistoryRequest, raw_request: Request
) -> list[dict[str, Any]]:
    caller = _require_user(raw_request)
    _require_access(thread_id, caller)
    checkpointer = getattr(raw_request.app.state, "checkpointer", None)
    if checkpointer is None:
        return []
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
