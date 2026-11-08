"""The one access check for anything that reads or writes a thread.

A thread id alone grants nothing: a fullscreen id sits in the page URL. Access
needs the thread's owner, or the claim key from the browser that started it.
Every refusal is the same 404 as an unknown thread, never 403, so a caller can't
tell which threads exist.
"""

from fastapi import HTTPException

from .. import turn_reporter
from ..thread_owners import (
    ThreadAccess,
    get_thread_owner_store,
    is_fullscreen_thread_id,
    is_widget_session,
)


def refuse_thread() -> HTTPException:
    return HTTPException(status_code=404, detail="Thread not found")  # never 403


def _query_may_continue(thread_id: str, caller: str | None) -> bool:
    """/query's exception for threads with no key on record.

    The widget and non-browser callers (redteam, API clients) use /query with
    ids of their own, never in a URL. Bare uuids are refused because those are
    fullscreen ids. Only the widget has a signed-in flip, so an exact widget id
    is the only one a signed-in caller can take.
    """
    if caller:
        return is_widget_session(thread_id)
    return not is_fullscreen_thread_id(thread_id)


def _link_to_caller(
    thread_id: str, caller: str, claim_key: str | None, access: ThreadAccess
) -> None:
    """Give a signed-in writer ownership, then surface the thread in their sidebar.

    Backfill has no ownership check of its own, so it runs only for the owner or
    the caller who won the flip: two callers can both reach this point, and a
    loser who backfilled would tag the winner's turns with their own hash. For
    an owner the sweep is idempotent, and repairs a flip whose backfill failed.
    """
    store = get_thread_owner_store()
    if access is ThreadAccess.KEY_HOLDER:
        flipped = store.claim_with_key(thread_id, caller, claim_key)
    elif access is ThreadAccess.KEYLESS:
        flipped = store.upgrade_widget_session(thread_id, caller)
    else:
        flipped = False
    if not flipped and not store.owned_by(thread_id, caller):
        raise refuse_thread()
    turn_reporter.get_turn_reporter().backfill_user_hash(thread_id, caller)


def _allowed(access: ThreadAccess, thread_id: str, caller: str | None, *, via_query: bool) -> bool:
    if access in (ThreadAccess.OWNER, ThreadAccess.KEY_HOLDER):
        return True
    return via_query and access is ThreadAccess.KEYLESS and _query_may_continue(thread_id, caller)


def require_thread_access(
    thread_id: str,
    caller: str | None,
    claim_key: str | None,
    *,
    write: bool = False,
    create: bool = False,
    via_query: bool = False,
) -> None:
    """Pass, or raise the unknown-thread 404.

    ``write`` links the thread to a signed-in caller. ``create`` lets a caller
    make a new thread, owning it from this moment. ``via_query`` applies
    /query's exception.
    """
    store = get_thread_owner_store()
    access = store.access(thread_id, caller, claim_key)
    if create and access is ThreadAccess.NEW:
        if store.claim_thread(thread_id, caller, claim_key):
            return
        # The row is the only arbiter across workers: someone else may have got
        # there first, so judge this caller against what's there now.
        access = store.access(thread_id, caller, claim_key)
        if access is ThreadAccess.NEW:
            return  # no database, so there's nothing to own
    if not _allowed(access, thread_id, caller, via_query=via_query):
        raise refuse_thread()
    if write and caller:
        _link_to_caller(thread_id, caller, claim_key, access)
