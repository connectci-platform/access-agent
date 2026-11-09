"""Per-run identity for the agent, passed as LangGraph runtime context.

Graph state is the conversation, checkpointed and resumed. Who is calling,
and what they may do, is a fact about one run, so it travels as runtime
context: passed on every invocation and never written to a checkpoint, its
metadata or tracing. ``config["configurable"]`` would not do — LangGraph
copies primitive configurable values into checkpoint metadata, which
``/history`` returns to the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import settings

if TYPE_CHECKING:
    from ..agent_token import AgentIdentity

WRITE_SCOPE = "agent:write"


@dataclass(frozen=True)
class AgentContext:
    """Who the run acts for, and what that caller may do."""

    acting_user: str | None
    scopes: frozenset[str] = frozenset()
    # A run on the full-screen client's thread route, which shows follow-up
    # suggestions under each answer. The widget's /query and the eval leave it
    # False, so their runs are unchanged.
    thread_run: bool = False

    def __post_init__(self) -> None:
        # LangGraph builds the dataclass from a dict context without checking
        # field types, so a list of scopes or a non-string user would otherwise
        # pass require_context's isinstance check.
        if self.acting_user is not None and not isinstance(self.acting_user, str):
            raise TypeError("AgentContext.acting_user must be a str or None")
        if not isinstance(self.scopes, frozenset) or not all(
            isinstance(s, str) for s in self.scopes
        ):
            raise TypeError("AgentContext.scopes must be a frozenset of str")
        if not isinstance(self.thread_run, bool):
            raise TypeError("AgentContext.thread_run must be a bool")

    @property
    def can_write(self) -> bool:
        """True when the caller is signed in, holds ``agent:write``, and the
        deployment isn't ``READ_ONLY``."""
        return (
            self.acting_user is not None and WRITE_SCOPE in self.scopes and not settings.READ_ONLY
        )

    @classmethod
    def from_identity(
        cls, identity: AgentIdentity | None, *, thread_run: bool = False
    ) -> AgentContext:
        """The context for a request, from its verified token only."""
        if identity is None:
            return cls(acting_user=None, thread_run=thread_run)
        return cls(acting_user=identity.sub, scopes=identity.scopes, thread_run=thread_run)

    @classmethod
    def anonymous(cls) -> AgentContext:
        return cls(acting_user=None)

    @classmethod
    def for_eval(cls, acting_user: str | None, *, allow_writes: bool) -> AgentContext:
        """An eval run acting for ``acting_user``, which must say whether it
        may make changes. Until the eval mints real tokens, the scopes stand
        in for what a token would carry."""
        if acting_user is None:
            return cls.anonymous()
        scopes = {"agent:read", WRITE_SCOPE} if allow_writes else {"agent:read"}
        return cls(acting_user=acting_user, scopes=frozenset(scopes))


def require_context(context: Any) -> AgentContext:
    """Fail closed: a run without an ``AgentContext`` is an error, not anonymous.

    LangGraph coerces a dict into the dataclass but passes any other value
    through unchecked, so ``None`` and junk both arrive here.
    """
    if not isinstance(context, AgentContext):
        raise TypeError(f"agent run needs an AgentContext, got {type(context).__name__}")
    return context
