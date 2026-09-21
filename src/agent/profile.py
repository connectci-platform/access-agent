"""Per-request user profile — a request-boundary contract.

Imported by the API layer, the agent, and the eval CLI, so it lives here
rather than in ``state.py`` (the graph's internal state, and the wrong
owner of a contract three other layers share).

Facts are phrased as what the *request supplied*, never as assertions about
the user — this is an ungated body value, so the agent cannot vouch for it.
"""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

RP_SLUG_RE = r"^[a-z0-9]+$"
NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9 .\-_/()]*$"
RESOURCE_ID_RE = r"^[A-Za-z0-9.\-]+$"


class ProfileFragment(NamedTuple):
    """One field's contribution to the rendered profile section."""

    fact: str | None
    instruction: str | None
    # User-facing capability sentence; not rendered into the prompt. Consumed
    # by a future capabilities composer (see render_allocated_resources).
    hint: str | None = None


class AllocatedResource(BaseModel):
    """One resource the caller reports the user holds an allocation on.

    Three parts because users hold credits on *variants* (Delta GPU) while
    UKY's rp_name vocabulary is *groups*, and the mapping is lossy — many
    variants belong to no group at all. ``resource_id`` is identity only and
    must never be rendered as an rp_name (that reconstructs the
    ``_normalize_rp_name`` mangling trap in access_documents.py).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., max_length=64, pattern=NAME_RE)
    rp_slug: str | None = Field(None, max_length=32, pattern=RP_SLUG_RE)
    resource_id: str | None = Field(None, max_length=128, pattern=RESOURCE_ID_RE)


class UserProfile(BaseModel):
    """Optional per-request profile hint. Steers retrieval, authorizes nothing."""

    model_config = ConfigDict(extra="forbid")

    # None means "not supplied" — distinct from an explicitly empty list, which
    # still renders the "none supplied" fact. See render_allocated_resources.
    allocated_resources: list[AllocatedResource] | None = Field(None, max_length=32)


def _fact_name(resource: AllocatedResource) -> str:
    if resource.rp_slug:
        return f"{resource.name} (rp_name='{resource.rp_slug}')"
    return resource.name


def render_allocated_resources(resources: list[AllocatedResource]) -> ProfileFragment:
    """Render the allocated-resources fact + instruction + hint.

    Facts state what the *request supplied*, never something about the user,
    so anonymous callers stay truthful. The instruction composes clauses
    (slug-scoping, ungrouped, empty) rather than picking one branch, because
    real profiles mix a grouped variant with ungrouped storage.
    """
    if not resources:
        return ProfileFragment(
            fact="Allocated resources (as supplied): none supplied with this request",
            instruction=(
                'If the user asks about "this system" or their own cluster, '
                "ask which resource they mean rather than assuming one."
            ),
            hint=None,
        )

    fact = "Allocated resources (as supplied): " + ", ".join(_fact_name(r) for r in resources)

    slugs = {r.rp_slug for r in resources if r.rp_slug}
    ungrouped = [r for r in resources if not r.rp_slug]

    clauses: list[str] = []
    if len(slugs) == 1:
        (slug,) = slugs
        clauses.append(
            "When the user asks about *this* system, cluster, or machine — "
            f'or says "here" — they mean `{slug}` unless the question names '
            f"another resource; pass `rp_name='{slug}'` to "
            "`search_access_documents` and to any MCP tool that accepts a "
            "resource filter."
        )
        clauses.append(
            f"When you answer for `{slug}`, say so, and attribute "
            "resource-specific values (commands, purge windows, quotas, "
            f"partitions) to `{slug}` rather than to ACCESS generally."
        )
    elif len(slugs) >= 2:
        clauses.append(
            'Do not guess which one "this system" means: either answer for '
            "each of them or ask the user which resource they mean."
        )

    if ungrouped:
        names = ", ".join(r.name for r in ungrouped)
        clauses.append(
            f"{names} have no scoped documentation; answer about them from general ACCESS docs."
        )

    instruction = " ".join(clauses)

    hint_names = ", ".join(r.name for r in resources)
    hint = (
        f"You hold allocations on {hint_names}, so you can ask about your "
        "cluster and I'll answer for your resources."
    )

    return ProfileFragment(fact=fact, instruction=instruction, hint=hint)


_CLOSING_SENTENCES = (
    "When a question is clearly cross-resource or general-process "
    "(allocations policy, how ACCESS works, choosing a resource), omit the "
    "resource scope. If a resource context is set above, that resource wins "
    'for "this system" even if it is not listed here.'
)

_RESOURCE_CONTEXT_OVERRIDE_INSTRUCTION = (
    'A resource context is set above; that resource is what "this system" '
    "means. The allocations listed here are for questions about the user's "
    "own allocations."
)


def render_profile_section(profile: UserProfile, resource_context: str | None = None) -> str:
    """Compose the `## User profile` prompt section from per-field fragments.

    Empty profile (every field None) renders "" — no header, no closing
    sentences — so build_system_prompt appends nothing for a bare profile.

    A later sibling field is one more fragment appended to the same list —
    the facts-block + instructions-block composition and the closing
    sentences don't change shape to accommodate it.

    ``resource_context``, when set, overrides the allocated-resources
    fragment's instruction: without this, the profile's single-slug clause
    ("they mean `delta`") and the `## Resource context` section's own
    "this resource" claim contradict each other, with the tie-break buried in
    the last sentence of the profile block. The override collapses that to
    one unambiguous sentence and drops the now-unneeded precedence mention
    from the closing sentences.

    An empty ``allocated_resources`` list (``[]``, not ``None``) combined
    with ``resource_context`` is a distinct case: the fact line still states
    "none supplied", but the override sentence ("The allocations listed here
    are...") would refer to allocations that don't exist, and the ordinary
    "ask which resource" instruction is moot because ``resource_context``
    already answers that question. So this case renders the fact with an
    empty instructions block — no override, no "ask which resource", no
    closing sentences.
    """
    fragments: list[ProfileFragment] = []
    if profile.allocated_resources is not None:
        fragments.append(render_allocated_resources(profile.allocated_resources))

    facts: list[str] = []
    instructions: list[str] = []
    for fragment in fragments:
        if fragment.fact:
            facts.append(fragment.fact)
        if fragment.instruction and not resource_context:
            instructions.append(fragment.instruction)

    if not facts and not instructions:
        return ""

    # resource_context wins for "this system" — replace every fragment's own
    # scoping instruction with the one override sentence instead of letting
    # each fragment repeat (or contradict) the same claim. Gated on there
    # being at least one actual resource, not just a fact line: an empty
    # allocated_resources list still produces a "none supplied" fact, and the
    # override sentence would then dangle ("the allocations listed here")
    # with nothing listed.
    if resource_context and profile.allocated_resources:
        instructions = [_RESOURCE_CONTEXT_OVERRIDE_INSTRUCTION]

    lines = ["## User profile", "", "Supplied with this request:"]
    lines.extend(f"- {fact}" for fact in facts)
    lines.append("")
    lines.extend(instructions)
    if not resource_context:
        lines.append(_CLOSING_SENTENCES)
    return "\n".join(lines)
