"""Tests for profile rendering — pure functions, no prompt assembly."""

from src.agent.profile import AllocatedResource, render_allocated_resources, render_profile_section

DELTA_GPU = AllocatedResource(name="Delta GPU", rp_slug="delta")
DELTA_STORAGE = AllocatedResource(name="Delta Storage")
BRIDGES2_OCEAN = AllocatedResource(name="Bridges-2 Ocean")
JETSTREAM2_STORAGE = AllocatedResource(name="Jetstream2 Storage")


def test_render_allocated_resources_zero():
    fragment = render_allocated_resources([])
    assert fragment.fact == "Allocated resources (as supplied): none supplied with this request"
    assert "ask which resource" in fragment.instruction
    assert fragment.hint is None


def test_render_allocated_resources_one():
    fragment = render_allocated_resources([DELTA_GPU])
    assert "Delta GPU" in fragment.fact
    assert "rp_name='delta'" in fragment.fact
    assert "rp_name='delta'" in fragment.instruction
    assert "search_access_documents" in fragment.instruction


def test_render_allocated_resources_many_distinct_slugs():
    bridges = AllocatedResource(name="Bridges-2 RM", rp_slug="bridges2")
    fragment = render_allocated_resources([DELTA_GPU, bridges])
    assert "Delta GPU" in fragment.fact
    assert "rp_name='delta'" in fragment.fact
    assert "Bridges-2 RM" in fragment.fact
    assert "rp_name='bridges2'" in fragment.fact
    assert "do not guess" in fragment.instruction.lower()
    assert "rp_name=" not in fragment.instruction


def test_render_shared_slug_collapses_to_single_instruction():
    delta_storage_grouped = AllocatedResource(name="Delta Storage", rp_slug="delta")
    fragment = render_allocated_resources([DELTA_GPU, delta_storage_grouped])
    assert "Delta GPU" in fragment.fact
    assert "Delta Storage" in fragment.fact
    assert "rp_name='delta'" in fragment.instruction
    assert "do not guess" not in fragment.instruction.lower()


def test_render_resource_without_slug_omits_rp_name():
    fragment = render_allocated_resources([DELTA_STORAGE])
    assert "Delta Storage" in fragment.fact
    assert "(rp_name=" not in fragment.fact


def test_render_all_resources_ungrouped_says_scoped_docs_unavailable():
    fragment = render_allocated_resources([DELTA_STORAGE, BRIDGES2_OCEAN])
    assert "Delta Storage" in fragment.instruction
    assert "Bridges-2 Ocean" in fragment.instruction
    assert "no scoped documentation" in fragment.instruction
    assert "rp_name=" not in fragment.instruction


def test_render_mixed_grouped_and_ungrouped_composes_both_clauses():
    fragment = render_allocated_resources([DELTA_GPU, DELTA_STORAGE, BRIDGES2_OCEAN])
    assert "rp_name='delta'" in fragment.instruction
    assert "Delta Storage" in fragment.instruction
    assert "Bridges-2 Ocean" in fragment.instruction


def test_render_single_slug_includes_answer_framing():
    single = render_allocated_resources([DELTA_GPU])
    assert "say so" in single.instruction
    assert "attribute resource-specific values" in single.instruction

    bridges = AllocatedResource(name="Bridges-2 RM", rp_slug="bridges2")
    multi = render_allocated_resources([DELTA_GPU, bridges])
    assert "say so" not in multi.instruction


def test_render_never_emits_resource_id():
    resource = AllocatedResource(
        name="Delta GPU", rp_slug="delta", resource_id="delta-gpu.ncsa.access-ci.org"
    )
    fragment = render_allocated_resources([resource])
    assert "delta-gpu.ncsa.access-ci.org" not in fragment.fact
    assert "delta-gpu.ncsa.access-ci.org" not in fragment.instruction


def test_render_hint_names_resources():
    fragment = render_allocated_resources([DELTA_GPU, DELTA_STORAGE])
    assert "Delta GPU" in fragment.hint
    assert "Delta Storage" in fragment.hint


def test_render_hint_is_none_for_empty_list():
    fragment = render_allocated_resources([])
    assert fragment.hint is None


def test_render_profile_section_facts_precede_instructions():
    from src.agent.profile import UserProfile

    profile = UserProfile(allocated_resources=[DELTA_GPU])
    rendered = render_profile_section(profile)
    facts_idx = rendered.index("Supplied with this request:")
    instruction_idx = rendered.index("When the user asks about")
    assert facts_idx < instruction_idx


def test_render_profile_section_emits_closing_sentences_exactly_once():
    from src.agent.profile import UserProfile

    profile = UserProfile(allocated_resources=[DELTA_GPU, DELTA_STORAGE, BRIDGES2_OCEAN])
    rendered = render_profile_section(profile)
    closing_a = (
        "When a question is clearly cross-resource or general-process "
        "(allocations policy, how ACCESS works, choosing a resource), "
        "omit the resource scope."
    )
    closing_b = (
        "If a resource context is set above, that resource wins for "
        '"this system" even if it is not listed here.'
    )
    assert rendered.count(closing_a) == 1
    assert rendered.count(closing_b) == 1


def test_render_profile_section_returns_empty_string_when_no_fragments():
    from src.agent.profile import UserProfile

    profile = UserProfile()
    assert render_profile_section(profile) == ""


def test_render_profile_section_within_token_budget():
    from src.agent.profile import UserProfile

    single_slug = UserProfile(allocated_resources=[DELTA_GPU])
    mixed_worst_case = UserProfile(
        allocated_resources=[DELTA_GPU, DELTA_STORAGE, BRIDGES2_OCEAN, JETSTREAM2_STORAGE]
    )
    for profile in (single_slug, mixed_worst_case):
        rendered = render_profile_section(profile)
        assert len(rendered) / 4 <= 250


def test_render_profile_section_resource_context_overrides_slug_instruction():
    """resource_context wins for "this system"; the profile's own single-slug
    claim ("they mean `delta`") must not survive alongside it — one
    unambiguous sentence, not a contradiction the model has to arbitrate."""
    from src.agent.profile import UserProfile

    profile = UserProfile(allocated_resources=[DELTA_GPU])
    rendered = render_profile_section(profile, resource_context="anvil")

    assert "they mean `delta`" not in rendered
    assert 'that resource is what "this system" means' in rendered
    assert "Delta GPU" in rendered


def test_render_profile_section_resource_context_omits_closing_precedence_sentence():
    """The closing precedence sentence is now redundant with the override
    instruction — must not appear twice (once inline, once in the closer)."""
    from src.agent.profile import UserProfile

    profile = UserProfile(allocated_resources=[DELTA_GPU])
    rendered = render_profile_section(profile, resource_context="anvil")

    assert 'that resource wins for "this system"' not in rendered


def test_render_profile_section_empty_resources_with_resource_context_has_no_dangling_override():
    """UserProfile(allocated_resources=[]) + resource_context: the fact line
    still states none-supplied, but the override sentence must not render —
    it would claim "the allocations listed here" with nothing listed. The
    ordinary "ask which resource" instruction is also moot: resource_context
    already answers that question."""
    from src.agent.profile import UserProfile

    profile = UserProfile(allocated_resources=[])
    rendered = render_profile_section(profile, resource_context="anvil")

    assert "Allocated resources (as supplied): none supplied with this request" in rendered
    assert "allocations listed here" not in rendered
    assert "ask which resource" not in rendered


def test_render_profile_section_none_field_with_resource_context_renders_nothing():
    """UserProfile() (field absent, not empty) + resource_context: still no
    fragment contributed, so the section renders nothing at all."""
    from src.agent.profile import UserProfile

    assert render_profile_section(UserProfile(), resource_context="delta") == ""
