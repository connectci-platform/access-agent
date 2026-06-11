from src.services.resource_matcher import match_resources
from src.services.rp_cache import RPInfo


def _groups():
    return [
        RPInfo(slug="delta", title="Delta"),
        RPInfo(slug="bridges-2", title="Bridges-2"),
        RPInfo(slug="jetstream2", title="Jetstream2"),
        RPInfo(slug="anvil", title="Anvil"),
    ]


class TestMatchResources:
    def test_matches_title_case_insensitive(self):
        assert match_resources("how do I log into DELTA?", _groups()) == ["delta"]

    def test_hyphen_space_variants_match(self):
        # bridges2 / bridges 2 / Bridges-2 are all the same resource
        assert match_resources("is bridges2 down?", _groups()) == ["bridges-2"]
        assert match_resources("is bridges 2 down?", _groups()) == ["bridges-2"]
        assert match_resources("is Bridges-2 down?", _groups()) == ["bridges-2"]

    def test_separator_inserted_into_compact_names(self):
        # title "Jetstream2" must still match "Jetstream 2" in text
        assert match_resources("what is Jetstream 2?", _groups()) == ["jetstream2"]

    def test_word_boundaries_prevent_substring_hits(self):
        assert match_resources("see anvilteam.org for details", _groups()) == []
        assert match_resources("the deltas of the values", _groups()) == []

    def test_multiple_mentions_dedupe_and_sort(self):
        text = "Should I use Delta or Anvil? Delta has GPUs."
        assert match_resources(text, _groups()) == ["anvil", "delta"]

    def test_empty_inputs(self):
        assert match_resources("", _groups()) == []
        assert match_resources("anything about Delta", []) == []
