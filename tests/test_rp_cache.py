"""Tests for the RP section cache."""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.rp_cache import RPInfo, RPSectionCache

# ── Cache fetch ──────────────────────────────────────────────────────────


class TestRPCacheFetch:
    """Verify the cache builds correctly from mocked Drupal responses.

    The Drupal `/api/resource-groups` endpoint returns one entry per
    resource_group node with the slug, title, and pre-aggregated
    populated_sections (union across member CIDER variants). The cache
    just consumes that response — no slug extraction or merge logic
    happens client-side.
    """

    def _build_group(self, slug, title, populated_sections=None, variants=None):
        return {
            "nid": 1,
            "slug": slug,
            "title": title,
            "url": f"http://example.test/rp-documentation/{slug}",
            "populated_sections": populated_sections or [],
            "variants": variants or [],
        }

    def _mock_drupal_response(self, groups):
        """Build a mocked httpx response with the given groups payload."""
        mock_response = AsyncMock()
        mock_response.json = lambda: {"count": len(groups), "groups": groups}
        mock_response.raise_for_status = lambda: None
        return mock_response

    async def _run_fetch(self, groups):
        cache = RPSectionCache()
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.return_value = self._mock_drupal_response(groups)
            return await cache._fetch_resources()

    @pytest.mark.asyncio
    async def test_basic_fetch(self):
        result = await self._run_fetch(
            [
                self._build_group(
                    "anvil",
                    "Anvil",
                    populated_sections=["login", "storage", "queue_specs"],
                ),
            ]
        )

        assert "anvil" in result
        rp = result["anvil"]
        assert rp.slug == "anvil"
        assert rp.title == "Anvil"
        assert rp.populated_sections == ["login", "storage", "queue_specs"]

    @pytest.mark.asyncio
    async def test_multiple_groups(self):
        result = await self._run_fetch(
            [
                self._build_group("anvil", "Anvil", ["login"]),
                self._build_group("delta", "Delta", ["login", "top_software"]),
                self._build_group("bridges-2", "Bridges-2", []),
            ]
        )

        assert set(result.keys()) == {"anvil", "delta", "bridges-2"}
        assert result["bridges-2"].title == "Bridges-2"
        assert result["delta"].populated_sections == ["login", "top_software"]

    @pytest.mark.asyncio
    async def test_group_with_no_sections_still_cached(self):
        """A group with no populated sections is still a valid cached slug.

        The capabilities response shows support + analytics for any known
        slug, even if no documentation sections are populated.
        """
        result = await self._run_fetch(
            [
                self._build_group("kyric", "KyRIC", []),
            ]
        )

        assert "kyric" in result
        assert result["kyric"].populated_sections == []

    @pytest.mark.asyncio
    async def test_skips_groups_with_missing_slug(self):
        result = await self._run_fetch(
            [
                {"title": "No slug", "populated_sections": []},
                self._build_group("good", "Good"),
            ]
        )

        assert "good" in result
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_response(self):
        result = await self._run_fetch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_hyphenated_slug_passes_through(self):
        """Verify hyphenated slugs (e.g. 'bridges-2') are preserved.

        These slugs come from Html::getClass() on the Drupal side and
        match exactly what the embedded bot's data-resource-context
        attribute reports.
        """
        result = await self._run_fetch(
            [
                self._build_group("bridges-2", "Bridges-2", ["login"]),
            ]
        )

        assert "bridges-2" in result
        assert result["bridges-2"].title == "Bridges-2"


# ── Cache lifecycle ──────────────────────────────────────────────────────


class TestRPCacheLifecycle:
    """Cache TTL and lookup behavior."""

    def test_get_returns_none_for_unknown_slug(self):
        cache = RPSectionCache()
        cache._cache = {"anvil": RPInfo(slug="anvil", title="Anvil")}
        assert cache.get("unknown") is None

    def test_get_returns_rpinfo_for_known_slug(self):
        cache = RPSectionCache()
        cache._cache = {"anvil": RPInfo(slug="anvil", title="Anvil", populated_sections=["login"])}
        rp = cache.get("anvil")
        assert rp is not None
        assert rp.slug == "anvil"
        assert rp.populated_sections == ["login"]

    def test_is_valid_slug(self):
        cache = RPSectionCache()
        cache._cache = {"anvil": RPInfo(slug="anvil", title="Anvil")}
        assert cache.is_valid_slug("anvil") is True
        assert cache.is_valid_slug("unknown") is False

    def test_list_slugs(self):
        cache = RPSectionCache()
        cache._cache = {
            "anvil": RPInfo(slug="anvil", title="Anvil"),
            "delta": RPInfo(slug="delta", title="Delta"),
        }
        assert set(cache.list_slugs()) == {"anvil", "delta"}


# ── Variant resource ids (D8-3028) ───────────────────────────────────────


class TestRPCacheResourceIds:
    """Each group keeps its member variants' ACCESS global resource ids, so a
    resource a user holds an allocation on maps to its group exactly, never by
    name."""

    async def test_fetch_keeps_variant_global_resource_ids(self):
        variants = [
            {"title": "Delta CPU", "global_resource_id": "delta-cpu.ncsa.access-ci.org"},
            {"title": "Delta GPU", "global_resource_id": "delta-gpu.ncsa.access-ci.org"},
            {"title": "No id", "global_resource_id": None},
            {"title": "Blank id", "global_resource_id": ""},
            "not-a-dict",
        ]
        result = await TestRPCacheFetch()._run_fetch(
            [TestRPCacheFetch()._build_group("delta", "Delta", variants=variants)]
        )

        assert result["delta"].resource_ids == [
            "delta-cpu.ncsa.access-ci.org",
            "delta-gpu.ncsa.access-ci.org",
        ]

    async def test_fetch_tolerates_a_non_list_variants_field(self):
        group = TestRPCacheFetch()._build_group("fabric", "FABRIC")
        group["variants"] = None
        result = await TestRPCacheFetch()._run_fetch([group])

        assert result["fabric"].resource_ids == []

    def test_slugs_for_resource_id(self):
        cache = RPSectionCache()
        cache._cache = {
            "delta": RPInfo(slug="delta", title="Delta", resource_ids=["delta-gpu.x"]),
            "anvil": RPInfo(slug="anvil", title="Anvil", resource_ids=["anvil.x"]),
            "other": RPInfo(slug="other", title="Other", resource_ids=["anvil.x"]),
        }

        assert cache.slugs_for_resource_id("delta-gpu.x") == ["delta"]
        assert cache.slugs_for_resource_id("anvil.x") == ["anvil", "other"]
        assert cache.slugs_for_resource_id("unknown.x") == []
