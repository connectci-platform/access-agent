"""Tests for the RP section cache."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
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


# ── Bounded, backed-off, background refresh (D8-3028 review) ─────────────


class TestRPCacheRefreshBounds:
    """A slow or wrong groups URL must not stall requests, refetch on every
    miss, or log a traceback each time."""

    @pytest.fixture
    def clock(self, monkeypatch):
        import src.services.rp_cache as m

        now = [1_000_000.0]
        monkeypatch.setattr(m, "_now", lambda: now[0])
        return now

    @staticmethod
    def _cache_with_fetch(fetch):
        cache = RPSectionCache()
        cache._fetch_resources = fetch  # type: ignore[method-assign]
        return cache

    async def test_a_slow_fetch_is_cut_off_by_its_own_timeout(self, monkeypatch, clock):
        import src.services.rp_cache as m

        monkeypatch.setattr(m, "REFRESH_TIMEOUT_S", 0.01)

        async def _slow():
            await asyncio.sleep(5)
            return {}

        cache = self._cache_with_fetch(_slow)
        await cache.refresh()

        assert cache.list_slugs() == []
        assert cache._refreshing is False

    async def test_a_failed_refresh_is_not_retried_until_the_backoff_passes(self, clock):
        import src.services.rp_cache as m

        calls = []

        async def _fail():
            calls.append(1)
            raise httpx.ConnectError("nope")

        cache = self._cache_with_fetch(_fail)
        await cache.ensure_loaded()
        await cache.ensure_loaded()
        assert len(calls) == 1

        clock[0] += m.RETRY_AFTER_FAILURE_S + 1
        await cache.ensure_loaded()
        assert len(calls) == 2

    async def test_a_failure_logs_one_warning_naming_the_setting(self, clock, caplog):
        import src.services.rp_cache as m

        async def _fail():
            raise httpx.ConnectError("nope")

        cache = self._cache_with_fetch(_fail)
        with caplog.at_level("DEBUG", logger="src.services.rp_cache"):
            await cache.refresh()
            clock[0] += m.RETRY_AFTER_FAILURE_S + 1
            await cache.refresh()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "DRUPAL_RESOURCE_GROUPS_URL" in warnings[0].getMessage()
        assert "profile" in warnings[0].getMessage()
        assert all(r.exc_info is None for r in caplog.records)

    async def test_a_success_ends_the_failure_streak(self, clock, caplog):
        import src.services.rp_cache as m

        results: list = [httpx.ConnectError("a"), {"delta": RPInfo("delta", "Delta")}]

        async def _fetch():
            r = results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        cache = self._cache_with_fetch(_fetch)
        await cache.refresh()
        clock[0] += m.RETRY_AFTER_FAILURE_S + 1
        await cache.ensure_loaded()
        assert cache.list_slugs() == ["delta"]
        assert cache._retry_at == 0.0
        assert cache._failing is False

    async def test_background_refresh_does_not_block_and_runs_once(self, clock):
        release = asyncio.Event()
        calls = []

        async def _fetch():
            calls.append(1)
            await release.wait()
            return {"delta": RPInfo("delta", "Delta")}

        cache = self._cache_with_fetch(_fetch)
        cache.refresh_in_background()
        cache.refresh_in_background()  # already in flight
        await asyncio.sleep(0)
        cache.refresh_in_background()  # still in flight
        assert cache.list_slugs() == []

        release.set()
        await cache._task
        assert calls == [1]
        assert cache.list_slugs() == ["delta"]

        cache.refresh_in_background()  # fresh: nothing to do
        assert cache._task.done()
        await cache.stop_background_refresh()

    async def test_background_refresh_respects_the_backoff(self, clock):
        calls = []

        async def _fail():
            calls.append(1)
            raise httpx.ConnectError("nope")

        cache = self._cache_with_fetch(_fail)
        cache.refresh_in_background()
        await cache._task
        cache.refresh_in_background()
        assert cache._task.done()
        assert calls == [1]

    async def test_stop_background_refresh_cancels_a_fetch_in_flight(self, clock):
        cancelled = asyncio.Event()

        async def _hang():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {}

        cache = self._cache_with_fetch(_hang)
        cache.refresh_in_background()
        await asyncio.sleep(0)
        await cache.stop_background_refresh()

        assert cancelled.is_set()
        assert cache._task is None
        assert cache._refreshing is False
        await cache.stop_background_refresh()  # nothing running: a no-op
