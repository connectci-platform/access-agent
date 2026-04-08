"""Tests for the Drupal profile fetcher."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.services.drupal_profile import DrupalProfileFetcher, UserProfile


class TestUserProfile:
    """Test UserProfile data methods."""

    def test_empty_profile_context(self):
        profile = UserProfile(access_id="test@access-ci.org")
        assert profile.to_context_dict() == {}

    def test_full_profile_context(self):
        profile = UserProfile(
            access_id="test@access-ci.org",
            institution="MIT",
            hpc_experience="Advanced",
            skills=["python", "ml"],
            interests=["genomics"],
            affinity_groups=[{"name": "AI Institute", "is_coordinator": False}],
            active_allocations=[{"resource": "Delta", "project": "TG-123"}],
        )
        ctx = profile.to_context_dict()
        assert ctx["institution"] == "MIT"
        assert ctx["hpc_experience"] == "Advanced"
        assert ctx["skills"] == ["python", "ml"]
        assert ctx["interests"] == ["genomics"]
        assert len(ctx["affinity_groups"]) == 1
        assert len(ctx["active_allocations"]) == 1

    def test_highlighted_capabilities_coordinator(self):
        profile = UserProfile(
            access_id="test@access-ci.org",
            affinity_groups=[
                {"name": "Pegasus", "is_coordinator": True},
                {"name": "AI Institute", "is_coordinator": False},
            ],
        )
        highlights = profile.highlighted_capabilities()
        assert len(highlights) == 1
        assert highlights[0]["id"] == "manage_announcements"
        assert "Pegasus" in highlights[0]["label"]

    def test_highlighted_capabilities_allocations(self):
        profile = UserProfile(
            access_id="test@access-ci.org",
            active_allocations=[
                {"resource": "Delta", "project": "TG-123"},
                {"resource": "Anvil", "project": "TG-456"},
            ],
        )
        highlights = profile.highlighted_capabilities()
        assert len(highlights) == 1
        assert highlights[0]["id"] == "check_usage"
        assert "Delta" in highlights[0]["label"]
        assert "Anvil" in highlights[0]["label"]

    def test_highlighted_capabilities_both(self):
        profile = UserProfile(
            access_id="test@access-ci.org",
            affinity_groups=[{"name": "Pegasus", "is_coordinator": True}],
            active_allocations=[{"resource": "Delta", "project": "TG-123"}],
        )
        highlights = profile.highlighted_capabilities()
        assert len(highlights) == 2
        ids = {h["id"] for h in highlights}
        assert ids == {"manage_announcements", "check_usage"}

    def test_highlighted_capabilities_empty(self):
        profile = UserProfile(access_id="test@access-ci.org")
        assert profile.highlighted_capabilities() == []

    def test_system_prompt_section_full(self):
        profile = UserProfile(
            access_id="test@access-ci.org",
            name="Jane Doe",
            institution="MIT",
            hpc_experience="Advanced",
            skills=["python"],
            affinity_groups=[{"name": "AI Institute", "is_coordinator": True}],
            active_allocations=[{"resource": "Delta", "project": "TG-123"}],
        )
        section = profile.to_system_prompt_section()
        assert section.startswith("## USER CONTEXT")
        assert "Jane Doe" in section
        assert "MIT" in section
        assert "AI Institute (coordinator)" in section
        assert "Delta (TG-123)" in section

    def test_system_prompt_section_empty(self):
        profile = UserProfile(access_id="test@access-ci.org")
        assert profile.to_system_prompt_section() == ""

    def test_highlighted_capabilities_many_allocations(self):
        """More than 3 allocations shows '... and N more'."""
        profile = UserProfile(
            access_id="test@access-ci.org",
            active_allocations=[
                {"resource": "Delta", "project": ""},
                {"resource": "Anvil", "project": ""},
                {"resource": "Expanse", "project": ""},
                {"resource": "Bridges-2", "project": ""},
                {"resource": "Stampede3", "project": ""},
            ],
        )
        highlights = profile.highlighted_capabilities()
        usage = [h for h in highlights if h["id"] == "check_usage"][0]
        assert "and 2 more" in usage["label"]


class TestDrupalProfileFetcher:
    """Test cache behavior of the fetcher."""

    def test_cache_hit(self):
        fetcher = DrupalProfileFetcher()
        profile = UserProfile(access_id="test@access-ci.org", name="Test User")
        from src.services.drupal_profile import _CacheEntry

        fetcher._cache["test@access-ci.org"] = _CacheEntry(
            profile=profile, fetched_at=time.time()
        )

    @pytest.mark.asyncio
    async def test_cache_returns_fresh_entry(self):
        fetcher = DrupalProfileFetcher()
        profile = UserProfile(access_id="test@access-ci.org", name="Cached")
        from src.services.drupal_profile import _CacheEntry

        fetcher._cache["test@access-ci.org"] = _CacheEntry(
            profile=profile, fetched_at=time.time()
        )
        result = await fetcher.get_profile("test@access-ci.org", "fake-jwt")
        assert result.name == "Cached"

    @pytest.mark.asyncio
    async def test_cache_expires(self):
        fetcher = DrupalProfileFetcher()
        profile = UserProfile(access_id="test@access-ci.org", name="Stale")
        from src.services.drupal_profile import _CacheEntry

        fetcher._cache["test@access-ci.org"] = _CacheEntry(
            profile=profile, fetched_at=time.time() - 600  # 10 min ago
        )

        with patch.object(fetcher, "_fetch_profile", new_callable=AsyncMock) as mock_fetch:
            fresh = UserProfile(access_id="test@access-ci.org", name="Fresh")
            mock_fetch.return_value = fresh
            result = await fetcher.get_profile("test@access-ci.org", "fake-jwt")
            assert result.name == "Fresh"
            mock_fetch.assert_called_once()

    def test_invalidate(self):
        fetcher = DrupalProfileFetcher()
        from src.services.drupal_profile import _CacheEntry

        fetcher._cache["test@access-ci.org"] = _CacheEntry(
            profile=UserProfile(access_id="test@access-ci.org"),
            fetched_at=time.time(),
        )
        fetcher.invalidate("test@access-ci.org")
        assert "test@access-ci.org" not in fetcher._cache

    def test_invalidate_missing_key_no_error(self):
        fetcher = DrupalProfileFetcher()
        fetcher.invalidate("nonexistent@access-ci.org")  # Should not raise
