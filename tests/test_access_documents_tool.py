"""Tests for the search_access_documents tool wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.tools.access_documents import (
    _search_access_documents,
    search_access_documents,
)
from src.services.uky_client import UKYResponse


@pytest.fixture
def mock_client() -> Any:
    """Replace get_uky_client() with a mock for the duration of a test."""
    with patch("src.agent.tools.access_documents.get_uky_client") as factory:
        client = factory.return_value
        client.is_configured = True
        client.ask = AsyncMock()
        yield client


class TestStructuredToolShape:
    """Validate the LangChain tool surface the loop will see."""

    def test_name_is_stable(self) -> None:
        assert search_access_documents.name == "search_access_documents"

    def test_args_schema_exposes_three_fields(self) -> None:
        fields = search_access_documents.args_schema.model_fields
        assert set(fields) == {"query", "source", "rp_name"}

    def test_source_defaults_to_general(self) -> None:
        fields = search_access_documents.args_schema.model_fields
        assert fields["source"].default == "general"

    def test_rp_name_is_optional(self) -> None:
        fields = search_access_documents.args_schema.model_fields
        assert fields["rp_name"].default is None


class TestSearchAccessDocuments:
    """Body behavior: arg pass-through, error paths, response shape."""

    @pytest.mark.asyncio
    async def test_returns_uky_response_text(self, mock_client: Any) -> None:
        mock_client.ask.return_value = UKYResponse(
            response="ACCESS allocations are awarded via XRAS.",
            endpoint_type="general",
        )
        result = await _search_access_documents(query="how do allocations work")
        assert result == "ACCESS allocations are awarded via XRAS."

    @pytest.mark.asyncio
    async def test_passes_source_through_as_endpoint_type(self, mock_client: Any) -> None:
        mock_client.ask.return_value = UKYResponse(
            response="XDMoD job-realm fields:", endpoint_type="xdmod"
        )
        await _search_access_documents(query="what fields are in xdmod jobs realm", source="xdmod")
        kwargs = mock_client.ask.await_args.kwargs
        assert kwargs["endpoint_type"] == "xdmod"

    @pytest.mark.asyncio
    async def test_passes_rp_name_through(self, mock_client: Any) -> None:
        mock_client.ask.return_value = UKYResponse(
            response="Delta uses A100s.", endpoint_type="general"
        )
        await _search_access_documents(query="what GPUs", source="general", rp_name="delta")
        kwargs = mock_client.ask.await_args.kwargs
        assert kwargs["rp_name"] == "delta"

    @pytest.mark.asyncio
    async def test_unconfigured_client_returns_user_visible_string(self, mock_client: Any) -> None:
        mock_client.is_configured = False
        result = await _search_access_documents(query="anything")
        assert "unavailable" in result.lower()
        mock_client.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_uky_exception_returns_string_not_raise(self, mock_client: Any) -> None:
        mock_client.ask.side_effect = RuntimeError("connection reset")
        result = await _search_access_documents(query="anything")
        assert "failed" in result.lower()
        assert "connection reset" in result

    @pytest.mark.asyncio
    async def test_empty_response_returns_guidance(self, mock_client: Any) -> None:
        mock_client.ask.return_value = UKYResponse(response="", endpoint_type="general")
        result = await _search_access_documents(query="anything")
        assert "no content" in result.lower()
