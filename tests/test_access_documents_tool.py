"""Tests for the search_access_documents tool wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.agent.tools.access_documents import (
    _search_access_documents,
    search_access_documents,
)
from src.services.uky_client import UKYChunk, UKYResponse, UKYRetrieval


@pytest.fixture
def mock_client() -> Any:
    """Replace get_uky_client() with a mock for the duration of a test.

    The general path uses ``client.retrieve`` (chat-mcp chunks); the XDMoD
    path still uses ``client.ask`` (legacy synthesis endpoint). Both mocks
    are pre-wired so individual tests only configure the relevant one.
    """
    with patch("src.agent.tools.access_documents.get_uky_client") as factory:
        client = factory.return_value
        client.is_configured = True
        client.is_chatmcp_configured = True
        client.ask = AsyncMock()
        client.retrieve = AsyncMock()
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
    async def test_returns_uky_chunks_text(self, mock_client: Any) -> None:
        mock_client.retrieve.return_value = UKYRetrieval(
            chunks=[
                UKYChunk(
                    rank=1,
                    text="ACCESS allocations are awarded via XRAS.",
                    url="https://allocations.access-ci.org/",
                )
            ]
        )
        result = await _search_access_documents(query="how do allocations work")
        assert isinstance(result, str)
        assert "ACCESS allocations are awarded via XRAS." in result
        assert "https://allocations.access-ci.org/" in result

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
        mock_client.retrieve.return_value = UKYRetrieval(chunks=[])
        await _search_access_documents(query="what GPUs", source="general", rp_name="delta")
        kwargs = mock_client.retrieve.await_args.kwargs
        assert kwargs["rp_name"] == "delta"

    @pytest.mark.asyncio
    async def test_rp_name_normalized_before_call(self, mock_client: Any) -> None:
        # A guessed display-name variant ('Bridges-2') must reach UKY as the slug.
        mock_client.retrieve.return_value = UKYRetrieval(chunks=[])
        await _search_access_documents(query="what GPUs", source="general", rp_name="Bridges-2")
        kwargs = mock_client.retrieve.await_args.kwargs
        assert kwargs["rp_name"] == "bridges2"

    @pytest.mark.asyncio
    async def test_normalize_noop_on_valid_slug(self, mock_client: Any) -> None:
        # An already-valid slug must not be corrupted by normalization.
        mock_client.retrieve.return_value = UKYRetrieval(chunks=[])
        await _search_access_documents(query="what GPUs", source="general", rp_name="bridges2")
        kwargs = mock_client.retrieve.await_args.kwargs
        assert kwargs["rp_name"] == "bridges2"

    @pytest.mark.asyncio
    async def test_rp_name_normalized_before_xdmod_call(self, mock_client: Any) -> None:
        # The xdmod path (client.ask) must also receive the normalized value.
        mock_client.ask.return_value = UKYResponse(response="XDMoD fields:", endpoint_type="xdmod")
        await _search_access_documents(
            query="usage on bridges", source="xdmod", rp_name="Bridges-2"
        )
        kwargs = mock_client.ask.await_args.kwargs
        assert kwargs["rp_name"] == "bridges2"

    @pytest.mark.asyncio
    async def test_unconfigured_client_returns_user_visible_string(self, mock_client: Any) -> None:
        mock_client.is_chatmcp_configured = False
        result = await _search_access_documents(query="anything")
        assert "unavailable" in result.lower()
        mock_client.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_uky_http_status_error_returns_error_dict_with_status_code(
        self, mock_client: Any
    ) -> None:
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        response = httpx.Response(400, request=request)
        mock_client.retrieve.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=request, response=response
        )
        result = await _search_access_documents(query="anything")
        assert isinstance(result, dict)
        assert "error" in result
        assert "failed" in result["error"].lower()
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_uky_request_error_returns_error_dict_with_none_status_code(
        self, mock_client: Any
    ) -> None:
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        mock_client.retrieve.side_effect = httpx.RequestError("connection reset", request=request)
        result = await _search_access_documents(query="anything")
        assert isinstance(result, dict)
        assert "error" in result
        assert "connection reset" in result["error"]
        assert result["status_code"] is None

    @pytest.mark.asyncio
    async def test_xdmod_backend_error_returns_error_dict(self, mock_client: Any) -> None:
        # The xdmod path uses client.ask (not retrieve); its except block must also
        # surface a backend error as a structured dict, same as the general path.
        request = httpx.Request("POST", "https://uky.example/api/ask")
        response = httpx.Response(400, request=request)
        mock_client.ask.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=request, response=response
        )
        result = await _search_access_documents(query="anything", source="xdmod")
        assert isinstance(result, dict)
        assert "error" in result
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_empty_response_returns_guidance(self, mock_client: Any) -> None:
        mock_client.retrieve.return_value = UKYRetrieval(chunks=[])
        result = await _search_access_documents(query="anything")
        assert "no excerpts" in result.lower()


class TestInvalidRpNameRetry:
    """Invalid-rp_name 400 triggers exactly one unscoped retry, with a caveat note."""

    @staticmethod
    def _invalid_rp_name_error() -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        response = httpx.Response(400, request=request, text="Invalid rp_name: 'bogus'")
        return httpx.HTTPStatusError("Bad Request", request=request, response=response)

    @staticmethod
    def _other_400_error() -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        response = httpx.Response(400, request=request, text="Bad query syntax")
        return httpx.HTTPStatusError("Bad Request", request=request, response=response)

    @pytest.mark.asyncio
    async def test_invalid_rp_name_400_retries_unscoped_with_note(self, mock_client: Any) -> None:
        general_chunks = UKYRetrieval(
            chunks=[
                UKYChunk(
                    rank=1,
                    text="General ACCESS onboarding info.",
                    url="https://access-ci.org/onboarding",
                )
            ]
        )
        mock_client.retrieve.side_effect = [self._invalid_rp_name_error(), general_chunks]

        with patch("src.agent.tools.access_documents.record_retrieved_chunks") as mock_record:
            result = await _search_access_documents(
                query="what GPUs", source="general", rp_name="bogus"
            )

        assert isinstance(result, str)
        assert "General ACCESS onboarding info." in result
        assert "general ACCESS results" in result

        assert mock_client.retrieve.await_count == 2
        first_kwargs = mock_client.retrieve.await_args_list[0].kwargs
        second_kwargs = mock_client.retrieve.await_args_list[1].kwargs
        assert first_kwargs["rp_name"] == "bogus"
        assert second_kwargs["rp_name"] is None

        mock_record.assert_called_once_with(general_chunks.chunks)

    @pytest.mark.asyncio
    async def test_non_rp_name_400_does_not_retry(self, mock_client: Any) -> None:
        mock_client.retrieve.side_effect = self._other_400_error()

        result = await _search_access_documents(
            query="what GPUs", source="general", rp_name="bogus"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert result["status_code"] == 400
        assert mock_client.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_non_400_error_does_not_retry(self, mock_client: Any) -> None:
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        mock_client.retrieve.side_effect = httpx.RequestError("connection reset", request=request)

        result = await _search_access_documents(
            query="what GPUs", source="general", rp_name="bogus"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert mock_client.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_invalid_rp_name_retry_also_fails_surfaces_error(self, mock_client: Any) -> None:
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        mock_client.retrieve.side_effect = [
            self._invalid_rp_name_error(),
            httpx.RequestError("connection reset", request=request),
        ]

        result = await _search_access_documents(
            query="what GPUs", source="general", rp_name="bogus"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "connection reset" in result["error"]
        assert mock_client.retrieve.await_count == 2

    @pytest.mark.asyncio
    async def test_xdmod_invalid_rp_name_retries_unscoped_with_note(self, mock_client: Any) -> None:
        request = httpx.Request("POST", "https://uky.example/api/ask")
        response = httpx.Response(400, request=request, text="Invalid rp_name: 'bogus'")
        error = httpx.HTTPStatusError("Bad Request", request=request, response=response)
        mock_client.ask.side_effect = [
            error,
            UKYResponse(response="XDMoD general usage summary.", endpoint_type="xdmod"),
        ]

        result = await _search_access_documents(
            query="usage on bogus", source="xdmod", rp_name="bogus"
        )

        assert isinstance(result, str)
        assert "XDMoD general usage summary." in result
        assert "general ACCESS results" in result

        assert mock_client.ask.await_count == 2
        first_kwargs = mock_client.ask.await_args_list[0].kwargs
        second_kwargs = mock_client.ask.await_args_list[1].kwargs
        assert first_kwargs["rp_name"] == "bogus"
        assert second_kwargs["rp_name"] is None

    @pytest.mark.asyncio
    async def test_xdmod_invalid_rp_name_retry_also_fails_surfaces_error(
        self, mock_client: Any
    ) -> None:
        request = httpx.Request("POST", "https://uky.example/api/ask")
        response = httpx.Response(400, request=request, text="Invalid rp_name: 'bogus'")
        error = httpx.HTTPStatusError("Bad Request", request=request, response=response)
        mock_client.ask.side_effect = [
            error,
            httpx.RequestError("connection reset", request=request),
        ]

        result = await _search_access_documents(
            query="usage on bogus", source="xdmod", rp_name="bogus"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "connection reset" in result["error"]
        assert mock_client.ask.await_count == 2

    @pytest.mark.asyncio
    async def test_400_with_other_status_body_is_not_treated_as_invalid_rp_name(
        self, mock_client: Any
    ) -> None:
        # A non-400 status must short-circuit _is_invalid_rp_name before the body
        # is even inspected (covers the status_code != 400 branch explicitly).
        request = httpx.Request("POST", "https://uky.example/api/retrieve-docs")
        response = httpx.Response(500, request=request, text="invalid rp_name: 'bogus'")
        error = httpx.HTTPStatusError("Server Error", request=request, response=response)
        mock_client.retrieve.side_effect = error

        result = await _search_access_documents(
            query="what GPUs", source="general", rp_name="bogus"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert result["status_code"] == 500
        assert mock_client.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_successful_scoped_call_has_no_note(self, mock_client: Any) -> None:
        mock_client.retrieve.return_value = UKYRetrieval(
            chunks=[
                UKYChunk(
                    rank=1,
                    text="Delta-specific GPU details.",
                    url="https://access-ci.org/delta",
                )
            ]
        )
        result = await _search_access_documents(
            query="what GPUs", source="general", rp_name="delta"
        )
        assert isinstance(result, str)
        assert "Delta-specific GPU details." in result
        assert "general ACCESS results" not in result
        assert mock_client.retrieve.await_count == 1
