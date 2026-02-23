"""HTTP client for UKY RAG endpoints.

Provides access to UKY-hosted RAG services:
- General ACCESS Q&A (allocations, resources, how-tos, policies)
- XDMoD Q&A (metrics, charts, XDMoD features)

Both endpoints accept POST {"query": "..."} and return {"response": "..."}.
"""

import logging
import time
from typing import Literal

import httpx
from pydantic import BaseModel

from ..config import settings
from ..telemetry import get_tracer

logger = logging.getLogger(__name__)


class UKYResponse(BaseModel):
    """Response from a UKY RAG endpoint."""

    response: str
    endpoint_type: Literal["general", "xdmod"]
    duration_ms: int = 0


class UKYClient:
    """HTTP client for UKY RAG endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        general_url: str | None = None,
        xdmod_url: str | None = None,
        timeout: float | None = None,
    ):
        self._api_key = api_key or settings.uky_rag_api_key_resolved
        self._general_url = general_url or settings.UKY_RAG_GENERAL_URL
        self._xdmod_url = xdmod_url or settings.UKY_RAG_XDMOD_URL
        self._timeout = timeout or settings.UKY_RAG_TIMEOUT
        self._client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        """Check if the client is configured and enabled."""
        return settings.UKY_RAG_ENABLED and bool(self._api_key)

    def _get_url(self, endpoint_type: Literal["general", "xdmod"]) -> str:
        """Get the URL for the given endpoint type."""
        if endpoint_type == "xdmod":
            return self._xdmod_url
        return self._general_url

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def ask(
        self,
        query: str,
        endpoint_type: Literal["general", "xdmod"],
        session_id: str = "",
        question_id: str = "",
    ) -> UKYResponse:
        """Send a query to a UKY RAG endpoint.

        Args:
            query: The user's question.
            endpoint_type: Which endpoint to use ("general" or "xdmod").
            session_id: Session identifier for tracking.
            question_id: Question identifier for tracking.

        Returns:
            UKYResponse with the answer text.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            httpx.RequestError: On connection errors.
        """
        tracer = get_tracer("access-agent.uky-rag")
        client = await self._get_client()
        url = self._get_url(endpoint_type)

        headers = {
            "X-API-KEY": self._api_key,
            "X-Origin": "access-agent",
            "Content-Type": "application/json",
        }
        if session_id:
            headers["X-Session-ID"] = session_id
        if question_id:
            headers["X-Query-ID"] = question_id

        with tracer.start_as_current_span(
            "uky_rag.ask",
            attributes={
                "uky_rag.endpoint_type": endpoint_type,
                "uky_rag.url": url,
                "uky_rag.query_length": len(query),
            },
        ) as span:
            start_time = time.time()
            try:
                response = await client.post(
                    url,
                    json={"query": query},
                    headers=headers,
                )
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", response.status_code)

                response.raise_for_status()
                data = response.json()

                answer = data.get("response", "")
                span.set_attribute("uky_rag.answer_length", len(answer))

                logger.info(
                    f"UKY RAG ({endpoint_type}): got {len(answer)} char response in {duration_ms}ms"
                )

                return UKYResponse(
                    response=answer,
                    endpoint_type=endpoint_type,
                    duration_ms=duration_ms,
                )

            except httpx.HTTPStatusError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", e.response.status_code)
                span.set_attribute("uky_rag.error", f"HTTP {e.response.status_code}")
                logger.error(
                    f"UKY RAG ({endpoint_type}) error: "
                    f"{e.response.status_code} - {e.response.text[:200]}"
                )
                raise
            except httpx.RequestError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("uky_rag.error", str(e)[:200])
                logger.error(f"UKY RAG ({endpoint_type}) request failed: {e}")
                raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_client: UKYClient | None = None


def get_uky_client() -> UKYClient:
    """Get the singleton UKY RAG client."""
    global _client
    if _client is None:
        _client = UKYClient()
    return _client
