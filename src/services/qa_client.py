"""HTTP client for access-qa-service.

Provides semantic search over verified Q&A pairs.
"""

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import settings
from ..telemetry import get_tracer

logger = logging.getLogger(__name__)


class QAMatch(BaseModel):
    """A matched Q&A pair from semantic search."""

    id: str
    question: str
    answer: str
    domain: str
    entity_id: str
    similarity_score: float
    metadata: dict[str, object] = {}


class QAServiceClient:
    """HTTP client for access-qa-service API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        """Initialize the client.

        Args:
            base_url: Base URL for the QA service. Defaults to settings.
            timeout: Request timeout in seconds.
        """
        self._base_url = (base_url or settings.QA_SERVICE_URL).rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        """Check if the service URL is configured."""
        return bool(self._base_url)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def search(
        self,
        query: str,
        limit: int = 3,
        threshold: float | None = None,
    ) -> list[QAMatch]:
        """Search for matching Q&A pairs.

        Args:
            query: The search query.
            limit: Maximum number of results.
            threshold: Minimum similarity score (uses service default if None).

        Returns:
            List of matching Q&A pairs.
        """
        tracer = get_tracer("access-agent.rag")
        client = await self._get_client()

        payload: dict[str, Any] = {"query": query, "limit": limit}
        if threshold is not None:
            payload["threshold"] = threshold

        with tracer.start_as_current_span(
            "rag.search",
            attributes={
                "rag.service": "qa-service",
                "rag.endpoint": "/search",
                "rag.query_length": len(query),
                "rag.limit": limit,
                "rag.threshold": threshold if threshold is not None else -1,
            },
        ) as span:
            start_time = time.time()
            try:
                response = await client.post("/search", json=payload)
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", response.status_code)

                response.raise_for_status()
                data = response.json()

                matches = [QAMatch(**m) for m in data.get("matches", [])]
                span.set_attribute("rag.matches_returned", len(matches))
                span.set_attribute("rag.cached", data.get("cached", False))

                if matches:
                    span.set_attribute("rag.best_score", matches[0].similarity_score)

                logger.debug(
                    f"QA service search: {len(matches)} matches for '{query[:50]}...' "
                    f"(cached={data.get('cached', False)}, {duration_ms}ms)"
                )
                return matches

            except httpx.HTTPStatusError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", e.response.status_code)
                span.set_attribute("rag.error", f"HTTP {e.response.status_code}")
                logger.error(f"QA service error: {e.response.status_code} - {e.response.text}")
                raise
            except httpx.RequestError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("rag.duration_ms", duration_ms)
                span.set_attribute("rag.error", str(e)[:200])
                logger.error(f"QA service request failed: {e}")
                raise

    async def search_by_domain(
        self,
        query: str,
        domain: str,
        limit: int = 3,
        threshold: float | None = None,
    ) -> list[QAMatch]:
        """Search for matching Q&A pairs filtered by domain.

        Args:
            query: The search query.
            domain: Domain to filter by.
            limit: Maximum number of results.
            threshold: Minimum similarity score.

        Returns:
            List of matching Q&A pairs.
        """
        client = await self._get_client()

        payload: dict[str, Any] = {"query": query, "domain": domain, "limit": limit}
        if threshold is not None:
            payload["threshold"] = threshold

        try:
            response = await client.post("/search/by-domain", json=payload)
            response.raise_for_status()
            data = response.json()

            return [QAMatch(**m) for m in data.get("matches", [])]

        except httpx.HTTPStatusError as e:
            logger.error(f"QA service error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"QA service request failed: {e}")
            raise

    async def validate_citations(
        self,
        citations: list[tuple[str, str]],
    ) -> tuple[bool, list[str]]:
        """Validate citations against the registry.

        Args:
            citations: List of (domain, entity_id) tuples.

        Returns:
            Tuple of (all_valid, list_of_invalid_citations).
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/citations/validate",
                json={"citations": citations},
            )
            response.raise_for_status()
            data = response.json()

            return data.get("all_valid", False), data.get("invalid", [])

        except httpx.HTTPStatusError as e:
            logger.error(f"Citation validation error: {e.response.status_code}")
            # On error, assume citations are invalid for safety
            return False, [f"<<SRC:{d}:{e}>>" for d, e in citations]
        except httpx.RequestError as e:
            logger.error(f"Citation validation request failed: {e}")
            return False, [f"<<SRC:{d}:{e}>>" for d, e in citations]

    async def health_check(self) -> bool:
        """Check if the QA service is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        client = await self._get_client()

        try:
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_client: QAServiceClient | None = None


def get_qa_client() -> QAServiceClient:
    """Get the singleton QA service client."""
    global _client
    if _client is None:
        _client = QAServiceClient()
    return _client
