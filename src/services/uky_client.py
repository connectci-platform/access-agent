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
    in_scope: bool | None = None


class UKYChunk(BaseModel):
    """A single retrieval chunk from the chat-mcp endpoint's top_documents."""

    rank: int = 0
    score: float = 0.0
    text: str = ""
    url: str = ""


class UKYRetrieval(BaseModel):
    """Chunk-only result from the chat-mcp endpoint.

    UKY's own ``response`` synthesis and ``in_scope`` flag are deliberately
    dropped here: the agent synthesizes from the raw chunks itself so it can
    judge relevance and cite sources directly.
    """

    chunks: list[UKYChunk]
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
        self._chatmcp_url = settings.UKY_CHATMCP_URL
        self._chatmcp_api_key = settings.UKY_CHATMCP_API_KEY
        self._timeout = timeout or settings.UKY_RAG_TIMEOUT
        self._client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        """Check if the client is configured and enabled."""
        return settings.UKY_RAG_ENABLED and bool(self._api_key)

    @property
    def is_chatmcp_configured(self) -> bool:
        """Check if the chat-mcp (chunk retrieval) endpoint is usable."""
        return settings.UKY_RAG_ENABLED and bool(self._chatmcp_api_key)

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
        rp_name: str | None = None,
    ) -> UKYResponse:
        """Send a query to a UKY RAG endpoint.

        Args:
            query: The user's question.
            endpoint_type: Which endpoint to use ("general" or "xdmod").
            session_id: Session identifier for tracking.
            question_id: Question identifier for tracking.
            rp_name: RP slug for resource-scoped queries (e.g. 'delta').
                     Routes to that RP's vector database when set.

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
            "X-Origin": rp_name or "access-agent",
            "Content-Type": "application/json",
        }
        if session_id:
            headers["X-Session-ID"] = session_id
        if question_id:
            headers["X-Query-ID"] = question_id

        body: dict[str, str] = {"query": query}
        if rp_name:
            body["rp_name"] = rp_name

        with tracer.start_as_current_span(
            "uky_rag.ask",
            attributes={
                "uky_rag.endpoint_type": endpoint_type,
                "uky_rag.url": url,
                "uky_rag.query_length": len(query),
                **({"uky_rag.rp_name": rp_name} if rp_name else {}),
            },
        ) as span:
            start_time = time.time()
            try:
                response = await client.post(
                    url,
                    json=body,
                    headers=headers,
                )
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", response.status_code)

                response.raise_for_status()
                data = response.json()

                answer = data.get("response", "")
                in_scope = data.get("in_scope")  # None until UKY implements it
                span.set_attribute("uky_rag.answer_length", len(answer))
                if in_scope is not None:
                    span.set_attribute("uky_rag.in_scope", in_scope)
                if rp_name:
                    span.set_attribute("uky_rag.rp_name", rp_name)

                rp_suffix = f" (rp={rp_name}, in_scope={in_scope})" if rp_name else ""
                logger.info(
                    f"UKY RAG ({endpoint_type}): got {len(answer)} char response "
                    f"in {duration_ms}ms{rp_suffix}"
                )

                return UKYResponse(
                    response=answer,
                    endpoint_type=endpoint_type,
                    duration_ms=duration_ms,
                    in_scope=in_scope,
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

    async def retrieve(
        self,
        query: str,
        session_id: str = "",
        question_id: str = "",
        rp_name: str | None = None,
    ) -> UKYRetrieval:
        """Fetch retrieval chunks from the UKY chat-mcp endpoint.

        Unlike `ask()`, this returns the raw `top_documents` chunks and
        discards UKY's `response` synthesis and `in_scope` flag — the agent
        synthesizes and judges relevance itself.

        Args:
            query: The user's question.
            session_id: Session identifier for tracking.
            question_id: Question identifier for tracking.
            rp_name: RP slug for resource-scoped queries (e.g. 'delta').

        Returns:
            UKYRetrieval with the parsed chunks.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            httpx.RequestError: On connection errors.
        """
        tracer = get_tracer("access-agent.uky-rag")
        client = await self._get_client()

        headers = {
            "X-API-KEY": self._chatmcp_api_key,
            "X-Origin": rp_name or "access-agent",
            "Content-Type": "application/json",
        }
        if session_id:
            headers["X-Session-ID"] = session_id
        if question_id:
            headers["X-Query-ID"] = question_id

        body: dict[str, str] = {"query": query}
        if rp_name:
            body["rp_name"] = rp_name

        with tracer.start_as_current_span(
            "uky_rag.retrieve",
            attributes={
                "uky_rag.url": self._chatmcp_url,
                "uky_rag.query_length": len(query),
                **({"uky_rag.rp_name": rp_name} if rp_name else {}),
            },
        ) as span:
            start_time = time.time()
            try:
                response = await client.post(
                    self._chatmcp_url,
                    json=body,
                    headers=headers,
                )
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", response.status_code)

                response.raise_for_status()
                data = response.json()

                raw_docs = data.get("top_documents") or []
                chunks = [
                    UKYChunk(
                        rank=doc.get("rank", 0),
                        score=doc.get("score", 0.0),
                        text=doc.get("text", ""),
                        url=doc.get("url", ""),
                    )
                    for doc in raw_docs
                ]
                span.set_attribute("uky_rag.chunk_count", len(chunks))

                rp_suffix = f" (rp={rp_name})" if rp_name else ""
                logger.info(f"UKY chat-mcp: got {len(chunks)} chunks in {duration_ms}ms{rp_suffix}")

                return UKYRetrieval(chunks=chunks, duration_ms=duration_ms)

            except httpx.HTTPStatusError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("http.status_code", e.response.status_code)
                span.set_attribute("uky_rag.error", f"HTTP {e.response.status_code}")
                logger.error(
                    f"UKY chat-mcp error: {e.response.status_code} - {e.response.text[:200]}"
                )
                raise
            except httpx.RequestError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                span.set_attribute("uky_rag.duration_ms", duration_ms)
                span.set_attribute("uky_rag.error", str(e)[:200])
                logger.error(f"UKY chat-mcp request failed: {e}")
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
