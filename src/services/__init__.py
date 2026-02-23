"""Services for Q&A retrieval and UKY RAG endpoints.

This module provides:
- QAServiceClient: HTTP client for access-qa-service (pgvector RAG retrieval)
- UKYClient: HTTP client for UKY-hosted RAG endpoints (general + XDMoD)
"""

from .qa_client import (
    QAMatch,
    QAServiceClient,
    get_qa_client,
)
from .uky_client import (
    UKYClient,
    UKYResponse,
    get_uky_client,
)

__all__ = [
    "QAMatch",
    "QAServiceClient",
    "UKYClient",
    "UKYResponse",
    "get_qa_client",
    "get_uky_client",
]
