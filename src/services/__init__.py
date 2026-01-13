"""Services for Q&A retrieval.

This module provides:
- QAServiceClient: HTTP client for access-qa-service (RAG retrieval)
"""

from .qa_client import (
    QAMatch,
    QAServiceClient,
    get_qa_client,
)

__all__ = [
    "QAMatch",
    "QAServiceClient",
    "get_qa_client",
]
