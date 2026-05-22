"""Services for UKY RAG endpoints.

This module provides:
- UKYClient: HTTP client for UKY-hosted RAG endpoints (general + XDMoD)
"""

from .uky_client import (
    UKYClient,
    UKYResponse,
    get_uky_client,
)

__all__ = [
    "UKYClient",
    "UKYResponse",
    "get_uky_client",
]
