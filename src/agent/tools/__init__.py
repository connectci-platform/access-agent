"""Agent-side LangChain tools.

Distinct from `src/tools/` (the MCP-client infrastructure) and from
`src/agent/domains/tools.py` (which wraps MCP catalog entries as
LangChain tools). This package holds tools authored directly for the
loop's catalog — currently the doc-search tool that wraps UKY's RAG
endpoints.
"""

from .access_documents import search_access_documents

__all__ = ["search_access_documents"]
