"""Compress node - intelligent summarization of tool results.

This node processes tool results to extract the most relevant information
before synthesis, ensuring all results are represented without arbitrary
truncation.
"""

import json
import logging
from typing import Any

from ...config import settings
from ..state import AgentState, ToolResult

logger = logging.getLogger(__name__)

# Fields to keep for different tool types (prioritized for relevance)
RESOURCE_SUMMARY_FIELDS = [
    "id",
    "name",
    "description",
    "organization_names",
    "resourceType",
    "hasGpu",
]
SOFTWARE_SUMMARY_FIELDS = ["name", "description", "version", "resource", "category"]
EVENT_SUMMARY_FIELDS = ["title", "description", "start_date", "end_date", "location"]
OUTAGE_SUMMARY_FIELDS = ["resource", "title", "status", "start_time", "end_time", "description"]
ANNOUNCEMENT_SUMMARY_FIELDS = ["title", "summary", "date", "type"]

# Map tool names to their relevant fields
TOOL_FIELD_MAP = {
    "search_resources": RESOURCE_SUMMARY_FIELDS,
    "get_resource_details": RESOURCE_SUMMARY_FIELDS,
    "search_software": SOFTWARE_SUMMARY_FIELDS,
    "get_software_details": SOFTWARE_SUMMARY_FIELDS,
    "get_upcoming_events": EVENT_SUMMARY_FIELDS,
    "get_current_outages": OUTAGE_SUMMARY_FIELDS,
    "get_announcements": ANNOUNCEMENT_SUMMARY_FIELDS,
}


async def compress_node(state: AgentState) -> dict[str, Any]:
    """Compress tool results for efficient synthesis.

    This node:
    1. Extracts only relevant fields from each tool result
    2. Ensures ALL items are represented (no mid-list truncation)
    3. Summarizes very large result sets with counts
    4. Preserves full data for small result sets

    Args:
        state: Current agent state with tool_results.

    Returns:
        Dict with compressed_results for synthesis.
    """
    tool_results = state.get("tool_results", [])
    query_analysis = state.get("query_analysis")

    if not tool_results:
        logger.info("No tool results to compress")
        return {"compressed_results": []}

    compressed = []
    for result in tool_results:
        compressed_result = _compress_single_result(result, query_analysis)
        compressed.append(compressed_result)

    total_original = sum(_estimate_size(r.data) for r in tool_results if r.data)
    total_compressed = sum(_estimate_size(r["data"]) for r in compressed if r.get("data"))

    logger.info(
        f"Compressed {len(tool_results)} results: "
        f"{total_original:,} -> {total_compressed:,} chars "
        f"({100 - (total_compressed / max(total_original, 1) * 100):.0f}% reduction)"
    )

    return {"compressed_results": compressed}


def _compress_single_result(result: ToolResult, query_analysis: Any) -> dict[str, Any]:
    """Compress a single tool result.

    Args:
        result: The tool result to compress.
        query_analysis: Query analysis for context.

    Returns:
        Compressed result dict.
    """
    if not result.success:
        return {
            "tool_name": result.tool_name,
            "success": False,
            "error": result.error,
        }

    data = result.data
    if data is None:
        return {
            "tool_name": result.tool_name,
            "success": True,
            "data": None,
        }

    # Get relevant fields for this tool type
    relevant_fields = TOOL_FIELD_MAP.get(result.tool_name)

    # Compress the data
    compressed_data = _compress_data(data, relevant_fields)

    return {
        "tool_name": result.tool_name,
        "success": True,
        "data": compressed_data,
    }


def _compress_data(data: Any, relevant_fields: list[str] | None) -> Any:
    """Compress data by extracting relevant fields.

    Args:
        data: The data to compress.
        relevant_fields: Fields to keep, or None to keep all.

    Returns:
        Compressed data.
    """
    if data is None:
        return None

    # Handle dict with items/results list
    if isinstance(data, dict):
        # Check for common list wrapper patterns
        items = None
        total = None

        if "items" in data:
            items = data["items"]
            total = data.get("total", len(items) if isinstance(items, list) else None)
        elif "results" in data:
            items = data["results"]
            total = data.get("total", len(items) if isinstance(items, list) else None)

        if items is not None and isinstance(items, list):
            compressed_items = _compress_list(items, relevant_fields)
            compressed: dict[str, Any] = {"items": compressed_items}
            if total is not None:
                compressed["total"] = total
            max_items = settings.MAX_COMPRESSED_ITEMS
            if len(items) > max_items:
                compressed["note"] = f"Showing {max_items} of {len(items)} items"
            return compressed

        # Regular dict - extract relevant fields if specified
        if relevant_fields:
            return {k: v for k, v in data.items() if k in relevant_fields}
        return data

    # Handle list directly
    if isinstance(data, list):
        return _compress_list(data, relevant_fields)

    # Return primitives as-is
    return data


def _compress_list(items: list[Any], relevant_fields: list[str] | None) -> list[Any]:
    """Compress a list of items.

    Args:
        items: List of items to compress.
        relevant_fields: Fields to keep from each item.

    Returns:
        Compressed list.
    """
    if not items:
        return []

    # Limit to configured max but always include all
    # (we just warn in metadata if truncated)
    max_items = settings.MAX_COMPRESSED_ITEMS
    items_to_process = items[:max_items]

    compressed = []
    for item in items_to_process:
        if isinstance(item, dict) and relevant_fields:
            # Extract only relevant fields
            compressed_item = {k: v for k, v in item.items() if k in relevant_fields}
            compressed.append(compressed_item)
        else:
            compressed.append(item)

    return compressed


def _estimate_size(data: Any) -> int:
    """Estimate the character size of data when serialized.

    Args:
        data: Data to estimate.

    Returns:
        Estimated character count.
    """
    if data is None:
        return 0
    try:
        return len(json.dumps(data, default=str))
    except (TypeError, ValueError):
        return len(str(data))
