"""
Knowledge Base Adapter Tools

This module provides adapter tools for the knowledge base tools.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from agents import function_tool, RunContextWrapper
from pydantic import Field

logger = logging.getLogger(__name__)

@function_tool(strict_mode=False)
async def get_kb_document_content_adapter(ctx: RunContextWrapper, document_type: str, query_or_identifier: Optional[str] = None, query: Optional[str] = None, included_file_ids: Optional[List[str]] = None) -> Any:
    """
    Adapter for the get_kb_document_content tool that handles both parameter names.

    This adapter accepts both 'query' and 'query_or_identifier' parameter names and
    forwards the call to the actual get_kb_document_content tool with the correct parameter name.

    Args:
        ctx: The run context wrapper
        document_type: Type of document to search for
        query_or_identifier: The query or identifier to search for (preferred parameter name)
        query: Alternative parameter name for query_or_identifier
        included_file_ids: Optional list of file IDs to limit the search to
    """
    logger.info(f"[Tool Call] get_kb_document_content_adapter: document_type={document_type}, query_or_identifier={query_or_identifier}, query={query}, included_file_ids={included_file_ids}")

    # Import the actual tool
    from tools.kb_tools.kb_retrieval import get_kb_document_content

    # Use query_or_identifier if provided, otherwise use query
    final_query = query_or_identifier if query_or_identifier is not None else query

    if final_query is None:
        return {"error_message": "Either query_or_identifier or query must be provided"}

    # Call the actual tool with the corrected parameters
    return await get_kb_document_content(
        ctx=ctx,
        document_type=document_type,
        query_or_identifier=final_query,
        included_file_ids=included_file_ids
    )
