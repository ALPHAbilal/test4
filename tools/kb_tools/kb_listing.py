"""
Knowledge Base Listing Tools

This module provides tools for listing files in the knowledge base.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from agents import function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

@function_tool(strict_mode=False)
async def list_knowledge_base_files(ctx: RunContextWrapper, vs_id: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, str]]:
    """Lists all files in the knowledge base.

    This tool provides a list of all documents available in the specified knowledge base.
    Use this tool when the user asks about what's in the knowledge base or what documents are available.

    Args:
        vs_id: Optional vector store ID. If not provided, uses the one from context.

    Returns:
        A dictionary containing the list of files and their metadata.
    """
    logger.info(f"[Tool Call] list_knowledge_base_files")
    tool_context = ctx.context
    vector_store_id = vs_id or tool_context.get("vector_store_id")
    tool_client = tool_context.get("client")

    if not tool_client or not vector_store_id:
        return {"error": "Tool configuration error. Missing client or vector store ID."}

    try:
        # Get the list of files in the vector store
        files_response = await asyncio.to_thread(tool_client.vector_stores.files.list, vector_store_id=vector_store_id)

        if not files_response or not files_response.data:
            return {"files": [], "message": "No files found in the knowledge base."}

        # Process the files to get more details
        files_info = []
        for file_obj in files_response.data:
            try:
                # Get more details about the file
                file_details = await asyncio.to_thread(tool_client.files.retrieve, file_id=file_obj.id)

                # Extract relevant information
                file_info = {
                    "id": file_obj.id,
                    "filename": file_details.filename,
                    "purpose": getattr(file_details, "purpose", "unknown"),
                    "created_at": str(getattr(file_details, "created_at", "unknown")),
                    "size": getattr(file_details, "bytes", 0),
                }

                files_info.append(file_info)
            except Exception as file_err:
                logger.error(f"Error getting details for file {file_obj.id}: {file_err}")
                # Include basic info even if details retrieval fails
                files_info.append({"id": file_obj.id, "filename": getattr(file_obj, "filename", f"File-{file_obj.id[-6:]}")})

        return {
            "files": files_info,
            "count": len(files_info),
            "vector_store_id": vector_store_id,
            "message": f"Found {len(files_info)} files in the knowledge base."
        }
    except Exception as e:
        logger.error(f"Error listing knowledge base files: {e}", exc_info=True)
        return {"error": f"Failed to list knowledge base files: {str(e)}"}
