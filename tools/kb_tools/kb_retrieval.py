"""
Knowledge Base Retrieval Tools

This module provides tools for retrieving content from the knowledge base.
"""

import asyncio
import logging
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from agents import function_tool, RunContextWrapper
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

# Import constants from app.py
try:
    from app import MAX_SEARCH_RESULTS_TOOL, SEARCH_RANKER
except ImportError:
    # Default values if app.py cannot be imported
    MAX_SEARCH_RESULTS_TOOL = 5
    SEARCH_RANKER = "hybrid"

# Import semantic cache
try:
    from data.semantic_cache import semantic_search_cache
except ImportError:
    semantic_search_cache = None
    logger.warning("Could not import semantic_search_cache, caching will be disabled")

# Import chat_db
try:
    from app import chat_db
except ImportError:
    chat_db = None
    logger.warning("Could not import chat_db, file filtering will be limited")

# Define models for return types
class SourceMetadata(BaseModel):
    file_id: str
    file_name: str
    section: Optional[str] = None
    confidence: float = 0.0

class RetrievalSuccess(BaseModel):
    content: str
    source_filename: str
    sources: List[SourceMetadata] = []

class RetrievalError(BaseModel):
    error_message: str

@function_tool(strict_mode=False)
async def get_kb_document_content(ctx: RunContextWrapper, document_type: str, query_or_identifier: str, included_file_ids: Optional[List[str]] = Field(None, description="Optional list of file IDs to limit the search/retrieval to.")) -> Union[RetrievalSuccess, RetrievalError]:
    """Retrieves content from the knowledge base (Vector Store) based on document type and query/identifier."""
    logger.info(f"[Tool Call] get_kb_document_content: type='{document_type}', query='{query_or_identifier[:50]}...'")
    tool_context = ctx.context
    vs_id = tool_context.get("vector_store_id")
    tool_client = tool_context.get("client")
    chat_id = tool_context.get("chat_id")
    if not tool_client or not vs_id:
        return RetrievalError(error_message="Tool config error.")

    # Get file inclusion settings from parameter or context
    file_ids_to_filter = []

    # 1. Prioritize the 'included_file_ids' parameter if provided by the agent
    if included_file_ids is not None and len(included_file_ids) > 0:
        file_ids_to_filter = included_file_ids
        logger.info(f"Using file IDs from tool parameter: {file_ids_to_filter}")
    # 2. If parameter not provided, fall back to chat_db lookup (original logic)
    elif chat_id and chat_db:
        try:
            chat_files = await asyncio.to_thread(chat_db.get_chat_files, chat_id)
            file_ids_to_filter = [file["file_id"] for file in chat_files if file["included"]]
            if file_ids_to_filter:
                logger.info(f"Using {len(file_ids_to_filter)} included files for chat {chat_id} from DB")
        except Exception as e:
            logger.warning(f"Error getting included files for chat {chat_id} from DB: {e}")
    else:
        # No specific file IDs provided via parameter or chat context
        logger.info("No specific file IDs for filtering. Performing general search.")

    # Prepare filter information for cache lookup
    filter_info = {
        'document_type': document_type,
        'included_file_ids': file_ids_to_filter
    }

    # Check semantic cache first
    if semantic_search_cache:
        try:
            cached_result = await semantic_search_cache.find_semantic_match(
                new_query=query_or_identifier,
                vector_store_id=vs_id,
                filters=filter_info,
                chat_id=chat_id or "default"
            )

            if cached_result:
                logger.info(f"[SEMANTIC CACHE HIT] Retrieved semantically equivalent result for query: '{query_or_identifier[:30]}...'")
                # Ensure the cached result is the correct type
                if isinstance(cached_result, dict) and 'content' in cached_result and 'source_filename' in cached_result:
                    return RetrievalSuccess(**cached_result)
                return cached_result
        except Exception as cache_err:
            logger.warning(f"Semantic cache lookup failed: {cache_err}. Falling back to direct search.")

    # We've already determined file_ids_to_filter above, so we'll use that instead of fetching again

    # First try with a filter for more precise results
    try:
        # Build filters
        filters = []

        # Add document type filter if specified and not a general type
        if document_type and document_type.lower() not in ["kb", "knowledge base", "general", "unknown", "none", ""]:
            logger.info(f"Adding document_type filter for: {document_type}")
            filters.append({"type": "eq", "key": "document_type", "value": document_type})
        else:
            logger.info(f"Not filtering by document_type: '{document_type}' is considered general")

        # Add file filter if we have files to filter
        if file_ids_to_filter:
            # OpenAI API doesn't support 'in' filter type, so we need to use 'or' with multiple 'eq' filters
            if len(file_ids_to_filter) == 1:
                # If only one file ID, use a simple 'eq' filter
                filters.append({"type": "eq", "key": "id", "value": file_ids_to_filter[0]})
            elif len(file_ids_to_filter) > 1:
                # If multiple file IDs, use 'or' with multiple 'eq' filters
                file_filters = []
                for file_id in file_ids_to_filter:
                    file_filters.append({"type": "eq", "key": "id", "value": file_id})

                # Add the combined OR filter
                if file_filters:
                    filters.append({"type": "or", "filters": file_filters})

        # Create filter object if we have any filters
        filter_obj = None
        if len(filters) > 1:
            # If multiple filters, combine with AND
            filter_obj = {"type": "and", "filters": filters}
        elif len(filters) == 1:
            # If single filter, use it directly
            filter_obj = filters[0]

        # Define search variants to run in parallel
        search_tasks = []
        search_variant_descriptions = []

        # Special case for filename-based searches
        if '.pdf' in query_or_identifier.lower() or '.doc' in query_or_identifier.lower() or '.txt' in query_or_identifier.lower():
            # Extract the filename from the query
            filename_match = re.search(r'([a-zA-Z0-9_-]+\.(pdf|doc|docx|txt))', query_or_identifier)
            if filename_match:
                filename = filename_match.group(1)
                logger.info(f"Detected filename in query: {filename}")

                # Try to find files by name using a direct API call
                try:
                    # List files in the vector store
                    files_response = await asyncio.to_thread(tool_client.vector_stores.files.list, vector_store_id=vs_id)

                    # Find files that match the filename
                    matching_files = []
                    for file in files_response.data:
                        file_info = await asyncio.to_thread(tool_client.files.retrieve, file_id=file.id)
                        if filename.lower() in file_info.filename.lower():
                            matching_files.append(file.id)

                    if matching_files:
                        logger.info(f"Found matching files by name: {matching_files}")
                        # Add a filename-based search variant
                        for file_id in matching_files:
                            file_filter = {"type": "eq", "key": "file_id", "value": file_id}
                            search_params = {
                                "vector_store_id": vs_id,
                                "query": "document content",  # Generic query to get content
                                "filters": file_filter,
                                "max_num_results": MAX_SEARCH_RESULTS_TOOL,
                                "ranking_options": {"ranker": SEARCH_RANKER}
                            }
                            search_tasks.append(asyncio.to_thread(tool_client.vector_stores.search, **search_params))
                            search_variant_descriptions.append(f"Filename match filter: {json.dumps(file_filter, indent=2)}")
                except Exception as e:
                    logger.warning(f"Error in filename-based search: {e}")

        # Variant 1: Search with all filters (if we have any)
        if filter_obj:
            search_params = {
                "vector_store_id": vs_id,
                "query": query_or_identifier,
                "filters": filter_obj,
                "max_num_results": MAX_SEARCH_RESULTS_TOOL,
                "ranking_options": {"ranker": SEARCH_RANKER}
            }
            search_tasks.append(asyncio.to_thread(tool_client.vector_stores.search, **search_params))
            search_variant_descriptions.append(f"All filters: {json.dumps(filter_obj, indent=2)}")

        # Variant 2: Search with just file filters (if we have files to filter)
        if file_ids_to_filter:
            # Create file filter using supported filter types
            if len(file_ids_to_filter) == 1:
                file_filter = {"type": "eq", "key": "id", "value": file_ids_to_filter[0]}
            else:
                file_filters = []
                for file_id in file_ids_to_filter:
                    file_filters.append({"type": "eq", "key": "id", "value": file_id})
                file_filter = {"type": "or", "filters": file_filters}

            # Only add this variant if it's different from the first one
            if not filter_obj or filter_obj != file_filter:
                search_params = {
                    "vector_store_id": vs_id,
                    "query": query_or_identifier,
                    "filters": file_filter,
                    "max_num_results": MAX_SEARCH_RESULTS_TOOL,
                    "ranking_options": {"ranker": SEARCH_RANKER}
                }
                search_tasks.append(asyncio.to_thread(tool_client.vector_stores.search, **search_params))
                search_variant_descriptions.append(f"File-only filter: {json.dumps(file_filter, indent=2)}")

        # Variant 3: Search with just document type filter (if specified and not general)
        if document_type and document_type.lower() not in ["kb", "knowledge base", "general", "unknown", "none", ""]:
            doc_type_filter = {"type": "eq", "key": "document_type", "value": document_type}

            # Only add this variant if it's different from previous ones
            if (not filter_obj or filter_obj != doc_type_filter) and (not file_ids_to_filter or doc_type_filter != file_filter):
                search_params = {
                    "vector_store_id": vs_id,
                    "query": query_or_identifier,
                    "filters": doc_type_filter,
                    "max_num_results": MAX_SEARCH_RESULTS_TOOL,
                    "ranking_options": {"ranker": SEARCH_RANKER}
                }
                search_tasks.append(asyncio.to_thread(tool_client.vector_stores.search, **search_params))
                search_variant_descriptions.append(f"Document-type-only filter: {json.dumps(doc_type_filter, indent=2)}")

        # Variant 4: Search without any filters (semantic fallback)
        search_params = {
            "vector_store_id": vs_id,
            "query": query_or_identifier,
            "max_num_results": MAX_SEARCH_RESULTS_TOOL,
            "ranking_options": {"ranker": SEARCH_RANKER}
        }
        search_tasks.append(asyncio.to_thread(tool_client.vector_stores.search, **search_params))
        search_variant_descriptions.append("No filters (semantic fallback)")

        # Log the search variants
        logger.info(f"[CACHE MISS] Running {len(search_tasks)} parallel search variants for query: '{query_or_identifier[:50]}...'")
        for i, desc in enumerate(search_variant_descriptions):
            logger.info(f"Search variant {i+1}: {desc}")

        # Run all search variants in parallel
        search_results_list = await asyncio.gather(*search_tasks)

        # Process results from all variants
        combined_results = []
        result_counts = []

        for i, results in enumerate(search_results_list):
            if results and results.data:
                count = len(results.data)
                result_counts.append(count)
                logger.info(f"Search variant {i+1} returned {count} results")
                combined_results.extend(results.data)
            else:
                result_counts.append(0)
                logger.info(f"Search variant {i+1} returned no results")

        # If we have any results, use them
        if combined_results:
            # Deduplicate results based on file_id and content hash
            unique_results = {}
            for res in combined_results:
                # Create a unique key for each result based on file_id and content hash
                content_hash = hashlib.md5(''.join(part.text for part in res.content if part.type == 'text').encode()).hexdigest()
                key = f"{res.file_id}_{content_hash}"

                # Only keep the result with the highest score if we have duplicates
                if key not in unique_results or res.score > unique_results[key].score:
                    unique_results[key] = res

            # Convert back to list and sort by score
            deduplicated_results = list(unique_results.values())
            deduplicated_results.sort(key=lambda x: x.score, reverse=True)

            # Limit to MAX_SEARCH_RESULTS_TOOL
            final_results = deduplicated_results[:MAX_SEARCH_RESULTS_TOOL]

            logger.info(f"Combined {sum(result_counts)} results, deduplicated to {len(deduplicated_results)}, returning top {len(final_results)}")

            # Create a dummy search results object with our final results
            search_results = type('obj', (object,), {'data': final_results})
        else:
            # No results from any variant
            search_results = None

        if search_results and search_results.data:
            content = "\n\n".join(re.sub(r'\s+', ' ', part.text).strip() for res in search_results.data for part in res.content if part.type == 'text')
            source_filename = search_results.data[0].filename or f"FileID:{search_results.data[0].file_id[-6:]}"
            logger.info(f"[Tool Result] KB Content Found for query '{query_or_identifier[:30]}...'. Len: {len(content)}")

            # Collect source metadata
            sources = []
            for res in search_results.data:
                source = SourceMetadata(
                    file_id=res.file_id,
                    file_name=res.filename or f"FileID:{res.file_id[-6:]}",
                    section=None,  # Would need to extract from metadata if available
                    confidence=res.score  # Assuming score is available
                )
                sources.append(source)

            # Create result object with sources
            result = RetrievalSuccess(
                content=content,
                source_filename=source_filename,
                sources=sources
            )

            # Store in semantic cache
            if semantic_search_cache:
                try:
                    await semantic_search_cache.set(
                        vector_store_id=vs_id,
                        query=query_or_identifier,
                        filters=filter_info,
                        chat_id=chat_id or "default",
                        document_type=document_type,
                        result=result
                    )
                    logger.info(f"[CACHE SET] Stored new result in semantic cache")
                except Exception as cache_err:
                    logger.warning(f"Failed to store result in semantic cache: {cache_err}")

            return result
        else:
            logger.warning(f"[Tool Result] No KB content found for query: '{query_or_identifier[:50]}...'")
            error_result = RetrievalError(error_message=f"No KB content found for query related to '{document_type}'.")

            # Also cache negative results to avoid repeated failed searches
            if semantic_search_cache:
                try:
                    await semantic_search_cache.set(
                        vector_store_id=vs_id,
                        query=query_or_identifier,
                        filters=filter_info,
                        chat_id=chat_id or "default",
                        document_type=document_type,
                        result=error_result
                    )
                    logger.info(f"[CACHE SET] Stored negative result in semantic cache")
                except Exception as cache_err:
                    logger.warning(f"Failed to store negative result in semantic cache: {cache_err}")

            return error_result
    except Exception as e:
        logger.error(f"[Tool Error] KB Search failed for query '{query_or_identifier[:30]}...': {e}", exc_info=True)
        return RetrievalError(error_message=f"KB Search error: {str(e)}")
