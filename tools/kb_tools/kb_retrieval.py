"""
Knowledge Base Retrieval Tools

This module provides tools for retrieving content from the knowledge base.
"""

import asyncio
import logging
import re
from typing import List, Optional, Union
from agents import function_tool, RunContextWrapper
from pydantic import Field, BaseModel

# Import sequential search implementation
from tools.kb_tools.sequential_search import sequential_search_with_early_termination, prioritize_search_strategies

logger = logging.getLogger(__name__)

# Import constants from app.py
try:
    from app import MAX_SEARCH_RESULTS_TOOL, SEARCH_RANKER
except ImportError:
    # Default values if app.py cannot be imported
    MAX_SEARCH_RESULTS_TOOL = 5
    SEARCH_RANKER = "hybrid"

# Global dictionary to track search strategy performance
STRATEGY_METRICS = {}

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

        # Define search strategies for sequential execution
        search_strategies = []

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
                        # Add filename-based search strategies (high priority)
                        for file_id in matching_files:
                            file_filter = {"type": "eq", "key": "file_id", "value": file_id}
                            search_strategies.append({
                                "name": "filename_match",
                                "description": f"Filename match filter: {filename}",
                                "filters": file_filter,
                                "params": {
                                    "max_num_results": MAX_SEARCH_RESULTS_TOOL,
                                    "ranking_options": {"ranker": SEARCH_RANKER},
                                    "query": "document content"  # Generic query to get content
                                }
                            })
                except Exception as e:
                    logger.warning(f"Error in filename-based search: {e}")

        # Get query analysis if available
        query_analysis = None
        if 'query_analysis' in locals():
            # Use the query analysis we just generated
            pass
        elif ctx:
            # Try to get query analysis from context
            if hasattr(ctx, "get") and ctx.get("query_analysis"):
                # Context is a RunContextWrapper
                query_analysis = ctx.get("query_analysis")
            elif isinstance(ctx, dict) and "query_analysis" in ctx:
                # Context is a dictionary
                query_analysis = ctx["query_analysis"]

        # Determine search parameters based on query analysis or system defaults
        search_params = {
            "max_num_results": MAX_SEARCH_RESULTS_TOOL,
            "ranking_options": {"ranker": SEARCH_RANKER}
        }

        # Only add rewrite_query if we don't have query analysis
        # Otherwise, let the query analysis determine this parameter
        if not query_analysis:
            search_params["rewrite_query"] = True

        # Update parameters if query analysis is available
        if query_analysis:
            # Use recommended parameters from query analysis
            if "recommended_max_results" in query_analysis:
                search_params["max_num_results"] = min(query_analysis["recommended_max_results"], MAX_SEARCH_RESULTS_TOOL)

            # Use search priority from query analysis
            if "search_priority" in query_analysis:
                if query_analysis["search_priority"] == "precision":
                    search_params["ranking_options"] = {"ranker": "best_match"}
                elif query_analysis["search_priority"] == "recall":
                    search_params["ranking_options"] = {"ranker": "hybrid"}

        # Strategy 1: Search with all filters (if we have any)
        if filter_obj:
            search_strategies.append({
                "name": "all_filters",
                "description": f"All filters combined",
                "filters": filter_obj,
                "params": search_params
            })

        # Strategy 2: Search with just file filters (if we have files to filter)
        if file_ids_to_filter:
            # Create file filter using supported filter types
            if len(file_ids_to_filter) == 1:
                file_filter = {"type": "eq", "key": "id", "value": file_ids_to_filter[0]}
            else:
                file_filters = []
                for file_id in file_ids_to_filter:
                    file_filters.append({"type": "eq", "key": "id", "value": file_id})
                file_filter = {"type": "or", "filters": file_filters}

            # Only add this strategy if it's different from the first one
            if not filter_obj or filter_obj != file_filter:
                search_strategies.append({
                    "name": "file_only",
                    "description": "File-only filter",
                    "filters": file_filter,
                    "params": search_params.copy()  # Use the same parameters
                })

        # Strategy 3: Search with just document type filter (if specified)
        if document_type:
            # Check if this is a general document type using query analysis
            is_general_type = False

            if query_analysis and "general_document_types" in query_analysis:
                # Use the agent's knowledge of general document types
                general_types = query_analysis.get("general_document_types", [])
                is_general_type = document_type.lower() in [t.lower() for t in general_types]
            else:
                # Fallback check only if we don't have agent analysis
                is_general_type = document_type.lower() in ["kb", "knowledge base", "general", "unknown", "none", ""]

            # Only create a filter if it's not a general type
            if not is_general_type:
                doc_type_filter = {"type": "eq", "key": "document_type", "value": document_type}

            # Only add this strategy if it's different from previous ones
            if (not filter_obj or filter_obj != doc_type_filter) and (not file_ids_to_filter or doc_type_filter != file_filter):
                # For document type searches, we might want to adjust parameters
                doc_type_params = search_params.copy()

                # If we have query analysis, check if this is the recommended document type
                if query_analysis and query_analysis.get("document_type_hint") == document_type:
                    # This is the recommended document type, increase max results
                    doc_type_params["max_num_results"] = min(doc_type_params["max_num_results"] + 2, MAX_SEARCH_RESULTS_TOOL)

                search_strategies.append({
                    "name": "document_type",
                    "description": f"Document type filter: {document_type}",
                    "filters": doc_type_filter,
                    "params": doc_type_params
                })

        # Strategy 4: Search without any filters (semantic fallback)
        # For semantic fallback, we might want different parameters
        fallback_params = search_params.copy()

        # If this is a summary request, we want more results from the fallback
        if query_analysis and query_analysis.get("is_summary_request"):
            fallback_params["max_num_results"] = min(fallback_params["max_num_results"] + 5, MAX_SEARCH_RESULTS_TOOL)

        search_strategies.append({
            "name": "semantic_fallback",
            "description": "No filters (semantic fallback)",
            "filters": None,
            "params": fallback_params
        })

        # Prioritize strategies based on historical performance
        global STRATEGY_METRICS
        prioritized_strategies = prioritize_search_strategies(search_strategies, STRATEGY_METRICS)

        # Log the search strategies
        logger.info(f"[CACHE MISS] Running sequential search with early termination for query: '{query_or_identifier[:50]}...'")
        logger.info(f"Search strategies in priority order: {', '.join([s['name'] for s in prioritized_strategies])}")

        # Analyze the query using the QueryAnalyzerAgent
        try:
            from agents.query_analyzer_agent import analyze_query

            # Create a context wrapper for the agent
            from agents import RunContextWrapper
            agent_context = RunContextWrapper({"client": tool_client})

            # Analyze the query using the agent
            query_analysis = await analyze_query(query_or_identifier, agent_context)

            # Extract search parameters from analysis
            min_results = query_analysis.get("recommended_min_results", 5)
            max_results = min(query_analysis.get("recommended_max_results", 10), MAX_SEARCH_RESULTS_TOOL)

            # Store query analysis in context for later use if possible
            if hasattr(agent_context, "set"):
                agent_context.set("query_analysis", query_analysis)
            elif isinstance(agent_context, dict):
                agent_context["query_analysis"] = query_analysis

            logger.info(f"Query analysis: type={query_analysis.get('query_type', 'unknown')}, min_results={min_results}, max_results={max_results}, priority={query_analysis.get('search_priority', 'balanced')}")
        except Exception as e:
            logger.error(f"Error analyzing query, using default parameters: {e}")
            # Default parameters if query analysis fails
            min_results = 5
            max_results = MAX_SEARCH_RESULTS_TOOL

        # Execute sequential search with early termination
        processed_results = await sequential_search_with_early_termination(
            client=tool_client,
            vector_store_id=vs_id,
            query=query_or_identifier,
            search_strategies=prioritized_strategies,
            min_results_threshold=min_results,  # Adjusted based on query type
            max_total_results=max_results,      # Adjusted based on query type
            strategy_metrics=STRATEGY_METRICS,
            context=ctx  # Pass the context for query analysis
        )

        # If we have any results, convert them to the expected format
        if processed_results:
            # Convert processed results to the format expected by the rest of the function
            final_results = []

            for result in processed_results:
                # Create a dummy result object with the necessary attributes
                result_obj = type('obj', (object,), {
                    'file_id': result['file_id'],
                    'score': result['score'],
                    'content': [type('obj', (object,), {'type': 'text', 'text': result['content']})],
                    'filename': result.get('metadata', {}).get('original_filename', f"FileID:{result['file_id'][-6:]}")
                })

                final_results.append(result_obj)

            logger.info(f"Sequential search returned {len(final_results)} results")

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
