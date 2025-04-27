"""
Enhanced Search

This module provides enhanced search capabilities using document maps.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from core.document_map_storage import get_document_map
from agents.enhanced_document_analyzer import RunContextWrapper

logger = logging.getLogger(__name__)

def extract_document_id_from_query(query: str) -> Optional[str]:
    """Extract document ID from query if present."""
    # Simple implementation - look for doc_* pattern
    import re
    match = re.search(r'doc_\d+_\w+', query)
    if match:
        return match.group(0)
    return None

def extract_text_from_content(content) -> str:
    """Extract text from content object."""
    if content is None:
        return "No content available"

    # If content is a string
    if isinstance(content, str):
        return content

    # If content is an object with text attribute
    if hasattr(content, 'text'):
        return content.text

    # If content is an object with value attribute
    if hasattr(content, 'value'):
        return content.value

    # If content can be converted to string
    try:
        return str(content)
    except:
        return "Content not extractable"

async def identify_relevant_sections(ctx: RunContextWrapper, query: str, document_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify sections relevant to a query."""
    client = ctx.get("client")

    # Extract section information
    sections = document_map.get("sections", [])
    if not sections:
        logger.warning(f"No sections found in document map {document_map.get('document_id')}")
        return []

    # Create section summaries
    section_summaries = [
        f"Section: {s.get('title', 'Untitled')}\nSummary: {s.get('summary', 'No summary')}\nKeywords: {', '.join(s.get('keywords', []))}"
        for s in sections
    ]

    # Use OpenAI to identify relevant sections
    section_text = '\n\n'.join(section_summaries)
    prompt = f"""
    Based on the user's query, identify which sections of the document are most relevant.

    User query: "{query}"

    Document sections:
    {section_text}

    Return a JSON object with an array named "relevant_sections" containing the indices of the relevant sections (0-based).
    Only include sections that are truly relevant to answering the query.
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a document search expert."},
                    {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        relevant_indices = result.get("relevant_sections", [])

        # Get the relevant sections
        relevant_sections = [sections[i] for i in relevant_indices if i < len(sections)]
        logger.info(f"Identified {len(relevant_sections)} relevant sections for query: {query}")

        return relevant_sections
    except Exception as e:
        logger.error(f"Error identifying relevant sections: {e}")
        return []

async def document_specific_search(ctx: RunContextWrapper, query: str, document_id: str, vector_store_id: str) -> Dict[str, Any]:
    """Search within a specific document using its document map."""
    client = ctx.get("client")

    # Step 1: Retrieve document map
    document_map = await get_document_map(document_id)
    if not document_map:
        logger.warning(f"Document map not found for {document_id}")
        return await general_search(ctx, query, vector_store_id)

    # Step 2: Identify relevant sections
    relevant_sections = await identify_relevant_sections(ctx, query, document_map)

    # Step 3: Perform targeted searches
    all_results = []

    if relevant_sections:
        # Search in relevant sections
        for section in relevant_sections:
            # Create filter for this section's part
            filter_params = {
                "document_id": document_id,
                "part_number": section.get("part_number")
            }

            try:
                # Perform search
                section_results = await client.vector_stores.search(
                    vector_store_id=vector_store_id,
                    query=query,
                    filter=filter_params
                )

                # Process results
                for result in section_results.data:
                    all_results.append({
                        "file_id": result.file_id,
                        "score": result.score,
                        "content": result.content,
                        "section": section.get("title"),
                        "section_summary": section.get("summary")
                    })
            except Exception as e:
                logger.error(f"Error searching section {section.get('title')}: {e}")
    else:
        # Fallback: search across all parts of the document
        filter_params = {"document_id": document_id}

        try:
            # Perform search
            results = await client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
                filter=filter_params
            )

            # Process results
            for result in results.data:
                all_results.append({
                    "file_id": result.file_id,
                    "score": result.score,
                    "content": result.content
                })
        except Exception as e:
            logger.error(f"Error searching document {document_id}: {e}")

    # Sort results by score
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "results": all_results,
        "document_map": document_map,
        "query": query
    }

async def general_search(ctx: RunContextWrapper, query: str, vector_store_id: str) -> Dict[str, Any]:
    """Perform general search across all documents."""
    client = ctx.get("client")

    try:
        # Perform search
        results = await client.vector_stores.search(
            vector_store_id=vector_store_id,
            query=query
        )

        # Process results
        all_results = []
        for result in results.data:
            result_data = {
                "file_id": result.file_id,
                "score": result.score,
                "content": result.content
            }

            # Try to get document_id from metadata
            if hasattr(result, "attributes") and result.attributes:
                document_id = result.attributes.get("document_id")
                if document_id:
                    result_data["document_id"] = document_id

                    # Try to get document map
                    document_map = await get_document_map(document_id)
                    if document_map:
                        result_data["document_type"] = document_map.get("document_type")

            all_results.append(result_data)

        return {
            "results": all_results,
            "query": query
        }
    except Exception as e:
        logger.error(f"Error performing general search: {e}")
        return {
            "results": [],
            "query": query,
            "error": str(e)
        }

async def enhanced_search(ctx: RunContextWrapper, query: str, vector_store_id: str) -> Dict[str, Any]:
    """Perform enhanced search using document maps."""
    # Step 1: Check if query mentions a specific document
    document_id = extract_document_id_from_query(query)

    if document_id:
        # Document-specific search
        return await document_specific_search(ctx, query, document_id, vector_store_id)
    else:
        # General search across all documents
        return await general_search(ctx, query, vector_store_id)
