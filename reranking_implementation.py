"""
Re-ranking Implementation for Knowledge Base UX Optimization

This module contains the implementation of the re-ranking approach
to improve the relevance of retrieved documents.
"""

from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
from agents import Agent, Runner, RunContextWrapper
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)

# --- Models for Re-ranking ---

class DocumentForRanking(BaseModel):
    """Model representing a document to be ranked."""
    document_id: str = Field(description="Unique identifier for the document")
    file_name: str = Field(description="Name of the file")
    content: str = Field(description="Text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    original_score: float = Field(description="Original retrieval score")

class ReRankResult(BaseModel):
    """Model for the output of the ReRankAgent."""
    ranked_documents: List[str] = Field(
        description="List of document IDs sorted by relevance (most relevant first)"
    )
    relevance_scores: Dict[str, float] = Field(
        description="Dictionary mapping document IDs to relevance scores (0-10)"
    )
    reasoning: str = Field(
        description="Explanation of the ranking decisions"
    )

# --- Re-Rank Agent Definition ---

rerank_agent = Agent(
    name="ReRankAgent",
    instructions="""Re-rank retrieved documents based on relevance to the user query.
    
    You will receive a list of documents and the original user query.
    Your task is to re-rank these documents based on their relevance to the query.
    
    EVALUATION CRITERIA:
    
    1. DIRECT RELEVANCE:
       - How directly does the document answer the specific question?
       - Does it contain the exact information requested?
    
    2. INFORMATION QUALITY:
       - How specific and detailed is the information?
       - Is it comprehensive for the query topic?
    
    3. INFORMATION RELIABILITY:
       - Is the source authoritative for this type of information?
       - How recent or up-to-date is the information?
    
    4. CONTEXT MATCH:
       - Does the document match the implied context of the query?
       - Does it address the likely intent behind the query?
    
    SCORING SYSTEM:
    - 9-10: Perfect match, directly answers the query with high-quality information
    - 7-8: Strong match, contains most of the requested information
    - 5-6: Moderate match, contains some relevant information but may be incomplete
    - 3-4: Weak match, tangentially related to the query
    - 1-2: Poor match, minimally related to the query
    - 0: Not relevant at all
    
    OUTPUT FORMAT:
    Return a JSON object with:
    - ranked_documents: List of document IDs sorted by relevance (most relevant first)
    - relevance_scores: Dictionary mapping document IDs to scores (0-10)
    - reasoning: Brief explanation of your ranking decisions
    """,
    model="gpt-4o-mini",  # Use appropriate model
    output_type=ReRankResult
)

# --- Functions for re-ranking implementation ---

async def retrieve_with_reranking(
    client, 
    vector_store_id: str, 
    query: str, 
    filters: Optional[Dict] = None, 
    initial_k: int = 15, 
    final_k: int = 5
) -> List[Any]:
    """
    Retrieve documents from vector store and re-rank them.
    
    Args:
        client: OpenAI client
        vector_store_id: ID of the vector store
        query: Search query
        filters: Optional filters for the search
        initial_k: Number of initial results to retrieve
        final_k: Number of final results to return after re-ranking
        
    Returns:
        List of re-ranked documents
    """
    logger.info(f"Retrieving initial {initial_k} results for query: '{query[:50]}...'")
    
    # Perform initial retrieval with larger k
    search_params = {
        "vector_store_id": vector_store_id,
        "query": query,
        "max_num_results": initial_k
    }
    
    if filters:
        search_params["filters"] = filters
    
    # Perform the search
    search_results = await asyncio.to_thread(client.vector_stores.search, **search_params)
    
    if not search_results or not search_results.data or len(search_results.data) == 0:
        logger.warning(f"No results found for query: '{query[:50]}...'")
        return []
    
    logger.info(f"Retrieved {len(search_results.data)} initial results")
    
    # Prepare documents for re-ranking
    documents_for_ranking = []
    for i, res in enumerate(search_results.data):
        # Extract text content
        content = "\n\n".join(part.text for part in res.content if part.type == 'text')
        
        # Create document object
        doc = DocumentForRanking(
            document_id=f"doc_{i}",
            file_name=res.filename or f"FileID:{res.file_id[-6:]}",
            content=content,
            metadata={
                "file_id": res.file_id,
                "original_index": i
            },
            original_score=res.score
        )
        documents_for_ranking.append(doc)
    
    # Prepare input for the re-ranking agent
    rerank_input = {
        "user_query": query,
        "documents": [doc.dict() for doc in documents_for_ranking]
    }
    
    # Run the re-ranking agent
    logger.info(f"Running re-ranking agent on {len(documents_for_ranking)} documents")
    rerank_result = await Runner.run(rerank_agent, input=rerank_input)
    rerank_output = rerank_result.final_output
    
    logger.info(f"Re-ranking complete. Reasoning: {rerank_output.reasoning[:100]}...")
    
    # Sort the original results based on the new ranking
    reranked_results = []
    for doc_id in rerank_output.ranked_documents[:final_k]:  # Take only the top final_k
        # Find the original document index
        for doc in documents_for_ranking:
            if doc.document_id == doc_id:
                original_index = doc.metadata["original_index"]
                original_result = search_results.data[original_index]
                
                # Add the new relevance score to the result
                original_result.relevance_score = rerank_output.relevance_scores.get(doc_id, 0)
                
                reranked_results.append(original_result)
                break
    
    logger.info(f"Returning top {len(reranked_results)} re-ranked results")
    return reranked_results

# --- Example of how to modify get_kb_document_content to use re-ranking ---

"""
@function_tool(strict_mode=False)
async def enhanced_get_kb_document_content(ctx: RunContextWrapper, document_type: str, query_or_identifier: str) -> Union[RetrievalSuccess, RetrievalError]:
    """Enhanced version of get_kb_document_content that includes re-ranking.
    
    This is a template for how to modify the existing function.
    The actual implementation would need to be integrated with the existing function.
    """
    logger.info(f"[Tool Call] enhanced_get_kb_document_content: type='{document_type}', query='{query_or_identifier[:50]}...'")
    tool_context = ctx.context
    vs_id = tool_context.get("vector_store_id")
    tool_client = tool_context.get("client")
    chat_id = tool_context.get("chat_id")
    
    if not tool_client or not vs_id:
        return RetrievalError(error_message="Tool config error.")
    
    # Check semantic cache first (as in original function)
    
    # Build filters (as in original function)
    
    # Instead of running multiple search variants, use the re-ranking approach
    try:
        # Create filter object
        filter_obj = None
        if document_type and document_type.lower() not in ["kb", "knowledge base", "general", "unknown", "none", ""]:
            filter_obj = {"type": "eq", "key": "document_type", "value": document_type}
        
        # Get re-ranked results
        reranked_results = await retrieve_with_reranking(
            client=tool_client,
            vector_store_id=vs_id,
            query=query_or_identifier,
            filters=filter_obj,
            initial_k=15,  # Retrieve more initial results
            final_k=5      # Return top 5 after re-ranking
        )
        
        if reranked_results:
            # Process results as in the original function, but use the re-ranked results
            content = "\n\n".join(re.sub(r'\s+', ' ', part.text).strip() 
                                for res in reranked_results 
                                for part in res.content 
                                if part.type == 'text')
            
            # Include re-ranking scores in the source metadata
            sources = []
            for res in reranked_results:
                source = SourceMetadata(
                    file_id=res.file_id,
                    file_name=res.filename or f"FileID:{res.file_id[-6:]}",
                    section=None,
                    confidence=getattr(res, 'relevance_score', res.score)  # Use re-ranking score if available
                )
                sources.append(source)
            
            # Return enhanced result with sources
            result = EnhancedRetrievalSuccess(
                content=content,
                sources=sources
            )
            
            # Cache the result (as in original function)
            
            return result
        else:
            # Handle no results case (as in original function)
            
    except Exception as e:
        # Handle errors (as in original function)
"""
