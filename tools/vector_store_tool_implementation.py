"""
Vector Store Tool Implementation for Knowledge Base UX Optimization

This module contains the implementation of the Vector Store Tool
that encapsulates vector store interactions for agent use.
"""

import asyncio
import json
import logging
import re
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
from agents import Agent, Runner, RunContextWrapper, function_tool

# Configure logging
logger = logging.getLogger(__name__)

# --- Models for Vector Store Tool ---

class VectorSearchDocument(BaseModel):
    """Model representing a document retrieved from vector search."""
    file_id: str = Field(description="ID of the file in the vector store")
    file_name: str = Field(description="Name of the file")
    content: str = Field(description="Text content of the document")
    score: float = Field(description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class VectorSearchSuccess(BaseModel):
    """Model for successful vector store search results."""
    documents: List[VectorSearchDocument] = Field(
        description="List of retrieved documents"
    )
    query: str = Field(
        description="The query that was used for the search"
    )
    document_type: str = Field(
        description="The document type that was used for filtering"
    )

class VectorSearchError(BaseModel):
    """Model for vector store search errors."""
    error_message: str = Field(description="Error message")
    query: str = Field(description="The query that was attempted")
    document_type: Optional[str] = Field(None, description="The document type that was used")

# --- Vector Store Tool Implementation ---

@function_tool(strict_mode=False)
async def search_vector_store(
    ctx: RunContextWrapper, 
    query: str, 
    document_type: str = "general", 
    max_results: int = 5, 
    use_hybrid_search: bool = False
) -> Union[VectorSearchSuccess, VectorSearchError]:
    """Searches the vector store for relevant documents matching the query.
    
    Args:
        query: The search query
        document_type: Optional document type to filter by
        max_results: Maximum number of results to return
        use_hybrid_search: Whether to use hybrid search (if available)
        
    Returns:
        Search results with content and metadata
    """
    logger.info(f"[Tool Call] search_vector_store: query='{query[:50]}...', type='{document_type}'")
    
    # Get context variables
    tool_context = ctx.context
    vs_id = tool_context.get("vector_store_id")
    tool_client = tool_context.get("client")
    
    if not tool_client or not vs_id:
        return VectorSearchError(
            error_message="Vector store configuration error.",
            query=query,
            document_type=document_type
        )
    
    try:
        # Build filter if document type is specified
        filter_obj = None
        if document_type and document_type.lower() not in ["kb", "knowledge base", "general", "unknown", "none", ""]:
            filter_obj = {"type": "eq", "key": "document_type", "value": document_type}
        
        # Prepare search parameters
        search_params = {
            "vector_store_id": vs_id,
            "query": query,
            "max_num_results": max_results
        }
        
        if filter_obj:
            search_params["filters"] = filter_obj
        
        # Add hybrid search parameters if requested and available
        # Note: This is a placeholder - actual implementation would depend on the OpenAI API capabilities
        if use_hybrid_search:
            search_params["search_type"] = "hybrid"
            search_params["hybrid_search_options"] = {
                "alpha": 0.5  # Balance between vector and keyword search
            }
        
        # Perform the search
        logger.info(f"Executing vector store search with params: {json.dumps(search_params, default=str)}")
        search_results = await asyncio.to_thread(tool_client.vector_stores.search, **search_params)
        
        if not search_results or not search_results.data or len(search_results.data) == 0:
            logger.warning(f"No results found for query: '{query[:50]}...'")
            return VectorSearchError(
                error_message=f"No documents found matching the query for document type '{document_type}'.",
                query=query,
                document_type=document_type
            )
        
        # Process the results
        documents = []
        for res in search_results.data:
            # Extract text content
            content = "\n\n".join(re.sub(r'\s+', ' ', part.text).strip() 
                                for part in res.content 
                                if part.type == 'text')
            
            # Create document object
            doc = VectorSearchDocument(
                file_id=res.file_id,
                file_name=res.filename or f"FileID:{res.file_id[-6:]}",
                content=content,
                score=res.score,
                metadata={
                    "attributes": getattr(res, "attributes", {}),
                    "created_at": getattr(res, "created_at", None)
                }
            )
            documents.append(doc)
        
        logger.info(f"[Tool Result] Found {len(documents)} documents for query '{query[:30]}...'")
        
        return VectorSearchSuccess(
            documents=documents,
            query=query,
            document_type=document_type
        )
        
    except Exception as e:
        logger.error(f"[Tool Error] Vector store search failed: {e}", exc_info=True)
        return VectorSearchError(
            error_message=f"Search error: {str(e)}",
            query=query,
            document_type=document_type
        )

# --- Knowledge Retrieval Agent Definition ---

knowledge_retrieval_agent = Agent(
    name="KnowledgeRetrievalAgent",
    instructions="""Retrieve relevant knowledge from the vector store.
    
    Your job is to find the most relevant information in the knowledge base to answer the user's query.
    
    RETRIEVAL STRATEGY:
    
    1. ANALYZE THE QUERY:
       - Identify the main topic and specific information requested
       - Determine the appropriate document type to search in
    
    2. FORMULATE SEARCH APPROACH:
       - Use the search_vector_store tool to find relevant documents
       - Start with specific document types if appropriate
       - Fall back to general search if needed
    
    3. EVALUATE RESULTS:
       - Check if the retrieved documents contain the requested information
       - If results are insufficient, try alternative queries or document types
       - Combine information from multiple documents if necessary
    
    4. RETURN COMPREHENSIVE INFORMATION:
       - Include all relevant information from the retrieved documents
       - Maintain source attribution for each piece of information
       - Ensure the information is accurate and complete
    
    OUTPUT FORMAT:
    Return a JSON object with:
    - retrieved_content: The combined relevant content from all documents
    - sources: List of source documents with metadata
    - query_used: The query that was used for the successful retrieval
    """,
    tools=[search_vector_store],
    model="gpt-4o-mini",  # Use appropriate model
    output_type=VectorSearchSuccess  # This could be enhanced with a custom output type
)

# --- Example of how to use the Knowledge Retrieval Agent in the workflow ---

"""
async def agent_driven_rag_workflow(user_query: str, history: List[Dict[str, str]], workflow_context: Dict, vs_id: Optional[str] = None) -> str:
    """Implements an agent-driven RAG workflow.
    
    This is a template for how to implement a fully agent-driven workflow.
    The actual implementation would need to be integrated with the existing functions.
    """
    logger.info(f"Running Agent-Driven RAG workflow for query: '{user_query[:50]}...'")
    try:
        # First, generate optimized search queries
        search_query_plan = await generate_optimized_search_query(user_query, history)
        logger.info(f"Generated optimized query: '{search_query_plan.primary_query[:50]}...'")
        
        # Set up context for knowledge retrieval
        retrieval_context = workflow_context.copy()
        retrieval_context["vector_store_id"] = vs_id
        
        # Run the knowledge retrieval agent
        retrieval_input = {
            "user_query": user_query,
            "optimized_query": search_query_plan.primary_query,
            "alternative_queries": search_query_plan.alternative_queries,
            "document_type": search_query_plan.document_type_filter
        }
        
        retrieval_result = await Runner.run(
            knowledge_retrieval_agent,
            input=retrieval_input,
            context=retrieval_context
        )
        
        # Process the retrieval result
        if isinstance(retrieval_result.final_output, VectorSearchSuccess):
            kb_content = "\n\n".join(doc.content for doc in retrieval_result.final_output.documents)
            sources = [{"file_name": doc.file_name, "score": doc.score} for doc in retrieval_result.final_output.documents]
            
            # Create a prompt that includes the query and KB content
            prompt = f"Answer the following question using ONLY the knowledge base content provided below.\n\nQuestion: {user_query}\n\n"
            prompt += "IMPORTANT: If the knowledge base content does not contain information to answer this question, clearly state this limitation. DO NOT fabricate information.\n\n"
            prompt += f"Relevant Knowledge Base Content:\n{kb_content}\n\n"
            prompt += f"Sources:\n{json.dumps(sources, indent=2)}"
            
            # Run the final synthesizer
            synthesis_messages = history + [{"role": "user", "content": prompt}]
            final_synth_raw = await Runner.run(final_synthesizer_agent, input=synthesis_messages, context=workflow_context)
            
            # Extract and return the final answer
            final_markdown_response = extract_final_answer(final_synth_raw)
            return final_markdown_response
        else:
            # Handle retrieval error
            error_message = "I couldn't find specific information about this topic in the knowledge base."
            if hasattr(retrieval_result.final_output, 'error_message'):
                error_message = retrieval_result.final_output.error_message
                
            return f"# No Information Found\n\n{error_message}\n\nPlease try a different query or check if the relevant documents are included in the knowledge base."
            
    except Exception as e:
        logger.error(f"Agent-driven RAG workflow failed: {e}", exc_info=True)
        return f"Sorry, an error occurred during processing: {html.escape(str(e))}"
"""
