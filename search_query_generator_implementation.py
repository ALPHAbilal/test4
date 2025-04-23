"""
Search Query Generator Implementation for Knowledge Base UX Optimization

This module contains the implementation of the SearchQueryGeneratorAgent
that transforms user queries into optimized vector search queries.
"""

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from agents import Agent, Runner, RunContextWrapper

# --- Models for Search Query Generation ---

class SearchQueryPlan(BaseModel):
    """Model for the output of the SearchQueryGeneratorAgent."""
    primary_query: str = Field(
        description="The main optimized query for vector search"
    )
    alternative_queries: List[str] = Field(
        default_factory=list,
        description="Alternative queries to try if the primary query fails"
    )
    document_type_filter: str = Field(
        default="general",
        description="Suggested document type to filter by"
    )
    reasoning: str = Field(
        description="Explanation of how the queries were formulated"
    )

# --- Search Query Generator Agent Definition ---

search_query_generator_agent = Agent(
    name="SearchQueryGeneratorAgent",
    instructions="""Generate optimal search queries for vector store retrieval.
    
    Your job is to transform the user's natural language query into the most effective search query
    for a vector store. Vector stores work best with clear, concise queries that contain key terms
    and concepts.
    
    GUIDELINES FOR QUERY OPTIMIZATION:
    
    1. EXTRACT KEY CONCEPTS:
       - Identify the main entities, topics, and concepts in the user's query
       - Focus on nouns, technical terms, and specific identifiers
    
    2. REMOVE CONVERSATIONAL ELEMENTS:
       - Remove phrases like "Can you tell me about..." or "I'd like to know..."
       - Remove filler words, pleasantries, and conversational language
    
    3. ADD RELEVANT SYNONYMS:
       - Include alternative terms for key concepts
       - Consider industry-specific terminology
    
    4. HANDLE COMPLEX QUERIES:
       - For multi-part questions, create a primary query for the main question
       - Create alternative queries for sub-questions or different aspects
    
    5. USE CONVERSATION HISTORY:
       - Incorporate relevant context from previous messages
       - Resolve pronouns and references to previous topics
    
    6. DETERMINE DOCUMENT TYPE:
       - Suggest a document type filter if the query clearly relates to a specific type
       - Use "general" if uncertain or if the query spans multiple document types
    
    OUTPUT FORMAT:
    Return a JSON object with:
    - primary_query: The main optimized query
    - alternative_queries: List of alternative queries to try if primary fails
    - document_type_filter: Suggested document type to filter by (or "general")
    - reasoning: Brief explanation of your query formulation process
    """,
    model="gpt-4o-mini",  # Use appropriate model
    output_type=SearchQueryPlan
)

# --- Function to integrate the agent into the workflow ---

async def generate_optimized_search_query(user_query: str, conversation_history: List[Dict[str, str]] = None) -> SearchQueryPlan:
    """
    Generate an optimized search query for vector store retrieval.
    
    Args:
        user_query: The original user query
        conversation_history: Optional conversation history for context
        
    Returns:
        SearchQueryPlan object containing optimized queries
    """
    # Prepare input for the agent
    if conversation_history:
        # Include recent conversation history (last 3 exchanges)
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        input_text = f"""User Query: {user_query}

Recent Conversation History:
{history_text}

Based on this query and conversation history, generate optimized search queries for vector store retrieval."""
    else:
        input_text = f"""User Query: {user_query}

Generate optimized search queries for vector store retrieval based on this query."""

    # Run the agent
    result = await Runner.run(search_query_generator_agent, input=input_text)
    
    # Return the search query plan
    return result.final_output

# --- Example of how to modify run_standard_agent_rag to use the query generator ---

"""
async def enhanced_run_standard_agent_rag(user_query: str, history: List[Dict[str, str]], workflow_context: Dict, vs_id: Optional[str] = None) -> str:
    """Implements an enhanced RAG workflow with query optimization.
    
    This is a template for how to modify the existing function.
    The actual implementation would need to be integrated with the existing function.
    """
    logger.info(f"Running Enhanced RAG workflow for query: '{user_query[:50]}...'")
    try:
        # First, generate optimized search queries
        search_query_plan = await generate_optimized_search_query(user_query, history)
        logger.info(f"Generated optimized query: '{search_query_plan.primary_query[:50]}...'")
        logger.info(f"Suggested document type: {search_query_plan.document_type_filter}")
        
        # Use the optimized query and document type for KB retrieval
        kb_content = ""
        if vs_id:
            logger.info(f"Retrieving KB content with optimized query")
            kb_context = workflow_context.copy()
            
            # Use the optimized query and document type
            kb_query = search_query_plan.primary_query
            document_type = search_query_plan.document_type_filter
            
            # Get KB content using the minimal data gathering agent
            kb_data_raw = await Runner.run(
                data_gathering_agent_minimal,
                input=f"Get KB content about '{document_type}' related to: {kb_query}",
                context=kb_context
            )
            kb_data = kb_data_raw.final_output
            
            # If first attempt fails, try alternative queries
            if isinstance(kb_data, RetrievalError) and search_query_plan.alternative_queries:
                logger.info(f"First KB retrieval attempt failed. Trying alternative queries.")
                
                for alt_query in search_query_plan.alternative_queries:
                    logger.info(f"Trying alternative query: '{alt_query[:50]}...'")
                    
                    kb_data_raw = await Runner.run(
                        data_gathering_agent_minimal,
                        input=f"Get KB content about '{document_type}' related to: {alt_query}",
                        context=kb_context
                    )
                    kb_data = kb_data_raw.final_output
                    
                    if isinstance(kb_data, RetrievalSuccess):
                        logger.info(f"Alternative query succeeded")
                        break
            
            # Process the results as in the original function...
            
        # Continue with the rest of the function as before...
        
    except Exception as e:
        logger.error(f"Enhanced RAG workflow failed: {e}", exc_info=True)
        return f"Sorry, an error occurred during processing: {html.escape(str(e))}"
"""
