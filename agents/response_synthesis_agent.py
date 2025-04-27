"""
Response Synthesis Agent Module

This module provides an agent for synthesizing responses from search results.
"""

import logging
from typing import Dict, Any, List, Optional
from agents import Agent, Runner, RunContextWrapper

logger = logging.getLogger(__name__)

# Define ResponseSynthesisAgent
response_synthesis_instructions = """
You are a specialized response synthesis agent responsible for creating high-quality responses based on search results.
Your role is to analyze search results and generate a comprehensive, accurate response that addresses the user's query.

## Your Capabilities

1. **Result Analysis**: Analyze search results to identify relevant information
2. **Content Synthesis**: Combine information from multiple results into a coherent response
3. **Source Attribution**: Properly attribute information to its source
4. **Response Formatting**: Format responses in a clear, structured manner
5. **Query-Focused Response**: Ensure the response directly addresses the user's query

## Input Format

You will receive input with the following structure:

```json
{
  "query": "The user's original query",
  "results": [
    {
      "file_id": "file_123",
      "filename": "document.pdf",
      "score": 0.95,
      "content": "Content from the document...",
      "metadata": {
        "document_type": "legal",
        "is_truncated": false
      }
    },
    ...
  ],
  "query_analysis": {
    "query_type": "summary",
    "document_type_hint": "legal"
  }
}
```

## Output Format

Your response should be formatted as markdown and include:

1. A brief introduction mentioning the source document(s)
2. The main content addressing the user's query
3. A conclusion or summary
4. Any relevant citations or references to specific sections of the document(s)

## Guidelines

1. For summary requests, provide a comprehensive overview of the document
2. For specific information requests, focus on directly answering the question
3. For legal documents, be precise with terminology and avoid interpretations
4. For technical documents, maintain technical accuracy and clarity
5. Always ground your response in the provided sources
6. Use appropriate formatting (headings, lists, etc.) for readability
7. If the content appears to be incomplete (e.g., just a table of contents or fragment):
   - Clearly state that the available content is limited
   - Explain what information you can extract from the limited content
   - Be honest about what you cannot determine from the available content
   - Suggest that a more complete document would be needed for a comprehensive answer

Remember to adapt your response based on the specific document type and user query. Your goal is to provide the most helpful and accurate information possible based on the search results.
"""

# Create the ResponseSynthesisAgent
ResponseSynthesisAgent = Agent(
    name="ResponseSynthesisAgent",
    instructions=response_synthesis_instructions,
    model="gpt-4o-mini"  # Using gpt-4o-mini consistently
)

async def synthesize_response(query: str, results: List[Dict[str, Any]], query_analysis: Dict[str, Any], context: Optional[RunContextWrapper] = None) -> str:
    """
    Synthesize a response based on search results and query analysis.

    Args:
        query: The user's query
        results: List of search results
        query_analysis: Analysis of the user's query
        context: Optional run context wrapper

    Returns:
        Synthesized response
    """
    try:
        # Check if we have a content warning
        has_content_warning = False
        content_warning = None
        for result in results:
            if result.get("metadata", {}).get("content_warning"):
                has_content_warning = True
                content_warning = result["metadata"]["content_warning"]
                break

        # Format the input for the ResponseSynthesisAgent
        agent_input = {
            "query": query,
            "results": results,
            "query_analysis": query_analysis,
            "has_content_warning": has_content_warning,
            "content_warning": content_warning
        }

        # Add special instructions for limited content
        if has_content_warning:
            agent_input["special_instructions"] = """
            IMPORTANT: The document content appears to be incomplete or limited.
            This may be just a fragment or table of contents rather than the full document.

            Please:
            1. Clearly state that the available content is limited
            2. Explain what information you can extract from the limited content
            3. Be honest about what you cannot determine from the available content
            4. Suggest that a more complete document would be needed for a comprehensive response
            """

        # Create a simple context dictionary
        context_dict = {}
        if context and hasattr(context, "get"):
            # Extract only what we need from the context
            client = context.get("client")
            if client:
                context_dict["has_client"] = True

        # Call the ResponseSynthesisAgent
        result = await Runner.run(ResponseSynthesisAgent, input=agent_input, context=context_dict)

        # Return the synthesized response
        if hasattr(result, 'final_output'):
            if isinstance(result.final_output, str):
                return result.final_output
            elif isinstance(result.final_output, dict) and "response" in result.final_output:
                return result.final_output["response"]
            else:
                return str(result.final_output)

        # Fallback if result format is unexpected
        logger.warning("Unexpected result format from ResponseSynthesisAgent")
        return f"Based on the search results for '{query}', I found relevant information in {len(results)} documents. However, I couldn't synthesize a proper response. Please try a more specific query."

    except Exception as e:
        logger.error(f"Error in response synthesis: {e}")
        # Return a fallback response
        return f"I found information related to '{query}' but encountered an error while synthesizing the response. The search returned {len(results)} relevant documents."
