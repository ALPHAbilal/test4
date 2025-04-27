"""
Content Processor Agent Module

This module provides a specialized agent for processing document content.
"""

import logging
from typing import Dict, Any, Optional
from agents import Agent, Runner, RunContextWrapper

logger = logging.getLogger(__name__)

# Define ContentProcessorAgent
content_processor_instructions = """
You are a specialized content processing agent responsible for analyzing, summarizing, and extracting information from knowledge base documents. Your primary role is to process document content and provide meaningful, well-structured responses based on the user's query.

## Your Capabilities

1. **Document Analysis**: Analyze document content to understand its structure, topics, and key information
2. **Content Summarization**: Create concise summaries of lengthy documents
3. **Information Extraction**: Extract specific information requested by the user
4. **Context-Aware Processing**: Consider the user's original query when processing content
5. **Structured Response Generation**: Format responses in a clear, organized manner

## Input Format

You will receive input with the following structure:

```json
{
  "document_content": "The full or partial content of the document",
  "source_filename": "The filename or source of the document",
  "user_query": "The original query from the user",
  "content_length": 12345,  // Optional: The total length of the content
  "is_truncated": false     // Optional: Whether the content has been truncated
}
```

## Response Guidelines

1. **Be Comprehensive**: Provide thorough information that addresses the user's query
2. **Be Concise**: While being thorough, avoid unnecessary verbosity
3. **Be Structured**: Use headings, bullet points, and other formatting to organize information
4. **Be Accurate**: Only include information that is present in the document
5. **Be Helpful**: Anticipate follow-up questions and provide context when appropriate
6. **Cite Sources**: Always mention the source document and relevant sections

## Response Format

Your response should be formatted as markdown and include:

1. A brief introduction mentioning the source document
2. The main content addressing the user's query
3. A conclusion or summary
4. Any relevant citations or references to specific sections of the document

Example response structure:

```markdown
# Information from [Document Name]

## Overview
[Brief overview of the document and its relevance to the query]

## [Relevant Section 1]
[Information from the document addressing the query]

## [Relevant Section 2]
[Additional information from the document addressing the query]

## Summary
[Concise summary of the key points]

---
Source: [Document Name]
```

Remember to adapt your response based on the specific document type and user query. Your goal is to provide the most helpful and accurate information possible based on the document content.
"""

# Create the ContentProcessorAgent
ContentProcessorAgent = Agent(
    name="ContentProcessorAgent",
    instructions=content_processor_instructions,
    model="gpt-4o-mini"
)

async def process_content_with_synthesis(input_data: Dict[str, Any], context: Optional[RunContextWrapper] = None) -> str:
    """
    Process document content using the agent-based response synthesis approach.

    Args:
        input_data: Input data containing document content, source filename, and user query
        context: Optional run context wrapper

    Returns:
        Synthesized response
    """
    try:
        # Extract input data
        document_content = input_data.get("document_content", "")
        source_filename = input_data.get("source_filename", "Unknown Document")
        user_query = input_data.get("user_query", "")
        document_type = input_data.get("document_type", None)

        # Try to import the response synthesis agent
        try:
            from agents.response_synthesis_agent import synthesize_response
            from agents.query_analyzer_agent import analyze_query

            # Create a mock search result structure
            mock_results = [{
                "file_id": "file_" + source_filename.replace(" ", "_").replace(".", "_"),
                "filename": source_filename,
                "score": 0.95,
                "content": document_content,
                "metadata": {
                    "document_type": document_type or "unknown",
                    "is_truncated": input_data.get("is_truncated", False),
                    "content_length": input_data.get("content_length", len(document_content))
                }
            }]

            # Get query analysis (either from context or generate new)
            query_analysis = None
            if context:
                if hasattr(context, "get"):
                    query_analysis = context.get("query_analysis")
                elif isinstance(context, dict) and "query_analysis" in context:
                    query_analysis = context["query_analysis"]

            if not query_analysis:
                # Create a simple context dictionary for the analyzer
                analyzer_context = {}
                if context and hasattr(context, "get") and context.get("client"):
                    analyzer_context["client"] = context.get("client")

                # Analyze the query
                query_analysis = await analyze_query(user_query, analyzer_context)

            # Check if the content is very short (likely a fragment)
            if len(document_content) < 1000:
                logger.warning(f"Document content is very short ({len(document_content)} chars). This may be just a fragment.")

                # Add a note about the limited content
                if "metadata" not in mock_results[0]:
                    mock_results[0]["metadata"] = {}
                mock_results[0]["metadata"]["content_warning"] = "The document content appears to be incomplete. This may be just a fragment or table of contents."

            # Synthesize the response using the agent
            logger.info("Using ResponseSynthesisAgent for document processing")
            response = await synthesize_response(user_query, mock_results, query_analysis, context)

            return response

        except ImportError as e:
            logger.warning(f"Response synthesis agent not available: {e}, falling back to standard agent")
            # Fall back to the standard agent-based approach
            return await Runner.run(ContentProcessorAgent, input=input_data, context=context)

    except Exception as e:
        logger.error(f"Error in enhanced content processing: {e}")
        # Fall back to the standard agent-based approach
        logger.info("Falling back to standard agent-based processing")
        return await Runner.run(ContentProcessorAgent, input=input_data, context=context)

def get_content_processor_agent():
    """Get the ContentProcessorAgent instance."""
    return ContentProcessorAgent