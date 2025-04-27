"""
Response Synthesis Module

This module provides functions for synthesizing responses based on search results.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results according to OpenAI guide recommendations.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted results string
    """
    formatted_results = ""
    
    for result in results:
        # Extract file information
        file_id = result.get("file_id", "unknown")
        file_name = result.get("filename", f"FileID:{file_id[-6:]}")
        score = result.get("score", 0)
        
        # Start result block
        formatted_results += f"<result file_id='{file_id}' file_name='{file_name}' score='{score:.2f}'>\n"
        
        # Add metadata if available
        if "metadata" in result and result["metadata"]:
            formatted_results += "<metadata>\n"
            for key, value in result["metadata"].items():
                formatted_results += f"  <{key}>{value}</{key}>\n"
            formatted_results += "</metadata>\n"
        
        # Add content
        formatted_results += "<content>\n"
        content = result.get("content", "")
        if isinstance(content, str):
            formatted_results += content
        elif isinstance(content, list):
            for part in content:
                if hasattr(part, "text"):
                    formatted_results += part.text + "\n"
                elif isinstance(part, dict) and "text" in part:
                    formatted_results += part["text"] + "\n"
                elif isinstance(part, str):
                    formatted_results += part + "\n"
        formatted_results += "\n</content>\n"
        
        # End result block
        formatted_results += "</result>\n\n"
    
    # Wrap in sources tag
    formatted_results = f"<sources>\n{formatted_results}\n</sources>"
    
    return formatted_results

def get_synthesis_prompt(query: str, formatted_results: str, document_type: str = None) -> Dict[str, List[Dict[str, str]]]:
    """
    Create a prompt for response synthesis based on search results.
    
    Args:
        query: User query
        formatted_results: Formatted search results
        document_type: Optional document type for specialized handling
        
    Returns:
        Prompt messages for chat completion
    """
    # Base system message
    system_message = """You are a document assistant that provides accurate, comprehensive answers based on the provided sources.
    Always ground your response in the provided sources and maintain a professional tone.
    If the sources don't contain relevant information, acknowledge this limitation."""
    
    # Customize system message based on document type
    if document_type:
        if document_type.lower() in ["legal", "contract", "agreement", "law", "code"]:
            system_message += """
            For legal documents:
            - Be precise with legal terminology
            - Avoid making legal interpretations beyond what's in the sources
            - Clearly indicate when information might be incomplete
            - Structure your response with clear sections
            """
        elif document_type.lower() in ["technical", "manual", "guide", "documentation"]:
            system_message += """
            For technical documents:
            - Use precise technical terminology from the sources
            - Provide step-by-step explanations when appropriate
            - Include relevant technical details
            - Structure your response logically
            """
    
    # Check if this is a summary request
    is_summary_request = any(term in query.lower() for term in ["summary", "summarize", "summarise", "overview"])
    
    if is_summary_request:
        system_message += """
        For document summaries:
        - Provide a comprehensive overview of the document
        - Include the main sections and their key points
        - Highlight the most important information
        - Structure your summary with clear headings
        - Be thorough but concise
        """
    
    # Create messages
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"Sources:\n{formatted_results}\n\nQuery: '{query}'\n\nProvide a well-structured response that directly answers the query based on the provided sources."
        }
    ]
    
    return messages
