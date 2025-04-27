"""
Query Analyzer Agent Module

This module provides an agent for analyzing user queries to determine search parameters.
"""

import logging
from typing import Dict, Any, Optional
from agents import Agent, Runner, RunContextWrapper

logger = logging.getLogger(__name__)

# Define QueryAnalyzerAgent
query_analyzer_instructions = """
You are a specialized query analysis agent responsible for analyzing user queries to determine optimal search parameters.
Your role is to examine the query and determine the best search strategy, parameters, and approach.

## Your Capabilities

1. **Query Intent Analysis**: Determine the user's intent without making assumptions
2. **Search Parameter Optimization**: Recommend optimal search parameters based on query characteristics
3. **Document Type Detection**: Identify likely document types relevant to the query if mentioned
4. **Query Reformulation**: Suggest improved query formulations for better search results

## Input Format

You will receive a user query as input.

## Output Format

You must return a JSON object with the following structure:

```json
{
  "query_type": "information_request|factual_question|exploratory|navigational",
  "recommended_min_results": 5,
  "recommended_max_results": 10,
  "document_type_hint": "legal|technical|financial|general|unknown",
  "reformulated_query": "improved query for search",
  "search_priority": "precision|recall|balanced",
  "general_document_types": ["kb", "knowledge base", "general", "unknown", "none", ""],
  "rewrite_query": true|false
}
```

## Guidelines

1. Do NOT make assumptions about what the user wants - analyze the query as presented
2. Adjust result counts based on query complexity, not presumed intent
3. For complex queries, recommend higher result counts to ensure sufficient information
4. For simple, direct queries, recommend lower result counts for precision
5. Only identify document types when clearly indicated in the query
6. Reformulate queries to improve search effectiveness while preserving the original intent
7. Set search priority based on query characteristics, not presumed user goals

Be thoughtful in your analysis and provide parameters that will optimize the search experience without making assumptions about user intent.
"""

# Create the QueryAnalyzerAgent
QueryAnalyzerAgent = Agent(
    name="QueryAnalyzerAgent",
    instructions=query_analyzer_instructions,
    model="gpt-4o-mini"
)

async def analyze_query(query: str, context: Optional[RunContextWrapper] = None) -> Dict[str, Any]:
    """
    Analyze a user query to determine optimal search parameters.

    Args:
        query: The user query to analyze
        context: Optional run context wrapper

    Returns:
        Dictionary with analysis results
    """
    try:
        # Create a context dictionary if needed
        context_dict = {}
        if context and hasattr(context, "get"):
            # Extract only what we need from the context
            client = context.get("client")
            if client:
                context_dict["has_client"] = True

        # Call the QueryAnalyzerAgent
        result = await Runner.run(QueryAnalyzerAgent, input=query, context=context_dict)

        # Parse the result
        if hasattr(result, 'final_output'):
            if isinstance(result.final_output, dict):
                return result.final_output
            elif isinstance(result.final_output, str):
                import json
                import re

                # Clean up the string to handle markdown formatting
                json_str = result.final_output

                # Remove markdown code blocks if present
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'```\s*$', '', json_str)

                # Remove any leading/trailing whitespace
                json_str = json_str.strip()

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse QueryAnalyzerAgent output as JSON: {result.final_output}")
                    logger.error(f"JSON decode error: {e}")

        # Use a simplified agent call to get basic analysis if parsing fails
        logger.warning("Using simplified query analysis due to parsing failure")
        try:
            # Create a simplified agent for basic analysis
            from agents import Agent

            basic_analyzer = Agent(
                name="BasicQueryAnalyzer",
                instructions="Analyze this query to determine what document type it might be referring to and how complex the query is. Return JSON with document_type_hint (string) and query_complexity (string: 'simple', 'moderate', or 'complex').",
                model="gpt-4o-mini"
            )

            # Call the simplified agent
            basic_result = await Runner.run(basic_analyzer, input=query)

            # Parse the result
            if hasattr(basic_result, 'final_output'):
                if isinstance(basic_result.final_output, dict):
                    basic_analysis = basic_result.final_output
                elif isinstance(basic_result.final_output, str):
                    import json
                    try:
                        basic_analysis = json.loads(basic_result.final_output)
                    except json.JSONDecodeError:
                        basic_analysis = {}
                else:
                    basic_analysis = {}
            else:
                basic_analysis = {}

            # Create analysis with dynamic values from basic analysis
            query_complexity = basic_analysis.get("query_complexity", "moderate")

            # Set parameters based on query complexity
            if query_complexity == "complex":
                min_results = 8
                max_results = 15
                search_priority = "recall"
            elif query_complexity == "simple":
                min_results = 3
                max_results = 7
                search_priority = "precision"
            else:  # moderate
                min_results = 5
                max_results = 10
                search_priority = "balanced"

            return {
                "query_type": "information_request",
                "recommended_min_results": min_results,
                "recommended_max_results": max_results,
                "document_type_hint": basic_analysis.get("document_type_hint", "unknown"),
                "reformulated_query": query,
                "search_priority": search_priority,
                "rewrite_query": True
            }
        except Exception as basic_error:
            logger.error(f"Error in simplified query analysis: {basic_error}")
            # Absolute fallback with minimal assumptions
            return {
                "query_type": "information_request",
                "recommended_min_results": 5,
                "recommended_max_results": 10,
                "document_type_hint": "unknown",
                "reformulated_query": query,
                "search_priority": "balanced",
                "rewrite_query": True,
                "general_document_types": ["kb", "knowledge base", "general", "unknown", "none", ""]
            }
    except Exception as e:
        logger.error(f"Error in query analysis: {e}")
        # Absolute last resort fallback with no assumptions
        return {
            "query_type": "information_request",
            "recommended_min_results": 5,
            "recommended_max_results": 10,
            "document_type_hint": "unknown",
            "reformulated_query": query,
            "search_priority": "balanced",
            "rewrite_query": True,
            "general_document_types": ["kb", "knowledge base", "general", "unknown", "none", ""]
        }
