"""
Sequential Search with Early Termination

This module provides an optimized search strategy that executes search variants
sequentially and terminates early when sufficient results are found.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

async def sequential_search_with_early_termination(
    client,
    vector_store_id: str,
    query: str,
    search_strategies: List[Dict[str, Any]],
    min_results_threshold: int = 3,
    max_total_results: int = 10,
    strategy_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    context: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Execute search strategies sequentially with early termination.

    Args:
        client: The OpenAI client
        vector_store_id: The vector store ID to search
        query: The search query
        search_strategies: List of search strategies, each containing:
            - name: Strategy name
            - description: Strategy description
            - filters: Filter object for the search
            - params: Additional search parameters (optional)
        min_results_threshold: Minimum number of results to consider search successful
        max_total_results: Maximum total results to return
        strategy_metrics: Optional dictionary to track strategy performance
        context: Optional context object that may contain query analysis

    Returns:
        List of search results with metadata about which strategy found them
    """
    all_results = []

    # If strategy_metrics is None, initialize an empty dict
    if strategy_metrics is None:
        strategy_metrics = {}

    # Execute strategies sequentially
    for strategy in search_strategies:
        strategy_name = strategy.get("name", "unnamed")
        strategy_description = strategy.get("description", strategy_name)

        # Skip remaining strategies if we already have sufficient results
        if len(all_results) >= min_results_threshold:
            logger.info(f"Early termination before strategy '{strategy_name}' with {len(all_results)} results")
            break

        # Prepare search parameters
        search_params = {
            "vector_store_id": vector_store_id,
            "query": query,
            "filters": strategy.get("filters"),
            "max_num_results": strategy.get("params", {}).get("max_num_results", 5)
        }

        # Add optional parameters if provided in strategy
        if strategy.get("params", {}).get("rewrite_query") is not None:
            search_params["rewrite_query"] = strategy.get("params", {}).get("rewrite_query")
        else:
            # Check if we have query analysis in the context
            if context and hasattr(context, "get"):
                query_analysis = context.get("query_analysis")
                if query_analysis and "rewrite_query" in query_analysis:
                    # Use the agent's recommendation
                    search_params["rewrite_query"] = query_analysis["rewrite_query"]
                else:
                    search_params["rewrite_query"] = True  # Default to True
            else:
                search_params["rewrite_query"] = True  # Default to True

        if strategy.get("params", {}).get("ranking_options"):
            search_params["ranking_options"] = strategy.get("params", {}).get("ranking_options")
        else:
            # Check if we have query analysis in the context
            if context and hasattr(context, "get"):
                query_analysis = context.get("query_analysis")
                if query_analysis and "search_priority" in query_analysis:
                    # Use the agent's recommendation for ranking
                    priority = query_analysis["search_priority"]
                    if priority == "precision":
                        search_params["ranking_options"] = {"ranker": "best_match"}
                    elif priority == "recall":
                        search_params["ranking_options"] = {"ranker": "hybrid"}
                    else:  # balanced
                        search_params["ranking_options"] = {"ranker": "best_match"}
                else:
                    search_params["ranking_options"] = {"ranker": "best_match"}  # Default ranker
            else:
                search_params["ranking_options"] = {"ranker": "best_match"}  # Default ranker

        # Add any additional parameters from the strategy
        if "params" in strategy:
            for key, value in strategy["params"].items():
                if key != "max_num_results":  # Already handled above
                    search_params[key] = value

        # Log the strategy being executed
        logger.info(f"Executing search strategy: {strategy_description}")

        # Record start time for performance tracking
        start_time = time.time()
        success = False
        result_count = 0

        try:
            # Execute the search
            search_result = await asyncio.to_thread(client.vector_stores.search, **search_params)

            # Process results if any were found
            if search_result and hasattr(search_result, "data") and search_result.data:
                result_count = len(search_result.data)
                logger.info(f"Strategy '{strategy_name}' returned {result_count} results")

                # Process each result
                for result in search_result.data:
                    # Extract content
                    content = ""
                    if hasattr(result, "content"):
                        # Handle different content formats
                        if isinstance(result.content, str):
                            content = result.content
                        elif hasattr(result.content, "__iter__"):
                            # Combine content parts
                            content_parts = []
                            for part in result.content:
                                if hasattr(part, "text"):
                                    content_parts.append(part.text)
                                elif isinstance(part, str):
                                    content_parts.append(part)
                            content = "\n\n".join(content_parts)

                    # Create result object
                    processed_result = {
                        "file_id": getattr(result, "file_id", None),
                        "score": getattr(result, "score", 0),
                        "content": content,
                        "strategy": strategy_name,
                        "strategy_description": strategy_description
                    }

                    # Add metadata if available
                    if hasattr(result, "attributes") and result.attributes:
                        processed_result["metadata"] = result.attributes

                    all_results.append(processed_result)

                # Mark as successful if we got any results
                success = result_count > 0
            else:
                logger.info(f"Strategy '{strategy_name}' returned no results")
        except Exception as e:
            logger.error(f"Error executing strategy '{strategy_name}': {e}")

        # Record performance metrics
        execution_time = time.time() - start_time

        # Update strategy metrics if provided
        if strategy_name in strategy_metrics:
            metrics = strategy_metrics[strategy_name]
            metrics["calls"] = metrics.get("calls", 0) + 1

            if success:
                metrics["successes"] = metrics.get("successes", 0) + 1
                metrics["total_latency"] = metrics.get("total_latency", 0) + execution_time
                metrics["result_counts"] = metrics.get("result_counts", []) + [result_count]

                # Calculate success rate
                metrics["success_rate"] = metrics["successes"] / metrics["calls"]

                # Calculate average latency
                if metrics["successes"] > 0:
                    metrics["avg_latency"] = metrics["total_latency"] / metrics["successes"]
            else:
                metrics["failures"] = metrics.get("failures", 0) + 1
        else:
            # Initialize metrics for this strategy
            strategy_metrics[strategy_name] = {
                "calls": 1,
                "successes": 1 if success else 0,
                "failures": 0 if success else 1,
                "total_latency": execution_time if success else 0,
                "result_counts": [result_count] if success else [],
                "success_rate": 1.0 if success else 0.0,
                "avg_latency": execution_time if success else 0
            }

    # Deduplicate results based on content
    unique_results = []
    seen_content_hashes = set()

    for result in all_results:
        # Create a hash of the content to identify duplicates
        content_hash = hash(result.get("content", ""))

        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            unique_results.append(result)

    # Sort by score
    unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Limit to max_total_results
    return unique_results[:max_total_results]

def prioritize_search_strategies(
    strategies: List[Dict[str, Any]],
    strategy_metrics: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prioritize search strategies based on historical performance.

    Args:
        strategies: List of search strategies
        strategy_metrics: Dictionary of strategy performance metrics

    Returns:
        Prioritized list of search strategies
    """
    # Calculate expected value for each strategy
    for strategy in strategies:
        strategy_name = strategy.get("name", "unnamed")

        if strategy_name in strategy_metrics:
            metrics = strategy_metrics[strategy_name]

            # Calculate expected value based on success rate and latency
            success_rate = metrics.get("success_rate", 0.5)  # Default to 0.5 if not available
            avg_latency = metrics.get("avg_latency", 1.0)    # Default to 1.0 if not available

            # Avoid division by zero
            if avg_latency <= 0:
                avg_latency = 0.1

            # Calculate expected value (success_rate / avg_latency)
            # Higher success rate and lower latency = higher expected value
            expected_value = success_rate / avg_latency

            strategy["expected_value"] = expected_value
        else:
            # For strategies without metrics, assign a default expected value
            strategy["expected_value"] = 0.5  # Middle priority

    # Sort strategies by expected value (descending)
    sorted_strategies = sorted(strategies, key=lambda x: x.get("expected_value", 0), reverse=True)

    return sorted_strategies
