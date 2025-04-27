"""
Test script for sequential search with early termination.

This script demonstrates the sequential search implementation.
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the sequential search implementation
from tools.kb_tools.sequential_search import sequential_search_with_early_termination, prioritize_search_strategies

async def test_sequential_search():
    """Test the sequential search implementation."""
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define test vector store ID (replace with a valid ID)
    vector_store_id = "vs_example"  # Replace with your vector store ID
    
    # Define test query
    query = "What is the termination notice period in my employment contract?"
    
    # Define test search strategies
    search_strategies = [
        {
            "name": "document_type",
            "description": "Document type filter: employment_contract",
            "filters": {"type": "eq", "key": "document_type", "value": "employment_contract"},
            "params": {
                "max_num_results": 5,
                "ranking_options": {"ranker": "hybrid"}
            }
        },
        {
            "name": "semantic_fallback",
            "description": "No filters (semantic fallback)",
            "filters": None,
            "params": {
                "max_num_results": 5,
                "ranking_options": {"ranker": "hybrid"}
            }
        }
    ]
    
    # Initialize strategy metrics
    strategy_metrics = {
        "document_type": {
            "calls": 10,
            "successes": 8,
            "failures": 2,
            "total_latency": 4.0,
            "result_counts": [3, 4, 5, 2, 3, 4, 5, 3],
            "success_rate": 0.8,
            "avg_latency": 0.5
        },
        "semantic_fallback": {
            "calls": 5,
            "successes": 5,
            "failures": 0,
            "total_latency": 3.0,
            "result_counts": [5, 5, 5, 5, 5],
            "success_rate": 1.0,
            "avg_latency": 0.6
        }
    }
    
    # Prioritize strategies based on metrics
    prioritized_strategies = prioritize_search_strategies(search_strategies, strategy_metrics)
    
    logger.info(f"Prioritized strategies: {[s['name'] for s in prioritized_strategies]}")
    
    # Execute sequential search
    try:
        results = await sequential_search_with_early_termination(
            client=client,
            vector_store_id=vector_store_id,
            query=query,
            search_strategies=prioritized_strategies,
            min_results_threshold=3,
            max_total_results=5,
            strategy_metrics=strategy_metrics
        )
        
        logger.info(f"Search returned {len(results)} results")
        
        # Print updated metrics
        logger.info(f"Updated metrics: {strategy_metrics}")
    except Exception as e:
        logger.error(f"Error executing sequential search: {e}")

if __name__ == "__main__":
    asyncio.run(test_sequential_search())
