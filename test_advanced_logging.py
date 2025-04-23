"""
Test script for advanced logging functionality.
"""

import time
import logging
import asyncio
import random
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the application
from enhanced_with_agents_fixed import app

# Create test client
client = TestClient(app)

async def test_query_endpoint():
    """Test the query endpoint with advanced logging"""
    logger.info("Testing query endpoint...")
    
    # Make a query request
    response = client.post(
        "/query",
        data={"question": "What is the purpose of this application?", "new_conversation": "true"}
    )
    
    # Check response
    assert response.status_code == 200
    logger.info(f"Query response: {response.json()}")
    
    # Make another query in the same conversation
    response = client.post(
        "/query",
        data={"question": "Can you provide more details?", "new_conversation": "false"}
    )
    
    # Check response
    assert response.status_code == 200
    logger.info(f"Follow-up query response: {response.json()}")
    
    return "Query endpoint tests completed successfully"

async def test_stream_endpoint():
    """Test the stream endpoint with advanced logging"""
    logger.info("Testing stream endpoint...")
    
    # Make a stream request
    response = client.post(
        "/stream",
        data={"question": "How does the streaming functionality work?", "new_conversation": "true"}
    )
    
    # Check response
    assert response.status_code == 200
    logger.info("Stream response received successfully")
    
    return "Stream endpoint tests completed successfully"

async def test_conversation_history():
    """Test the conversation history endpoint with advanced logging"""
    logger.info("Testing conversation history endpoint...")
    
    # Make a query to ensure there's history
    client.post(
        "/query",
        data={"question": "This is a test question for history", "new_conversation": "true"}
    )
    
    # Get conversation history
    response = client.get("/conversation-history")
    
    # Check response
    assert response.status_code == 200
    history = response.json()
    logger.info(f"Conversation history: {history}")
    
    return "Conversation history tests completed successfully"

async def test_clear_conversation():
    """Test the clear conversation endpoint with advanced logging"""
    logger.info("Testing clear conversation endpoint...")
    
    # Make a query to ensure there's a conversation
    client.post(
        "/query",
        data={"question": "This is a test question before clearing", "new_conversation": "true"}
    )
    
    # Clear conversation
    response = client.post("/clear-conversation")
    
    # Check response
    assert response.status_code == 200
    logger.info(f"Clear conversation response: {response.json()}")
    
    # Verify history is cleared
    history_response = client.get("/conversation-history")
    history = history_response.json()
    assert len(history.get("history", [])) == 0
    
    return "Clear conversation tests completed successfully"

async def test_error_handling():
    """Test error handling with advanced logging"""
    logger.info("Testing error handling...")
    
    # Simulate an error by making an invalid request
    try:
        response = client.post(
            "/query",
            data={"invalid_param": "This should cause an error"}
        )
        logger.info(f"Response: {response.status_code} - {response.text}")
    except Exception as e:
        logger.info(f"Expected error occurred: {e}")
    
    return "Error handling tests completed"

async def run_tests():
    """Run all tests"""
    logger.info("Starting advanced logging tests...")
    
    # Run tests with some delay between them
    test_results = []
    
    try:
        # Test query endpoint
        result = await test_query_endpoint()
        test_results.append(result)
        await asyncio.sleep(2)
        
        # Test stream endpoint
        result = await test_stream_endpoint()
        test_results.append(result)
        await asyncio.sleep(2)
        
        # Test conversation history
        result = await test_conversation_history()
        test_results.append(result)
        await asyncio.sleep(2)
        
        # Test clear conversation
        result = await test_clear_conversation()
        test_results.append(result)
        await asyncio.sleep(2)
        
        # Test error handling
        result = await test_error_handling()
        test_results.append(result)
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        test_results.append(f"Test failed with error: {e}")
    
    logger.info("All tests completed")
    logger.info("Test results:")
    for result in test_results:
        logger.info(f"- {result}")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())
