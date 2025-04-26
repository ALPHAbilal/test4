import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    logger.info("Testing imports...")
    
    # Test importing the agent registry
    try:
        from agents.registry.registry_loader import initialize_agent_registry
        logger.info("Successfully imported initialize_agent_registry")
    except Exception as e:
        logger.error(f"Error importing initialize_agent_registry: {e}")
    
    # Test importing the workflow registry
    try:
        from core.workflow_registry import initialize_workflow_registry
        logger.info("Successfully imported initialize_workflow_registry")
    except Exception as e:
        logger.error(f"Error importing initialize_workflow_registry: {e}")
    
    # Test importing the retrieve_template_content function
    try:
        from app import retrieve_template_content
        logger.info("Successfully imported retrieve_template_content")
    except Exception as e:
        logger.error(f"Error importing retrieve_template_content: {e}")
    
    logger.info("Import tests completed")

if __name__ == "__main__":
    test_imports()
