import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)

def test_agent_registry():
    logger.info("Testing agent registry...")
    
    try:
        # Import the agent registry
        from agents.registry.registry_loader import initialize_agent_registry
        logger.info("Successfully imported initialize_agent_registry")
        
        # Initialize the agent registry
        agent_registry = initialize_agent_registry()
        logger.info(f"Successfully initialized agent registry with {len(agent_registry.agents)} agents")
        
        # Check if WorkflowRouterAgent is registered
        workflow_router = agent_registry.get_agent("WorkflowRouterAgent")
        if workflow_router:
            logger.info("WorkflowRouterAgent is registered")
        else:
            logger.error("WorkflowRouterAgent is not registered")
        
        # List all registered agents
        logger.info("Registered agents:")
        for agent_name in agent_registry.agents:
            logger.info(f"- {agent_name}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_agent_registry()
