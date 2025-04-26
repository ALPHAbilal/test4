import os
import logging
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
)
logger = logging.getLogger(__name__)

# Test function
async def main():
    logger.info("Starting test...")
    try:
        # Import the app module
        import app
        logger.info("Successfully imported app module")
        
        # Test if the retrieve_template_content function is defined
        if hasattr(app, 'retrieve_template_content'):
            logger.info("retrieve_template_content function is defined")
        else:
            logger.error("retrieve_template_content function is not defined")
        
        # Test if the agent registry is initialized
        if hasattr(app, 'agent_registry') and app.agent_registry is not None:
            logger.info(f"Agent registry is initialized with {len(app.agent_registry.agents) if hasattr(app.agent_registry, 'agents') else 0} agents")
        else:
            logger.error("Agent registry is not initialized")
        
        # Test if the workflow registry is initialized
        if hasattr(app, 'workflow_registry') and app.workflow_registry is not None:
            logger.info(f"Workflow registry is initialized with {len(app.workflow_registry.workflows) if hasattr(app.workflow_registry, 'workflows') else 0} workflows")
        else:
            logger.error("Workflow registry is not initialized")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    
    logger.info("Test completed")

# Run the test
if __name__ == "__main__":
    asyncio.run(main())
