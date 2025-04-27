"""
Registry Loader Module

This module provides utility functions for loading and initializing the agent registry.
"""

import os
import logging
from typing import Optional
from .agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

# Import ToolRegistry
try:
    from tools.registry import initialize_tool_registry
except ImportError:
    initialize_tool_registry = None
    logger.warning("Could not import initialize_tool_registry, will use agent registry without tool registry")

def initialize_agent_registry(config_path: Optional[str] = None, tool_config_path: Optional[str] = None) -> AgentRegistry:
    """
    Initialize the agent registry with the specified configuration.

    Args:
        config_path: Path to the agent configuration file. If None, uses the default path.
        tool_config_path: Path to the tool configuration file. If None, uses the default path.

    Returns:
        Initialized AgentRegistry instance
    """
    # Use default path if none provided
    if config_path is None:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Use only the fixed agent definitions file to avoid JSON parsing errors
        config_paths = [
            os.path.join(os.path.dirname(current_dir), 'config', 'agent_definitions_fixed.json')
        ]

        # Use the first existing config file
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                logger.info(f"Using agent definitions from: {path}\n")
                break

        if not config_path or not os.path.exists(config_path):
            logger.warning(f"No valid agent definitions file found, falling back to original")
            config_path = os.path.join(os.path.dirname(current_dir), 'config', 'agent_definitions.json')

    # Initialize tool registry if available
    tool_registry = None
    if initialize_tool_registry is not None:
        try:
            tool_registry = initialize_tool_registry(tool_config_path)
            logger.info(f"Tool registry initialized with {len(tool_registry.tools)} tools")
        except Exception as e:
            logger.error(f"Error initializing tool registry: {e}")
            # Continue without tool registry

    # Create and initialize the agent registry with the tool registry
    registry = AgentRegistry(tool_registry=tool_registry)

    try:
        # Load agent definitions
        registry.load_agent_definitions(config_path)

        # Instantiate agents
        registry.instantiate_agents()

        # Verify critical agents are registered
        # Check for ContentProcessorAgent
        content_processor = registry.get_agent("ContentProcessorAgent")
        if not content_processor:
            logger.warning("ContentProcessorAgent not found in registry, importing from module")
            try:
                # Try to import from content_processor module
                from agents.content_processor import ContentProcessorAgent
                content_processor = ContentProcessorAgent
                logger.info("Successfully imported ContentProcessorAgent from module")
            except ImportError:
                logger.warning("ContentProcessorAgent module not found, creating manually")
                # Create ContentProcessorAgent manually if not found
                from agents import Agent
                content_processor = Agent(
                    name="ContentProcessorAgent",
                    instructions="""You are a specialized content processing agent responsible for analyzing, summarizing, and extracting information from knowledge base documents. Your primary role is to process document content and provide meaningful, well-structured responses based on the user's query.""",
                    model="gpt-4o-mini"
                )
            registry.agents["ContentProcessorAgent"] = content_processor
            logger.info("Manually created and registered ContentProcessorAgent")

        # Check for WorkflowRouterAgent
        workflow_router = registry.get_agent("WorkflowRouterAgent")
        if not workflow_router:
            logger.warning("WorkflowRouterAgent not found in registry, creating manually")
            # Create WorkflowRouterAgent manually if not found
            from agents import Agent
            workflow_router = Agent(
                name="WorkflowRouterAgent",
                instructions="""You are a workflow orchestration agent responsible for determining the next step in processing a user query. Your job is to analyze the user's query, available templates, temporary files, conversation history, and the current state of the workflow to decide what action should be taken next.""",
                model="gpt-4o-mini"
            )
            registry.agents["WorkflowRouterAgent"] = workflow_router
            logger.info("Manually created and registered WorkflowRouterAgent")

        # Check for QueryAnalyzerAgent
        query_analyzer = registry.get_agent("QueryAnalyzerAgent")
        if not query_analyzer:
            logger.warning("QueryAnalyzerAgent not found in registry, creating manually")
            from agents import Agent
            query_analyzer = Agent(
                name="QueryAnalyzerAgent",
                instructions="""Analyze the user's query, available templates, and temporary files to determine the true intent with high accuracy.""",
                model="gpt-4o-mini"
            )
            registry.agents["QueryAnalyzerAgent"] = query_analyzer
            logger.info("Manually created and registered QueryAnalyzerAgent")

        # Check for DataGatheringAgentMinimal
        data_gathering = registry.get_agent("DataGatheringAgentMinimal")
        if not data_gathering:
            logger.warning("DataGatheringAgentMinimal not found in registry, creating manually")
            from agents import Agent
            data_gathering = Agent(
                name="DataGatheringAgentMinimal",
                instructions="""You are a specialized data gathering agent. Your job is to retrieve relevant information based on the user's query and intent.""",
                model="gpt-4o-mini"
            )
            registry.agents["DataGatheringAgentMinimal"] = data_gathering
            logger.info("Manually created and registered DataGatheringAgentMinimal")

        logger.info(f"Agent registry initialized with {len(registry.agents)} agents")
        return registry
    except Exception as e:
        logger.error(f"Error initializing agent registry: {e}")
        raise
