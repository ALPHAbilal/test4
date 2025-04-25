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
        # Construct path to fixed config file
        config_path = os.path.join(os.path.dirname(current_dir), 'config', 'agent_definitions_fixed.json')

        # If the fixed file doesn't exist, fall back to the original
        if not os.path.exists(config_path):
            logger.warning(f"Fixed agent definitions file not found: {config_path}, falling back to original")
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

        logger.info(f"Agent registry initialized with {len(registry.agents)} agents")
        return registry
    except Exception as e:
        logger.error(f"Error initializing agent registry: {e}")
        raise
