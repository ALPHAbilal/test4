"""
Registry Loader Module

This module provides utility functions for loading and initializing the agent registry.
"""

import os
import logging
from typing import Optional
from .agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

def initialize_agent_registry(config_path: Optional[str] = None) -> AgentRegistry:
    """
    Initialize the agent registry with the specified configuration.
    
    Args:
        config_path: Path to the agent configuration file. If None, uses the default path.
        
    Returns:
        Initialized AgentRegistry instance
    """
    # Use default path if none provided
    if config_path is None:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to default config file
        config_path = os.path.join(os.path.dirname(current_dir), 'config', 'agent_definitions.json')
    
    # Create and initialize the registry
    registry = AgentRegistry()
    
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
