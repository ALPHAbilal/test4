"""
Agent Registry Module

This module provides functionality for registering and retrieving agents.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for agents."""

    def __init__(self):
        """Initialize the agent registry."""
        self.agents = {}

    def register_agent(self, agent):
        """Register an agent with the registry."""
        if hasattr(agent, 'name') and agent.name:
            self.agents[agent.name] = agent
            logger.info(f"Registered agent: {agent.name}")
        else:
            logger.warning("Attempted to register agent without a name")

    def get_agent(self, name):
        """Get an agent by name."""
        return self.agents.get(name)

    def get_all_agents(self):
        """Get all registered agents."""
        return list(self.agents.values())

def initialize_agent_registry():
    """Initialize the agent registry with agents from the configuration."""
    registry = AgentRegistry()

    try:
        # Prioritize configurations with ContentProcessorAgent
        config_paths = [
            os.path.join('agents', 'config', 'agent_definitions_updated_final.json'),
            os.path.join('agents', 'config', 'agent_definitions_updated_new.json'),
            os.path.join('agents', 'config', 'agent_definitions_updated.json'),
            os.path.join('agents', 'config', 'agent_definitions_fixed.json'),
            os.path.join('agents', 'config', 'agent_definitions.json')
        ]
        
        # Find the first existing config file
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
                
        if config_path:
            logger.info(f"Loading agent definitions from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Create and register agents from the configuration
            if 'agents' in config:
                for agent_def in config['agents']:
                    # Create a simple agent object with the properties from the definition
                    from agents import Agent
                    agent = Agent(
                        name=agent_def.get('name'),
                        instructions=agent_def.get('instructions'),
                        model=agent_def.get('model'),
                        tools=agent_def.get('tools', [])
                    )
                    registry.register_agent(agent)
                    
            # Verify ContentProcessorAgent is registered
            content_processor = registry.get_agent("ContentProcessorAgent")
            if not content_processor:
                logger.warning("ContentProcessorAgent not found in registry, creating manually")
                # Create ContentProcessorAgent manually if not found
                from agents import Agent
                content_processor = Agent(
                    name="ContentProcessorAgent",
                    instructions="""You are a specialized content processing agent responsible for analyzing, summarizing, and extracting information from knowledge base documents. Your primary role is to process document content and provide meaningful, well-structured responses based on the user's query.""",
                    model="gpt-4o-mini"
                )
                registry.register_agent(content_processor)
        else:
            logger.warning(f"No agent configuration files found. Tried paths: {config_paths}")
    except Exception as e:
        logger.error(f"Error initializing agent registry: {e}")

    return registry
