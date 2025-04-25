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
        # Try to load from the final updated definitions first
        config_path = os.path.join('agents', 'config', 'agent_definitions_updated_final.json')

        # If final updated definitions don't exist, try newest updated definitions
        if not os.path.exists(config_path):
            config_path = os.path.join('agents', 'config', 'agent_definitions_updated_new.json')

        # If newest updated definitions don't exist, try updated definitions
        if not os.path.exists(config_path):
            config_path = os.path.join('agents', 'config', 'agent_definitions_updated.json')

        # If updated definitions don't exist, try fixed definitions
        if not os.path.exists(config_path):
            config_path = os.path.join('agents', 'config', 'agent_definitions_fixed.json')

        # If fixed definitions don't exist, fall back to original definitions
        if not os.path.exists(config_path):
            config_path = os.path.join('agents', 'config', 'agent_definitions.json')

        if os.path.exists(config_path):
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
        else:
            logger.warning(f"No agent configuration files found. Tried: agent_definitions_updated.json, agent_definitions_fixed.json, agent_definitions.json")
    except Exception as e:
        logger.error(f"Error initializing agent registry: {e}")

    return registry
