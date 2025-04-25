"""
Agent Registry Module

This module provides a registry for dynamically loading and managing agents.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import importlib

logger = logging.getLogger(__name__)

# Import Agent class from OpenAI Agents SDK
from agents import Agent, function_tool

# Import ToolRegistry
try:
    from tools.registry import ToolRegistry
except ImportError:
    ToolRegistry = None
    logger.warning("Could not import ToolRegistry, will use fallback tool resolution")

class AgentRegistry:
    """
    A registry for dynamically loading and managing agents.

    This class provides functionality to:
    1. Load agent definitions from configuration files
    2. Instantiate agents based on these definitions
    3. Store and retrieve agent instances
    """

    def __init__(self, tool_registry=None):
        """
        Initialize the agent registry.

        Args:
            tool_registry: Optional ToolRegistry instance to use for resolving tools
        """
        self.agents = {}  # Dictionary mapping agent names to agent instances
        self.agent_definitions = {}  # Dictionary mapping agent names to their definitions
        self.tools_cache = {}  # Cache for tool functions
        self.tool_registry = tool_registry  # ToolRegistry instance

    def load_agent_definitions(self, config_path: str) -> None:
        """
        Load agent definitions from a configuration file.

        Args:
            config_path: Path to the configuration file (JSON)
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Store agent definitions
            for agent_def in config.get('agents', []):
                agent_name = agent_def.get('name')
                if agent_name:
                    self.agent_definitions[agent_name] = agent_def
                    logger.info(f"Loaded definition for agent: {agent_name}")
                else:
                    logger.warning(f"Skipping agent definition without name: {agent_def}")

            logger.info(f"Loaded {len(self.agent_definitions)} agent definitions from {config_path}")
        except Exception as e:
            logger.error(f"Error loading agent definitions from {config_path}: {e}")
            raise

    def _resolve_tools(self, tool_names: List[str]) -> List[Callable]:
        """
        Resolve tool names to actual tool functions.

        Args:
            tool_names: List of tool names to resolve

        Returns:
            List of resolved tool functions
        """
        resolved_tools = []

        for tool_name in tool_names:
            # Check if tool is already in cache
            if tool_name in self.tools_cache:
                resolved_tools.append(self.tools_cache[tool_name])
                continue

            # Try to get the tool from the tool registry if available
            if self.tool_registry is not None:
                tool = self.tool_registry.get_tool(tool_name)
                if tool is not None:
                    self.tools_cache[tool_name] = tool
                    resolved_tools.append(tool)
                    logger.info(f"Resolved tool {tool_name} from tool registry")
                    continue

            # Fallback: Try to import the tool from the app module
            try:
                # First, try to import directly from app
                import app
                if hasattr(app, tool_name):
                    tool = getattr(app, tool_name)
                    self.tools_cache[tool_name] = tool
                    resolved_tools.append(tool)
                    logger.info(f"Resolved tool {tool_name} from app module")
                    continue

                # If not found, try to import from tools module
                try:
                    tools_module = importlib.import_module('tools')
                    if hasattr(tools_module, tool_name):
                        tool = getattr(tools_module, tool_name)
                        self.tools_cache[tool_name] = tool
                        resolved_tools.append(tool)
                        logger.info(f"Resolved tool {tool_name} from tools module")
                        continue
                except (ImportError, AttributeError):
                    pass

                # If still not found, log warning
                logger.warning(f"Could not resolve tool: {tool_name}")
            except Exception as e:
                logger.error(f"Error resolving tool {tool_name}: {e}")

        return resolved_tools

    def _resolve_handoffs(self, handoff_names: List[str]) -> List[str]:
        """
        Resolve handoff names to actual agent names.

        Args:
            handoff_names: List of handoff names to resolve

        Returns:
            List of resolved agent names
        """
        # For now, just return the handoff names as is
        # In the future, this could be more sophisticated
        return handoff_names

    def instantiate_agents(self) -> None:
        """
        Instantiate agents based on their definitions.
        """
        for agent_name, agent_def in self.agent_definitions.items():
            try:
                # Resolve tools if specified
                tools = None
                if 'tools' in agent_def:
                    tools = self._resolve_tools(agent_def['tools'])

                # Resolve handoffs if specified
                handoffs = None
                if 'handoffs' in agent_def:
                    handoffs = self._resolve_handoffs(agent_def['handoffs'])

                # Create the agent instance
                agent = Agent(
                    name=agent_def['name'],
                    instructions=agent_def['instructions'],
                    model=agent_def.get('model', 'gpt-4o-mini'),  # Default model
                    tools=tools,
                    handoffs=handoffs,
                    model_settings=agent_def.get('model_settings'),
                    tool_use_behavior=agent_def.get('tool_use_behavior')
                )

                # Store the agent instance
                self.agents[agent_name] = agent
                logger.info(f"Instantiated agent: {agent_name}")
            except Exception as e:
                logger.error(f"Error instantiating agent {agent_name}: {e}")

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Get an agent by name.

        Args:
            agent_name: Name of the agent to retrieve

        Returns:
            The agent instance, or None if not found
        """
        return self.agents.get(agent_name)

    def get_all_agents(self) -> Dict[str, Agent]:
        """
        Get all registered agents.

        Returns:
            Dictionary mapping agent names to agent instances
        """
        return self.agents
