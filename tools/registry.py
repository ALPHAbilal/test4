"""
Tool Registry Module

This module provides functionality for registering and retrieving tools.
"""

import os
import json
import logging
import importlib
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry for tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools = []
        
    def register_tool(self, tool):
        """Register a tool with the registry."""
        self.tools.append(tool)
        logger.info(f"Registered tool: {tool.name if hasattr(tool, 'name') else 'unnamed'}")
            
    def get_tool(self, name):
        """Get a tool by name."""
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == name:
                return tool
        return None
    
    def get_all_tools(self):
        """Get all registered tools."""
        return self.tools

def initialize_tool_registry():
    """Initialize the tool registry with tools from the configuration."""
    registry = ToolRegistry()
    
    try:
        # Load tool definitions from the configuration file
        config_path = os.path.join('tools', 'config', 'tool_definitions.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Create and register tools from the configuration
            if 'tools' in config:
                for tool_def in config['tools']:
                    try:
                        # Get the module and function
                        module_name = tool_def.get('module')
                        function_name = tool_def.get('function')
                        
                        if module_name and function_name:
                            # Import the module
                            module = importlib.import_module(module_name)
                            
                            # Get the function
                            tool = getattr(module, function_name)
                            
                            # Set the name if not already set
                            if not hasattr(tool, 'name'):
                                tool.name = tool_def.get('name')
                                
                            # Register the tool
                            registry.register_tool(tool)
                        else:
                            logger.warning(f"Tool definition missing module or function: {tool_def}")
                    except Exception as e:
                        logger.error(f"Error registering tool {tool_def.get('name')}: {e}")
        else:
            logger.warning(f"Tool configuration file not found: {config_path}")
    except Exception as e:
        logger.error(f"Error initializing tool registry: {e}")
        
    return registry
