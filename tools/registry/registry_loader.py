"""
Registry Loader Module

This module provides utility functions for loading and initializing the tool registry.
"""

import os
import logging
from typing import Optional
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

def initialize_tool_registry(config_path: Optional[str] = None) -> ToolRegistry:
    """
    Initialize the tool registry with the specified configuration.
    
    Args:
        config_path: Path to the tool configuration file. If None, uses the default path.
        
    Returns:
        Initialized ToolRegistry instance
    """
    # Use default path if none provided
    if config_path is None:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to default config file
        config_path = os.path.join(os.path.dirname(current_dir), 'config', 'tool_definitions.json')
    
    # Create and initialize the registry
    registry = ToolRegistry()
    
    try:
        # Check if config file exists
        if os.path.exists(config_path):
            # Load tool definitions from config file
            registry.load_tool_definitions(config_path)
            
            # Instantiate tools
            registry.instantiate_tools()
            
            logger.info(f"Tool registry initialized with {len(registry.tools)} tools from config")
        else:
            # If config file doesn't exist, scan directories
            logger.info(f"Tool config file not found at {config_path}, scanning directories")
            
            # Get the tools directory
            tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Scan subdirectories for tools
            kb_tools_dir = os.path.join(tools_dir, 'kb_tools')
            file_tools_dir = os.path.join(tools_dir, 'file_tools')
            template_tools_dir = os.path.join(tools_dir, 'template_tools')
            extraction_tools_dir = os.path.join(tools_dir, 'extraction_tools')
            
            # Scan each directory
            registered_tools = set()
            for directory in [kb_tools_dir, file_tools_dir, template_tools_dir, extraction_tools_dir]:
                if os.path.exists(directory):
                    tools = registry.scan_directory(directory)
                    registered_tools.update(tools)
            
            logger.info(f"Tool registry initialized with {len(registry.tools)} tools from directory scan")
        
        return registry
    except Exception as e:
        logger.error(f"Error initializing tool registry: {e}")
        raise
