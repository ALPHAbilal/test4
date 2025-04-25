"""
Tool Registry Module

This module provides a registry for dynamically loading and managing tools.
"""

import os
import json
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Callable, Set
from agents import function_tool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    A registry for dynamically loading and managing tools.
    
    This class provides functionality to:
    1. Load tool definitions from configuration files
    2. Dynamically import tool functions
    3. Store and retrieve tool instances
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools = {}  # Dictionary mapping tool names to tool instances
        self.tool_definitions = {}  # Dictionary mapping tool names to their definitions
    
    def load_tool_definitions(self, config_path: str) -> None:
        """
        Load tool definitions from a configuration file.
        
        Args:
            config_path: Path to the configuration file (JSON)
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Store tool definitions
            for tool_def in config.get('tools', []):
                tool_name = tool_def.get('name')
                if tool_name:
                    self.tool_definitions[tool_name] = tool_def
                    logger.info(f"Loaded definition for tool: {tool_name}")
                else:
                    logger.warning(f"Skipping tool definition without name: {tool_def}")
            
            logger.info(f"Loaded {len(self.tool_definitions)} tool definitions from {config_path}")
        except Exception as e:
            logger.error(f"Error loading tool definitions from {config_path}: {e}")
            raise
    
    def register_tool(self, name: str, tool_func: Callable) -> None:
        """
        Register a tool function by name.
        
        Args:
            name: Name of the tool
            tool_func: The tool function to register
        """
        # Check if the function is already decorated with @function_tool
        if hasattr(tool_func, 'is_function_tool') and tool_func.is_function_tool:
            self.tools[name] = tool_func
        else:
            # Decorate the function with @function_tool
            decorated_func = function_tool(strict_mode=False)(tool_func)
            self.tools[name] = decorated_func
        
        logger.info(f"Registered tool: {name}")
    
    def import_tool(self, module_path: str, function_name: str) -> Optional[Callable]:
        """
        Import a tool function from a module.
        
        Args:
            module_path: Path to the module (e.g., 'tools.kb_tools')
            function_name: Name of the function to import
            
        Returns:
            The imported function, or None if import failed
        """
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, function_name):
                return getattr(module, function_name)
            else:
                logger.error(f"Function {function_name} not found in module {module_path}")
                return None
        except ImportError as e:
            logger.error(f"Error importing module {module_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error importing tool {function_name} from {module_path}: {e}")
            return None
    
    def instantiate_tools(self) -> None:
        """
        Instantiate tools based on their definitions.
        """
        for tool_name, tool_def in self.tool_definitions.items():
            try:
                # Get module and function name
                module_path = tool_def.get('module')
                function_name = tool_def.get('function')
                
                if not module_path or not function_name:
                    logger.error(f"Missing module or function for tool {tool_name}")
                    continue
                
                # Import the tool function
                tool_func = self.import_tool(module_path, function_name)
                if tool_func:
                    # Register the tool
                    self.register_tool(tool_name, tool_func)
                else:
                    logger.error(f"Failed to import tool {tool_name} from {module_path}.{function_name}")
            except Exception as e:
                logger.error(f"Error instantiating tool {tool_name}: {e}")
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The tool function, or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_tools(self, tool_names: List[str]) -> List[Callable]:
        """
        Get multiple tools by name.
        
        Args:
            tool_names: List of tool names to retrieve
            
        Returns:
            List of tool functions (only those that were found)
        """
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary mapping tool names to tool functions
        """
        return self.tools
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Set[str]:
        """
        Scan a directory for Python files and register any function_tool decorated functions.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            Set of tool names that were registered
        """
        registered_tools = set()
        
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            # Skip if not recursive and not the top directory
            if not recursive and root != directory:
                continue
            
            # Process Python files
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    # Construct the module path
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, os.path.dirname(directory))
                    module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                    
                    try:
                        # Import the module
                        module = importlib.import_module(module_path)
                        
                        # Find all function_tool decorated functions
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if callable(attr) and hasattr(attr, 'is_function_tool') and attr.is_function_tool:
                                # Register the tool
                                self.register_tool(attr_name, attr)
                                registered_tools.add(attr_name)
                    except ImportError as e:
                        logger.error(f"Error importing module {module_path}: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error scanning {module_path}: {e}")
        
        return registered_tools
