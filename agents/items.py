"""
Items Module

This module provides classes for representing items in agent runs.
"""

class ToolCallItem:
    """Item representing a tool call."""
    
    def __init__(self, call_id=None, tool_name=None, parameters=None):
        """Initialize the tool call item."""
        self.call_id = call_id
        self.tool_name = tool_name
        self.parameters = parameters or {}
        
class ToolCallOutputItem:
    """Item representing a tool call output."""
    
    def __init__(self, call_id=None, output=None):
        """Initialize the tool call output item."""
        self.call_id = call_id
        self.output = output
