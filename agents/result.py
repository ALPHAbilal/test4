"""
Result Module

This module provides classes for representing the results of agent runs.
"""

class RunResult:
    """Result of an agent run."""
    
    def __init__(self, final_output=None, messages=None, new_items=None):
        """Initialize the run result."""
        self.final_output = final_output
        self.messages = messages or []
        self.new_items = new_items or []
        
    def to_input_list(self):
        """Convert the result to an input list."""
        return self.messages
