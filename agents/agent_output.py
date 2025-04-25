"""
Agent Output Schema Module

This module provides a schema for agent output.
"""

class AgentOutputSchema:
    """Schema for agent output."""
    
    def __init__(self, **kwargs):
        """Initialize the agent output schema."""
        for key, value in kwargs.items():
            setattr(self, key, value)
