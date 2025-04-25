"""
Run Context Wrapper Module

This module provides a wrapper for the run context.
"""

class RunContextWrapper:
    """Wrapper for the run context."""
    
    def __init__(self, context=None):
        """Initialize the run context wrapper."""
        self.context = context or {}
