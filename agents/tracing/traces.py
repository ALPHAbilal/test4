"""
Traces Module

This module provides classes for representing traces.
"""

class Trace:
    """Representation of a trace."""
    
    def __init__(self, trace_id=None, spans=None):
        """Initialize the trace."""
        self.trace_id = trace_id
        self.spans = spans or []
