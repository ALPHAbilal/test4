"""
Spans Module

This module provides classes for representing spans in traces.
"""

class Span:
    """Representation of a span in a trace."""

    def __init__(self, span_id=None, parent_id=None, name=None, start_time=None, end_time=None):
        """Initialize the span."""
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.start_time = start_time
        self.end_time = end_time

    def __class_getitem__(cls, item):
        """Support for subscripting the class."""
        return cls
