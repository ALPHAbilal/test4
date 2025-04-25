"""
Async Utilities Module

This module provides utilities for working with asynchronous code in Flask.
"""

import asyncio
from functools import wraps
from typing import Callable, Any

def setup_async_for_flask():
    """
    Set up the event loop for Flask.
    """
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def async_to_sync(func: Callable) -> Callable:
    """
    Convert an async function to a sync function.
    
    Args:
        func: The async function to convert
        
    Returns:
        The converted sync function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper
