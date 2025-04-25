"""
Function Tool Module

This module provides a decorator for creating function tools.
"""

import functools
import inspect
import logging
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger(__name__)

def function_tool(func=None, *, strict_mode=True):
    """
    Decorator for creating function tools.
    
    Args:
        func: The function to decorate
        strict_mode: Whether to enforce strict parameter validation
        
    Returns:
        The decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # In a real implementation, this would validate parameters and handle errors
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in function tool {func.__name__}: {e}")
                raise
                
        # Add metadata to the function
        wrapper.is_function_tool = True
        wrapper.strict_mode = strict_mode
        
        # Add a method to invoke the tool directly
        async def on_invoke_tool(ctx, parameters):
            # In a real implementation, this would validate parameters and handle errors
            try:
                # Extract parameters from the input
                if isinstance(parameters, dict):
                    return await func(ctx, **parameters)
                else:
                    return await func(ctx, parameters)
            except Exception as e:
                logger.error(f"Error invoking tool {func.__name__}: {e}")
                raise
                
        wrapper.on_invoke_tool = on_invoke_tool
        
        return wrapper
        
    if func is None:
        return decorator
    return decorator(func)
