"""
Memory Tools

This module provides tools for working with agent memory.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import memory_store from core.memory
try:
    from core.memory import memory_store
except ImportError:
    logger.error("Could not import memory_store from core.memory")
    memory_store = None

class MemoryResult(BaseModel):
    """Result of a memory operation."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@function_tool(strict_mode=False)
async def get_agent_memory(ctx: RunContextWrapper, agent_name: Optional[str] = None) -> MemoryResult:
    """
    Get memory for an agent.
    
    Args:
        agent_name: Name of the agent. If not provided, uses the current agent's name.
        
    Returns:
        Memory for the agent
    """
    logger.info(f"[Tool Call] get_agent_memory: agent_name='{agent_name}'")
    
    if not memory_store:
        return MemoryResult(
            success=False,
            message="Memory store not available",
            data=None
        )
    
    try:
        # Get the current agent name if not provided
        if not agent_name:
            agent_name = ctx.context.get("current_agent_name")
            if not agent_name:
                return MemoryResult(
                    success=False,
                    message="Agent name not provided and not available in context",
                    data=None
                )
        
        # Get the session ID from the context
        session_id = ctx.context.get("chat_id")
        
        # Get the memory
        memory = memory_store.get_memory(agent_name, session_id)
        
        return MemoryResult(
            success=True,
            message=f"Retrieved memory for agent {agent_name}",
            data=memory
        )
    except Exception as e:
        logger.error(f"Error getting memory for agent {agent_name}: {e}")
        return MemoryResult(
            success=False,
            message=f"Error getting memory: {str(e)}",
            data=None
        )

@function_tool(strict_mode=False)
async def update_agent_memory(ctx: RunContextWrapper, memory_update: Dict[str, Any], agent_name: Optional[str] = None) -> MemoryResult:
    """
    Update memory for an agent.
    
    Args:
        memory_update: Memory update to apply
        agent_name: Name of the agent. If not provided, uses the current agent's name.
        
    Returns:
        Result of the update operation
    """
    logger.info(f"[Tool Call] update_agent_memory: agent_name='{agent_name}'")
    
    if not memory_store:
        return MemoryResult(
            success=False,
            message="Memory store not available",
            data=None
        )
    
    try:
        # Get the current agent name if not provided
        if not agent_name:
            agent_name = ctx.context.get("current_agent_name")
            if not agent_name:
                return MemoryResult(
                    success=False,
                    message="Agent name not provided and not available in context",
                    data=None
                )
        
        # Get the session ID from the context
        session_id = ctx.context.get("chat_id")
        
        # Update the memory
        memory_store.update_memory(agent_name, memory_update, session_id)
        
        # Get the updated memory
        updated_memory = memory_store.get_memory(agent_name, session_id)
        
        return MemoryResult(
            success=True,
            message=f"Updated memory for agent {agent_name}",
            data=updated_memory
        )
    except Exception as e:
        logger.error(f"Error updating memory for agent {agent_name}: {e}")
        return MemoryResult(
            success=False,
            message=f"Error updating memory: {str(e)}",
            data=None
        )

@function_tool(strict_mode=False)
async def add_to_agent_memory_list(ctx: RunContextWrapper, list_key: str, item: Any, agent_name: Optional[str] = None, max_items: Optional[int] = None) -> MemoryResult:
    """
    Add an item to a list in an agent's memory.
    
    Args:
        list_key: Key for the list in the memory
        item: Item to add to the list
        agent_name: Name of the agent. If not provided, uses the current agent's name.
        max_items: Maximum number of items to keep in the list
        
    Returns:
        Result of the operation
    """
    logger.info(f"[Tool Call] add_to_agent_memory_list: agent_name='{agent_name}', list_key='{list_key}'")
    
    if not memory_store:
        return MemoryResult(
            success=False,
            message="Memory store not available",
            data=None
        )
    
    try:
        # Get the current agent name if not provided
        if not agent_name:
            agent_name = ctx.context.get("current_agent_name")
            if not agent_name:
                return MemoryResult(
                    success=False,
                    message="Agent name not provided and not available in context",
                    data=None
                )
        
        # Get the session ID from the context
        session_id = ctx.context.get("chat_id")
        
        # Add the item to the list
        memory_store.add_to_memory_list(agent_name, list_key, item, session_id, max_items)
        
        # Get the updated memory
        updated_memory = memory_store.get_memory(agent_name, session_id)
        
        return MemoryResult(
            success=True,
            message=f"Added item to list {list_key} in memory for agent {agent_name}",
            data=updated_memory
        )
    except Exception as e:
        logger.error(f"Error adding to memory list for agent {agent_name}: {e}")
        return MemoryResult(
            success=False,
            message=f"Error adding to memory list: {str(e)}",
            data=None
        )
