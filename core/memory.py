"""
Memory Module

This module provides functionality for agents to store and retrieve memory.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import asyncio
import os

logger = logging.getLogger(__name__)

class MemoryStore:
    """
    Memory store for agents to store and retrieve memory.
    
    This class provides functionality to:
    1. Store memory for an agent
    2. Retrieve memory for an agent
    3. Update memory for an agent
    4. Delete memory for an agent
    """
    
    def __init__(self, persistence_dir: Optional[str] = None):
        """
        Initialize the memory store.
        
        Args:
            persistence_dir: Directory to persist memory to disk. If None, memory is not persisted.
        """
        self.memories = {}  # Dictionary mapping agent_id to memory
        self.persistence_dir = persistence_dir
        
        # Create persistence directory if it doesn't exist
        if persistence_dir and not os.path.exists(persistence_dir):
            os.makedirs(persistence_dir)
            logger.info(f"Created persistence directory: {persistence_dir}")
        
        # Load persisted memories if available
        if persistence_dir:
            self._load_persisted_memories()
    
    def _get_memory_key(self, agent_name: str, session_id: Optional[str] = None) -> str:
        """
        Get the memory key for an agent.
        
        Args:
            agent_name: Name of the agent
            session_id: Optional session ID for session-specific memory
            
        Returns:
            Memory key
        """
        if session_id:
            return f"{agent_name}:{session_id}"
        else:
            return agent_name
    
    def _get_persistence_path(self, memory_key: str) -> str:
        """
        Get the persistence path for a memory key.
        
        Args:
            memory_key: Memory key
            
        Returns:
            Persistence path
        """
        if not self.persistence_dir:
            return None
        
        # Replace any characters that are not allowed in filenames
        safe_key = memory_key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return os.path.join(self.persistence_dir, f"{safe_key}.json")
    
    def _load_persisted_memories(self) -> None:
        """
        Load persisted memories from disk.
        """
        if not self.persistence_dir:
            return
        
        try:
            # Get all JSON files in the persistence directory
            for filename in os.listdir(self.persistence_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.persistence_dir, filename)
                    
                    try:
                        # Load the memory from the file
                        with open(file_path, "r") as f:
                            memory_data = json.load(f)
                        
                        # Get the memory key from the filename
                        memory_key = os.path.splitext(filename)[0].replace("_", ":")
                        
                        # Store the memory
                        self.memories[memory_key] = memory_data
                        logger.info(f"Loaded persisted memory for {memory_key}")
                    except Exception as e:
                        logger.error(f"Error loading persisted memory from {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading persisted memories: {e}")
    
    def _persist_memory(self, memory_key: str) -> None:
        """
        Persist memory to disk.
        
        Args:
            memory_key: Memory key
        """
        if not self.persistence_dir:
            return
        
        try:
            # Get the persistence path
            persistence_path = self._get_persistence_path(memory_key)
            
            # Get the memory
            memory = self.memories.get(memory_key)
            
            # Persist the memory
            with open(persistence_path, "w") as f:
                json.dump(memory, f)
            
            logger.info(f"Persisted memory for {memory_key}")
        except Exception as e:
            logger.error(f"Error persisting memory for {memory_key}: {e}")
    
    def get_memory(self, agent_name: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory for an agent.
        
        Args:
            agent_name: Name of the agent
            session_id: Optional session ID for session-specific memory
            
        Returns:
            Memory for the agent
        """
        memory_key = self._get_memory_key(agent_name, session_id)
        return self.memories.get(memory_key, {})
    
    def set_memory(self, agent_name: str, memory: Dict[str, Any], session_id: Optional[str] = None) -> None:
        """
        Set memory for an agent.
        
        Args:
            agent_name: Name of the agent
            memory: Memory to set
            session_id: Optional session ID for session-specific memory
        """
        memory_key = self._get_memory_key(agent_name, session_id)
        self.memories[memory_key] = memory
        
        # Add timestamp
        self.memories[memory_key]["_last_updated"] = time.time()
        
        # Persist memory
        self._persist_memory(memory_key)
    
    def update_memory(self, agent_name: str, memory_update: Dict[str, Any], session_id: Optional[str] = None) -> None:
        """
        Update memory for an agent.
        
        Args:
            agent_name: Name of the agent
            memory_update: Memory update to apply
            session_id: Optional session ID for session-specific memory
        """
        memory_key = self._get_memory_key(agent_name, session_id)
        
        # Get existing memory or create new one
        memory = self.memories.get(memory_key, {})
        
        # Update memory
        memory.update(memory_update)
        
        # Add timestamp
        memory["_last_updated"] = time.time()
        
        # Store updated memory
        self.memories[memory_key] = memory
        
        # Persist memory
        self._persist_memory(memory_key)
    
    def delete_memory(self, agent_name: str, session_id: Optional[str] = None) -> None:
        """
        Delete memory for an agent.
        
        Args:
            agent_name: Name of the agent
            session_id: Optional session ID for session-specific memory
        """
        memory_key = self._get_memory_key(agent_name, session_id)
        
        # Remove memory
        if memory_key in self.memories:
            del self.memories[memory_key]
        
        # Delete persisted memory
        if self.persistence_dir:
            persistence_path = self._get_persistence_path(memory_key)
            if os.path.exists(persistence_path):
                os.remove(persistence_path)
                logger.info(f"Deleted persisted memory for {memory_key}")
    
    def add_to_memory_list(self, agent_name: str, list_key: str, item: Any, session_id: Optional[str] = None, max_items: Optional[int] = None) -> None:
        """
        Add an item to a list in an agent's memory.
        
        Args:
            agent_name: Name of the agent
            list_key: Key for the list in the memory
            item: Item to add to the list
            session_id: Optional session ID for session-specific memory
            max_items: Maximum number of items to keep in the list
        """
        memory_key = self._get_memory_key(agent_name, session_id)
        
        # Get existing memory or create new one
        memory = self.memories.get(memory_key, {})
        
        # Get existing list or create new one
        memory_list = memory.get(list_key, [])
        
        # Add item to list
        memory_list.append(item)
        
        # Limit list size if max_items is specified
        if max_items and len(memory_list) > max_items:
            memory_list = memory_list[-max_items:]
        
        # Update memory
        memory[list_key] = memory_list
        
        # Add timestamp
        memory["_last_updated"] = time.time()
        
        # Store updated memory
        self.memories[memory_key] = memory
        
        # Persist memory
        self._persist_memory(memory_key)


# Create a singleton instance of the memory store
memory_store = MemoryStore(persistence_dir=os.path.join(os.getcwd(), "data", "memory"))


def get_memory_store() -> MemoryStore:
    """
    Get the memory store instance.
    
    Returns:
        Memory store instance
    """
    return memory_store
