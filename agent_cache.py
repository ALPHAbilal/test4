"""
Agent Cache Module

This module provides a caching mechanism for OpenAI Agents to avoid creating
and deleting agents for each session, improving performance and reducing API calls.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from agents import Agent

# Setup logging
logger = logging.getLogger(__name__)

class AgentCache:
    """
    A cache for OpenAI Agents that allows reusing agents across sessions.
    
    This cache stores agent instances by their name and configuration,
    allowing the application to reuse existing agents instead of creating
    new ones for each request.
    """
    
    def __init__(self, max_age_seconds: int = 3600, max_size: int = 20):
        """
        Initialize the agent cache.
        
        Args:
            max_age_seconds: Maximum age of cached agents in seconds (default: 1 hour)
            max_size: Maximum number of agents to keep in the cache (default: 20)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_age_seconds = max_age_seconds
        self.max_size = max_size
        logger.info(f"Initialized AgentCache with max_age={max_age_seconds}s, max_size={max_size}")
    
    def get_agent(self, name: str, instructions: str, tools: List[Any], model: str, **kwargs) -> Optional[Agent]:
        """
        Get an agent from the cache or return None if not found.
        
        Args:
            name: Agent name
            instructions: Agent instructions
            tools: List of tools for the agent
            model: Model to use for the agent
            **kwargs: Additional agent parameters
            
        Returns:
            Cached agent instance or None if not in cache
        """
        # Create a cache key based on agent parameters
        cache_key = self._create_cache_key(name, instructions, tools, model, kwargs)
        
        # Check if the agent is in the cache and not expired
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            age = time.time() - entry["timestamp"]
            
            if age < self.max_age_seconds:
                logger.info(f"Cache hit for agent '{name}' (age: {age:.1f}s)")
                # Update the timestamp to keep the agent fresh
                entry["timestamp"] = time.time()
                return entry["agent"]
            else:
                logger.info(f"Cache expired for agent '{name}' (age: {age:.1f}s)")
                # Remove the expired entry
                del self._cache[cache_key]
        
        return None
    
    def add_agent(self, agent: Agent, name: str, instructions: str, tools: List[Any], model: str, **kwargs) -> None:
        """
        Add an agent to the cache.
        
        Args:
            agent: Agent instance to cache
            name: Agent name
            instructions: Agent instructions
            tools: List of tools for the agent
            model: Model to use for the agent
            **kwargs: Additional agent parameters
        """
        # Create a cache key based on agent parameters
        cache_key = self._create_cache_key(name, instructions, tools, model, kwargs)
        
        # Add the agent to the cache
        self._cache[cache_key] = {
            "agent": agent,
            "timestamp": time.time(),
            "name": name
        }
        
        logger.info(f"Added agent '{name}' to cache (cache size: {len(self._cache)})")
        
        # Check if we need to clean up the cache
        if len(self._cache) > self.max_size:
            self._cleanup_oldest()
    
    def _create_cache_key(self, name: str, instructions: str, tools: List[Any], model: str, kwargs: Dict[str, Any]) -> str:
        """
        Create a cache key based on agent parameters.
        
        Args:
            name: Agent name
            instructions: Agent instructions
            tools: List of tools for the agent
            model: Model to use for the agent
            kwargs: Additional agent parameters
            
        Returns:
            Cache key string
        """
        # Use the name and a hash of the instructions and tool names as the key
        tool_names = [getattr(tool, "__name__", str(tool)) for tool in tools]
        key_parts = [
            name,
            hash(instructions),
            hash(tuple(sorted(tool_names))),
            model
        ]
        
        # Add any additional kwargs that affect the agent's behavior
        for k, v in sorted(kwargs.items()):
            if k in ["output_type", "tool_use_behavior"]:
                key_parts.append(f"{k}:{v}")
        
        return ":".join(str(part) for part in key_parts)
    
    def _cleanup_oldest(self) -> None:
        """
        Remove the oldest entries from the cache to keep it under the maximum size.
        """
        # Sort entries by timestamp
        sorted_entries = sorted(
            [(k, v["timestamp"], v["name"]) for k, v in self._cache.items()],
            key=lambda x: x[1]
        )
        
        # Remove the oldest entries
        entries_to_remove = sorted_entries[:len(sorted_entries) - self.max_size]
        for key, _, name in entries_to_remove:
            del self._cache[key]
            logger.info(f"Removed oldest agent '{name}' from cache")
    
    def clear(self) -> None:
        """
        Clear all entries from the cache.
        """
        self._cache.clear()
        logger.info("Cleared agent cache")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["timestamp"] > self.max_age_seconds
        ]
        
        for key in expired_keys:
            name = self._cache[key]["name"]
            del self._cache[key]
            logger.info(f"Removed expired agent '{name}' from cache")
        
        return len(expired_keys)
    
    @property
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of agents in the cache
        """
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        now = time.time()
        ages = [now - v["timestamp"] for v in self._cache.values()]
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "max_age": self.max_age_seconds,
            "avg_age": sum(ages) / len(ages) if ages else 0,
            "oldest_age": max(ages) if ages else 0,
            "newest_age": min(ages) if ages else 0
        }

# Create a global instance of the agent cache
agent_cache = AgentCache()
