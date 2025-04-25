"""
Memory Module

This module provides functionality for agents to store and retrieve memory.
"""

import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Union, Set
import asyncio
import os
from datetime import datetime, timedelta

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

    def get_memory(self, agent_name: str, session_id: Optional[str] = None,
                  filter_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get memory for an agent.

        Args:
            agent_name: Name of the agent
            session_id: Optional session ID for session-specific memory
            filter_options: Optional filtering options for memory retrieval
                - max_size: Maximum size of memory in bytes
                - max_items_per_list: Maximum number of items per list
                - recency_days: Only include memory from the last N days
                - relevance_query: Query to use for relevance filtering
                - include_sections: List of memory sections to include
                - exclude_sections: List of memory sections to exclude
                - max_entries_per_section: Maximum number of entries per section

        Returns:
            Memory for the agent
        """
        memory_key = self._get_memory_key(agent_name, session_id)
        memory = self.memories.get(memory_key, {})

        # If no filter options, return the full memory
        if not filter_options:
            return memory

        # Apply filtering
        return self._filter_memory(memory, filter_options)

    def _filter_memory(self, memory: Dict[str, Any], filter_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter memory based on the provided options.

        Args:
            memory: Memory to filter
            filter_options: Filtering options

        Returns:
            Filtered memory
        """
        filtered_memory = {}

        # Get filter options
        max_size = filter_options.get("max_size", 1024 * 50)  # Default: 50KB
        max_items_per_list = filter_options.get("max_items_per_list", 10)
        recency_days = filter_options.get("recency_days", 30)
        relevance_query = filter_options.get("relevance_query", "")
        include_sections = filter_options.get("include_sections", [])
        exclude_sections = filter_options.get("exclude_sections", [])
        max_entries_per_section = filter_options.get("max_entries_per_section", 10)

        # Calculate recency threshold
        recency_threshold = time.time() - (recency_days * 24 * 60 * 60)

        # Extract keywords from relevance query
        keywords = set(re.findall(r'\b\w+\b', relevance_query.lower()))

        # Process each section in memory
        for section_key, section_value in memory.items():
            # Skip if section is in exclude_sections
            if section_key in exclude_sections:
                continue

            # Include only if in include_sections (if specified)
            if include_sections and section_key not in include_sections:
                continue

            # Handle special metadata fields
            if section_key.startswith("_"):
                filtered_memory[section_key] = section_value
                continue

            # Handle different section types
            if isinstance(section_value, dict):
                # For dictionary sections (like query_patterns, user_preferences)
                filtered_section = self._filter_dict_section(
                    section_value,
                    keywords,
                    recency_threshold,
                    max_entries_per_section
                )
                if filtered_section:
                    filtered_memory[section_key] = filtered_section

            elif isinstance(section_value, list):
                # For list sections
                filtered_section = self._filter_list_section(
                    section_value,
                    keywords,
                    recency_threshold,
                    max_items_per_list
                )
                if filtered_section:
                    filtered_memory[section_key] = filtered_section
            else:
                # For primitive values
                filtered_memory[section_key] = section_value

        # Check total size
        memory_str = json.dumps(filtered_memory)
        if len(memory_str) > max_size:
            # If still too large, apply more aggressive filtering
            logger.warning(f"Memory size ({len(memory_str)} bytes) exceeds max_size ({max_size} bytes). Applying aggressive filtering.")
            return self._aggressive_filter(filtered_memory, max_size)

        return filtered_memory

    def _filter_dict_section(self, section: Dict[str, Any], keywords: Set[str],
                            recency_threshold: float, max_entries: int) -> Dict[str, Any]:
        """
        Filter a dictionary section of memory.

        Args:
            section: Dictionary section to filter
            keywords: Keywords for relevance filtering
            recency_threshold: Timestamp threshold for recency filtering
            max_entries: Maximum number of entries to include

        Returns:
            Filtered dictionary section
        """
        # Score entries by relevance and recency
        scored_entries = []

        for key, value in section.items():
            score = 0

            # Check relevance
            key_text = key.lower()
            if any(keyword in key_text for keyword in keywords):
                score += 5  # High relevance for key match

            # Check content relevance
            content_text = json.dumps(value).lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in content_text)
            score += keyword_matches

            # Check recency
            last_used = None
            if isinstance(value, dict) and "last_used" in value:
                try:
                    last_used_str = value["last_used"]
                    last_used = time.mktime(datetime.strptime(last_used_str, "%Y-%m-%d").timetuple())
                except (ValueError, TypeError):
                    pass

            if isinstance(value, dict) and "_last_updated" in value:
                last_used = value["_last_updated"]

            if last_used and last_used > recency_threshold:
                # More recent = higher score
                recency_factor = min(5, (last_used - recency_threshold) / (30 * 24 * 60 * 60) * 5)
                score += recency_factor

            # Check success count for query patterns
            if isinstance(value, dict) and "success_count" in value:
                success_count = value["success_count"]
                if isinstance(success_count, (int, float)):
                    score += min(5, success_count)  # Max 5 points for success count

            scored_entries.append((key, value, score))

        # Sort by score (descending) and take top entries
        scored_entries.sort(key=lambda x: x[2], reverse=True)
        top_entries = scored_entries[:max_entries]

        # Create filtered section
        filtered_section = {}
        for key, value, _ in top_entries:
            filtered_section[key] = value

        return filtered_section

    def _filter_list_section(self, section: List[Any], keywords: Set[str],
                           recency_threshold: float, max_items: int) -> List[Any]:
        """
        Filter a list section of memory.

        Args:
            section: List section to filter
            keywords: Keywords for relevance filtering
            recency_threshold: Timestamp threshold for recency filtering
            max_items: Maximum number of items to include

        Returns:
            Filtered list section
        """
        # Score items by relevance and recency
        scored_items = []

        for item in section:
            score = 0

            # Check relevance
            item_text = json.dumps(item).lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in item_text)
            score += keyword_matches

            # Check recency
            timestamp = None
            if isinstance(item, dict) and "timestamp" in item:
                timestamp = item["timestamp"]

            if timestamp and timestamp > recency_threshold:
                # More recent = higher score
                recency_factor = min(5, (timestamp - recency_threshold) / (30 * 24 * 60 * 60) * 5)
                score += recency_factor

            scored_items.append((item, score))

        # Sort by score (descending) and take top items
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # If not enough items with scores, add most recent items
        if len(scored_items) < max_items and len(section) > len(scored_items):
            # Add remaining items sorted by recency (if available)
            remaining_items = [item for item in section if item not in [x[0] for x in scored_items]]

            # Try to sort by timestamp if available
            try:
                remaining_items.sort(key=lambda x: x.get("timestamp", 0) if isinstance(x, dict) else 0, reverse=True)
            except (AttributeError, TypeError):
                pass

            scored_items.extend([(item, 0) for item in remaining_items])

        top_items = [item for item, _ in scored_items[:max_items]]
        return top_items

    def _aggressive_filter(self, memory: Dict[str, Any], max_size: int) -> Dict[str, Any]:
        """
        Apply aggressive filtering to reduce memory size.

        Args:
            memory: Memory to filter
            max_size: Maximum size in bytes

        Returns:
            Aggressively filtered memory
        """
        # Start with essential sections
        essential_sections = ["query_patterns", "conversation_state"]
        filtered_memory = {k: memory[k] for k in essential_sections if k in memory}

        # Add metadata
        for key in memory:
            if key.startswith("_"):
                filtered_memory[key] = memory[key]

        # Check size
        if len(json.dumps(filtered_memory)) <= max_size:
            return filtered_memory

        # Further reduce query_patterns
        if "query_patterns" in filtered_memory:
            patterns = filtered_memory["query_patterns"]
            # Keep only top 3 patterns by success_count
            if len(patterns) > 3:
                sorted_patterns = sorted(
                    patterns.items(),
                    key=lambda x: x[1].get("success_count", 0) if isinstance(x[1], dict) else 0,
                    reverse=True
                )
                filtered_memory["query_patterns"] = dict(sorted_patterns[:3])

        # Check size again
        if len(json.dumps(filtered_memory)) <= max_size:
            return filtered_memory

        # Last resort: Keep only the most essential information
        minimal_memory = {
            "_memory_truncated": True,
            "_last_updated": memory.get("_last_updated", time.time())
        }

        # Add at least one query pattern if available
        if "query_patterns" in filtered_memory and filtered_memory["query_patterns"]:
            top_pattern = max(
                filtered_memory["query_patterns"].items(),
                key=lambda x: x[1].get("success_count", 0) if isinstance(x[1], dict) else 0
            )
            minimal_memory["query_patterns"] = {top_pattern[0]: top_pattern[1]}

        return minimal_memory

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
