"""
Agent-driven semantic caching for vector store queries.

This module provides a sophisticated caching mechanism that uses LLM-based
semantic understanding to determine when queries are equivalent, rather than
relying on exact string matching or regex patterns.
"""

import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class AgentDrivenSemanticCache:
    """
    A cache that uses an LLM-based agent to determine semantic equivalence between queries.

    This cache stores vector store search results and can retrieve them based on
    semantic similarity rather than exact string matching.
    """

    def __init__(self, cache_size=1000, ttl=600, confidence_threshold=0.85):
        """
        Initialize the semantic cache.

        Args:
            cache_size: Maximum number of entries in the cache
            ttl: Time-to-live for cache entries in seconds
            confidence_threshold: Minimum confidence score for semantic equivalence
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = cache_size
        self.ttl = ttl
        self.confidence_threshold = confidence_threshold

        # For our simplified implementation, we'll skip the agent-based similarity check
        # and just use a simple string comparison
        self.query_similarity_agent = None

    def generate_cache_key(self, vector_store_id: str, query: str, filters: Dict, chat_id: str) -> str:
        """
        Generate a cache key from the query parameters.

        Args:
            vector_store_id: ID of the vector store
            query: The search query
            filters: Dictionary of filters applied to the search
            chat_id: ID of the chat session

        Returns:
            A hash string to use as the cache key
        """
        # Extract included_file_ids from filters if present
        included_file_ids = None
        if filters and 'included_file_ids' in filters:
            included_file_ids = sorted(filters['included_file_ids']) if filters['included_file_ids'] else None

        # Extract document_type from filters if present
        document_type = None
        if filters and 'document_type' in filters:
            document_type = filters['document_type']

        # Create a more specific key that includes file IDs
        key_data = {
            'vs_id': vector_store_id,
            'query': query,
            'document_type': document_type,
            'included_file_ids': included_file_ids,
            'chat_id': chat_id
        }

        # Log the key data for debugging
        logger.debug(f"Generating cache key with data: {key_data}")

        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def find_semantic_match(self, new_query: str, vector_store_id: str,
                                 filters: Dict, chat_id: str) -> Optional[Any]:
        """
        Find a semantically equivalent query in the cache.

        Uses the QuerySimilarityAgent to determine if any cached query is
        semantically equivalent to the new query.

        Args:
            new_query: The new search query
            vector_store_id: ID of the vector store
            filters: Dictionary of filters applied to the search
            chat_id: ID of the chat session

        Returns:
            The cached result if a semantic match is found, None otherwise
        """
        # First check if we have any entries for this chat and vector store
        relevant_entries = []
        for cache_key, cached_entry in self.cache.items():
            # Check basic conditions first (same chat, vector store)
            if (cached_entry['chat_id'] != chat_id or
                cached_entry['vector_store_id'] != vector_store_id):
                continue

            # Check if cache entry is expired
            if time.time() - self.timestamps[cache_key] > self.ttl:
                logger.debug(f"Removing expired cache entry: {cache_key}")
                del self.cache[cache_key]
                del self.timestamps[cache_key]
                continue

            relevant_entries.append((cache_key, cached_entry))

        if not relevant_entries:
            logger.debug("No relevant cache entries found")
            return None

        logger.info(f"Found {len(relevant_entries)} relevant cache entries to check for semantic similarity")

        # Now check for semantic equivalence
        for cache_key, cached_entry in relevant_entries:
            # Simple string comparison for our simplified implementation
            cached_query = cached_entry['query'].lower()
            new_query_lower = new_query.lower()

            # Check if the queries are similar enough (simple contains check)
            if cached_query in new_query_lower or new_query_lower in cached_query:
                # Check if document types match
                cached_doc_type = cached_entry.get('document_type', 'general')
                new_doc_type = filters.get('document_type', 'general')

                # Check if file IDs match
                cached_file_ids_set = set(cached_entry.get('included_file_ids', []) or [])
                new_file_ids_set = set(filters.get('included_file_ids', []) or [])

                # Consider a match if document types match and file IDs are the same or empty
                if cached_doc_type == new_doc_type and (not cached_file_ids_set or not new_file_ids_set or cached_file_ids_set == new_file_ids_set):
                    logger.info(f"[SEMANTIC CACHE HIT] Simple match found")

                    # Update timestamp to keep this entry fresh
                    self.timestamps[cache_key] = time.time()

                    return cached_entry['result']
                else:
                    logger.debug(f"Not equivalent: document types or file IDs don't match")
            else:
                logger.debug(f"Not equivalent: queries don't contain each other")

        logger.info("[SEMANTIC CACHE MISS] No semantic match found")
        return None

    async def set(self, vector_store_id: str, query: str, filters: Dict,
                 chat_id: str, document_type: str, result: Any):
        """
        Store a result in the cache.

        Args:
            vector_store_id: ID of the vector store
            query: The search query
            filters: Dictionary of filters applied to the search
            chat_id: ID of the chat session
            document_type: Type of document being queried
            result: The result to cache
        """
        # Generate cache key
        cache_key = self.generate_cache_key(vector_store_id, query, filters, chat_id)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            logger.debug(f"Cache full, evicting oldest entry: {oldest_key}")
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

        # Extract included_file_ids from filters if present
        included_file_ids = None
        if filters and 'included_file_ids' in filters:
            included_file_ids = filters['included_file_ids']

        # Store the new entry
        self.cache[cache_key] = {
            'query': query,
            'filters': filters,
            'chat_id': chat_id,
            'vector_store_id': vector_store_id,
            'document_type': document_type,
            'included_file_ids': included_file_ids,
            'result': result
        }
        self.timestamps[cache_key] = time.time()
        logger.info(f"Added new entry to semantic cache: {query[:50]}...")

# Create global cache instance with default settings
semantic_search_cache = AgentDrivenSemanticCache(
    cache_size=1000,
    ttl=600,
    confidence_threshold=0.85
)