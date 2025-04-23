"""
Document caching system to avoid redundant processing of the same documents.
Includes fingerprinting to detect duplicate documents even with different filenames.
"""

import os
import hashlib
import logging
import time
import json
from typing import Dict, Optional, Tuple, List, Any

# Set up logging
logger = logging.getLogger(__name__)

class DocumentCache:
    """
    A caching system for document content and processed chunks to avoid redundant processing.
    Includes document fingerprinting for efficient cache lookups.
    """
    
    def __init__(self):
        self.content_cache: Dict[str, str] = {}  # Maps file_path to document content
        self.chunk_cache: Dict[str, List[str]] = {}  # Maps document fingerprint to chunks
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}  # Maps file_path to metadata
        self.fingerprint_map: Dict[str, str] = {}  # Maps file_path to fingerprint
        self.access_timestamps: Dict[str, float] = {}  # Maps file_path to last access time
        
    def generate_fingerprint(self, content: str) -> str:
        """
        Generate a unique fingerprint for document content.
        
        Args:
            content: The document content to fingerprint
            
        Returns:
            A unique hash representing the document content
        """
        # Use SHA-256 for a reliable fingerprint
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_document_content(self, file_path: str) -> Optional[str]:
        """
        Retrieve document content from cache if available.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document content if cached, None otherwise
        """
        if file_path in self.content_cache:
            # Update access timestamp
            self.access_timestamps[file_path] = time.time()
            logger.info(f"Cache hit: Retrieved content for {file_path}")
            return self.content_cache[file_path]
        
        logger.info(f"Cache miss: Content for {file_path} not found in cache")
        return None
    
    def get_document_chunks(self, file_path: str) -> Optional[List[str]]:
        """
        Retrieve document chunks from cache if available.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document chunks if cached, None otherwise
        """
        # First check if we have a fingerprint for this file
        if file_path not in self.fingerprint_map:
            logger.info(f"Cache miss: No fingerprint for {file_path}")
            return None
        
        # Get the fingerprint and look up chunks
        fingerprint = self.fingerprint_map[file_path]
        if fingerprint in self.chunk_cache:
            # Update access timestamp
            self.access_timestamps[file_path] = time.time()
            logger.info(f"Cache hit: Retrieved chunks for {file_path} (fingerprint: {fingerprint[:8]}...)")
            return self.chunk_cache[fingerprint]
        
        logger.info(f"Cache miss: Chunks for {file_path} (fingerprint: {fingerprint[:8]}...) not found in cache")
        return None
    
    def cache_document(self, file_path: str, content: str, chunks: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Cache document content, chunks, and metadata.
        
        Args:
            file_path: Path to the document
            content: Document content
            chunks: Optional list of document chunks
            metadata: Optional metadata about the document
            
        Returns:
            Document fingerprint
        """
        # Generate fingerprint
        fingerprint = self.generate_fingerprint(content)
        
        # Cache content
        self.content_cache[file_path] = content
        self.fingerprint_map[file_path] = fingerprint
        self.access_timestamps[file_path] = time.time()
        
        # Cache chunks if provided
        if chunks:
            self.chunk_cache[fingerprint] = chunks
            
        # Cache metadata if provided
        if metadata:
            self.metadata_cache[file_path] = metadata
        
        logger.info(f"Cached document {file_path} with fingerprint {fingerprint[:8]}...")
        return fingerprint
    
    def update_chunks(self, file_path: str, chunks: List[str]) -> bool:
        """
        Update chunks for an already cached document.
        
        Args:
            file_path: Path to the document
            chunks: List of document chunks
            
        Returns:
            True if successful, False if document not in cache
        """
        if file_path not in self.fingerprint_map:
            logger.warning(f"Cannot update chunks: {file_path} not in cache")
            return False
        
        fingerprint = self.fingerprint_map[file_path]
        self.chunk_cache[fingerprint] = chunks
        self.access_timestamps[file_path] = time.time()
        logger.info(f"Updated chunks for {file_path} (fingerprint: {fingerprint[:8]}...)")
        return True
    
    def is_document_cached(self, file_path: str) -> bool:
        """
        Check if a document is in the cache.
        
        Args:
            file_path: Path to the document
            
        Returns:
            True if document is cached, False otherwise
        """
        return file_path in self.content_cache
    
    def get_fingerprint(self, file_path: str) -> Optional[str]:
        """
        Get the fingerprint for a cached document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document fingerprint if cached, None otherwise
        """
        return self.fingerprint_map.get(file_path)
    
    def find_duplicate_documents(self, content: str) -> List[str]:
        """
        Find documents with identical content.
        
        Args:
            content: Document content to check
            
        Returns:
            List of file paths with identical content
        """
        fingerprint = self.generate_fingerprint(content)
        duplicates = []
        
        for file_path, doc_fingerprint in self.fingerprint_map.items():
            if doc_fingerprint == fingerprint:
                duplicates.append(file_path)
                
        return duplicates
    
    def clear_cache(self, older_than_days: Optional[float] = None) -> int:
        """
        Clear the cache, optionally only removing entries older than specified days.
        
        Args:
            older_than_days: Optional number of days, entries older than this will be removed
            
        Returns:
            Number of entries removed
        """
        if older_than_days is None:
            # Clear everything
            count = len(self.content_cache)
            self.content_cache.clear()
            self.chunk_cache.clear()
            self.metadata_cache.clear()
            self.fingerprint_map.clear()
            self.access_timestamps.clear()
            logger.info(f"Cleared entire document cache ({count} entries)")
            return count
        
        # Clear only old entries
        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 86400)  # 86400 seconds in a day
        to_remove = []
        
        for file_path, timestamp in self.access_timestamps.items():
            if timestamp < cutoff_time:
                to_remove.append(file_path)
        
        # Remove old entries
        for file_path in to_remove:
            fingerprint = self.fingerprint_map.get(file_path)
            if fingerprint:
                self.chunk_cache.pop(fingerprint, None)
            
            self.content_cache.pop(file_path, None)
            self.metadata_cache.pop(file_path, None)
            self.fingerprint_map.pop(file_path, None)
            self.access_timestamps.pop(file_path, None)
        
        logger.info(f"Cleared {len(to_remove)} old entries from document cache")
        return len(to_remove)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "document_count": len(self.content_cache),
            "unique_document_count": len(set(self.fingerprint_map.values())),
            "chunk_cache_count": len(self.chunk_cache),
            "total_content_size_bytes": sum(len(content) for content in self.content_cache.values()),
            "total_chunks": sum(len(chunks) for chunks in self.chunk_cache.values()),
            "duplicate_count": len(self.content_cache) - len(set(self.fingerprint_map.values()))
        }

# Create a global instance of the document cache
document_cache = DocumentCache()
