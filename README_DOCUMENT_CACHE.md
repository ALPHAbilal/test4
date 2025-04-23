# Document Caching System

This document describes the document caching system implemented to improve processing efficiency and avoid redundant document processing.

## Overview

The document caching system provides the following benefits:

1. **Reduced Processing Time**: Avoids redundant processing of the same documents
2. **Duplicate Detection**: Identifies when the same document is uploaded multiple times, even with different filenames
3. **Memory Efficiency**: Stores processed content and chunks for quick retrieval
4. **Fingerprinting**: Uses SHA-256 hashing to create unique fingerprints for document content

## Components

### 1. DocumentCache Class

The core of the caching system is the `DocumentCache` class in `document_cache.py`. It provides:

- Content caching: Stores document text content
- Chunk caching: Stores processed document chunks
- Fingerprinting: Creates unique identifiers for documents based on content
- Metadata storage: Keeps track of document properties and processing information
- Duplicate detection: Identifies documents with identical content

### 2. Integration Points

The caching system is integrated at several key points:

- **PDF Extraction**: Modified `extract_text_from_pdf()` to check cache before processing
- **Temporary File Processing**: Updated `process_temporary_file()` to use cache for uploaded files
- **Template Retrieval**: Enhanced `retrieve_template_content()` to cache template content

### 3. Cache Management

The system includes tools for cache management:

- `/cache-stats` API endpoint: Returns statistics about the cache
- `/clear-cache` API endpoint: Allows clearing the entire cache or entries older than a specified number of days
- `/cache-dashboard` web interface: Provides a visual dashboard for monitoring and managing the cache

## Usage

### Checking Cache Status

Visit the cache dashboard at `/cache-dashboard` to view:
- Total documents in cache
- Unique document count
- Duplicate document count
- Total content size
- Sample of cached files
- Duplicate file groups

### Clearing the Cache

You can clear the cache in two ways:
1. Through the dashboard UI
2. By making a POST request to `/clear-cache`

To clear specific entries:
```json
POST /clear-cache
{
  "older_than_days": 7
}
```

To clear all entries:
```json
POST /clear-cache
{}
```

## Implementation Details

### Fingerprinting Algorithm

Documents are fingerprinted using SHA-256 hashing of their content:

```python
def generate_fingerprint(self, content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

This creates a unique identifier for each document based on its content, regardless of filename.

### Cache Structure

The cache maintains several dictionaries:
- `content_cache`: Maps file paths to document content
- `chunk_cache`: Maps document fingerprints to processed chunks
- `metadata_cache`: Maps file paths to metadata
- `fingerprint_map`: Maps file paths to fingerprints
- `access_timestamps`: Maps file paths to last access times

### Performance Considerations

- The cache is memory-based for fast access
- Timestamps track when documents were last accessed
- The `clear_cache()` method can remove old entries to manage memory usage

## Future Improvements

Potential enhancements to the caching system:

1. **Persistent Storage**: Save cache to disk between application restarts
2. **Cache Eviction Policies**: Implement LRU or other eviction strategies
3. **Compression**: Compress cached content to reduce memory usage
4. **Partial Content Matching**: Detect similar (not just identical) documents
5. **Distributed Caching**: Support for distributed cache in multi-server deployments
