# ContentProcessorAgent

The ContentProcessorAgent is a specialized agent responsible for processing and summarizing knowledge base content. It plays a crucial role in the agent-based architecture by handling document content processing tasks.

## Purpose

The primary purpose of the ContentProcessorAgent is to:

1. Analyze document content to understand its structure and key information
2. Create concise summaries of lengthy documents
3. Extract specific information requested by the user
4. Format responses in a clear, organized manner

## Implementation

The ContentProcessorAgent is implemented as a standalone agent with its own instructions and model configuration. It is designed to be called by the WorkflowRouterAgent when knowledge base content needs to be processed.

```json
{
  "name": "ContentProcessorAgent",
  "instructions": "You are a specialized content processing agent responsible for analyzing, summarizing, and extracting information from knowledge base documents...",
  "model": "gpt-4o-mini"
}
```

## Input Format

The ContentProcessorAgent receives input with the following structure:

```json
{
  "document_content": "The full or partial content of the document",
  "source_filename": "The filename or source of the document",
  "user_query": "The original query from the user",
  "content_length": 12345,  // Optional: The total length of the content
  "is_truncated": false     // Optional: Whether the content has been truncated
}
```

## Response Format

The agent's response is formatted as markdown and includes:

1. A brief introduction mentioning the source document
2. The main content addressing the user's query
3. A conclusion or summary
4. Any relevant citations or references to specific sections of the document

Example response structure:

```markdown
# Information from [Document Name]

## Overview
[Brief overview of the document and its relevance to the query]

## [Relevant Section 1]
[Information from the document addressing the query]

## [Relevant Section 2]
[Additional information from the document addressing the query]

## Summary
[Concise summary of the key points]

---
Source: [Document Name]
```

## Integration with Orchestration Engine

The ContentProcessorAgent is integrated with the orchestration engine through the following mechanism:

1. The WorkflowRouterAgent determines that knowledge base content needs to be processed
2. The orchestration engine calls the ContentProcessorAgent with the document content
3. The ContentProcessorAgent processes the content and returns a formatted response
4. The orchestration engine returns the response to the user

## Benefits

The ContentProcessorAgent provides several benefits to the application:

1. **Specialized Processing**: Dedicated agent for document content processing
2. **Consistent Formatting**: Standardized response format for knowledge base content
3. **Context-Aware Processing**: Considers the user's original query when processing content
4. **Improved User Experience**: Well-structured responses that address the user's query

## Future Improvements

Potential future improvements for the ContentProcessorAgent include:

1. **Multi-Document Processing**: Process and synthesize information from multiple documents
2. **Enhanced Summarization**: Improve summarization capabilities for different document types
3. **Personalized Responses**: Tailor responses based on user preferences and history
4. **Multilingual Support**: Process documents in multiple languages
