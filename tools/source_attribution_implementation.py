"""
Source Attribution Implementation for Knowledge Base UX Optimization

This module contains the enhanced models and functions needed to implement
source attribution in the KB responses.
"""

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

# --- Enhanced Models for Source Attribution ---

class SourceMetadata(BaseModel):
    """Metadata about a source document used in retrieval."""
    file_id: str = Field(description="The ID of the file in the vector store")
    file_name: str = Field(description="The name of the file")
    section: Optional[str] = Field(None, description="Section or page information if available")
    confidence: float = Field(description="Confidence score for this source (0-1)")
    
class EnhancedRetrievalSuccess(BaseModel):
    """Enhanced version of RetrievalSuccess that includes source metadata."""
    content: str = Field(description="The retrieved content")
    sources: List[SourceMetadata] = Field(description="Metadata about the sources of this content")
    
    @property
    def source_filename(self) -> str:
        """Backward compatibility with original RetrievalSuccess."""
        if self.sources and len(self.sources) > 0:
            return self.sources[0].file_name
        return "Unknown source"

class EnhancedRetrievalError(BaseModel):
    """Enhanced version of RetrievalError that includes more detailed error information."""
    error_message: str = Field(description="The error message")
    details: Optional[str] = Field(None, description="Additional error details")
    query_attempted: Optional[str] = Field(None, description="The query that was attempted")

# --- Enhanced Final Answer Model ---

class EnhancedFinalAnswer(BaseModel):
    """Enhanced version of FinalAnswer that includes source information."""
    markdown_response: str = Field(description="The markdown formatted response")
    sources_used: List[SourceMetadata] = Field(
        default_factory=list,
        description="Sources used to generate this response"
    )
    
    def format_with_sources(self) -> str:
        """Format the response with source attribution."""
        if not self.sources_used:
            return self.markdown_response
            
        # Add sources section at the end
        sources_section = "\n\n## Sources\n"
        for i, source in enumerate(self.sources_used, 1):
            sources_section += f"{i}. **{source.file_name}** "
            if source.section:
                sources_section += f"(Section: {source.section}) "
            sources_section += f"- Confidence: {source.confidence:.2f}\n"
            
        return self.markdown_response + sources_section

# --- Enhanced get_kb_document_content Function ---

async def enhanced_get_kb_document_content(ctx, document_type: str, query_or_identifier: str) -> Union[EnhancedRetrievalSuccess, EnhancedRetrievalError]:
    """Enhanced version of get_kb_document_content that includes source metadata.
    
    This is a template for how to modify the existing function to include source attribution.
    The actual implementation would need to be integrated with the existing function.
    """
    # Most of the implementation would be the same as the original function
    # The key differences are:
    
    # 1. When processing search results, collect source metadata
    # Example (assuming search_results is the result from vector store search):
    sources = []
    if search_results and search_results.data:
        for res in search_results.data:
            source = SourceMetadata(
                file_id=res.file_id,
                file_name=res.filename or f"FileID:{res.file_id[-6:]}",
                section=None,  # Would need to extract from metadata if available
                confidence=res.score  # Assuming score is available
            )
            sources.append(source)
            
        # Combine content as before
        content = "\n\n".join(re.sub(r'\s+', ' ', part.text).strip() 
                             for res in search_results.data 
                             for part in res.content 
                             if part.type == 'text')
                             
        # Return enhanced result
        return EnhancedRetrievalSuccess(
            content=content,
            sources=sources
        )
    else:
        # Return enhanced error
        return EnhancedRetrievalError(
            error_message=f"No KB content found for query related to '{document_type}'.",
            query_attempted=query_or_identifier
        )

# --- Enhanced Final Synthesizer Agent Instructions ---

ENHANCED_SYNTHESIZER_INSTRUCTIONS = """Synthesize final answer from query & context (KB or temp file).

IMPORTANT RULES:
1. NEVER fabricate information that is not in the provided context
2. If the context doesn't contain information to answer the query, clearly state this limitation
3. Do not try to be helpful by making up information - accuracy is more important than helpfulness
4. Only provide information that is explicitly supported by the context
5. If asked about a specific country, language, or document that isn't in the context, clearly state that this information is not available

SOURCE ATTRIBUTION REQUIREMENTS:
1. For each piece of information you provide, indicate which source it came from
2. Use inline citations like [Source 1] or [Source 2] when appropriate
3. If multiple sources provide the same information, cite all relevant sources
4. If information comes from a specific section of a document, mention this
5. Include a "Sources" section at the end of your response listing all sources used

FORMAT:
- Use markdown formatting for your response
- Use headers, lists, and emphasis where appropriate
- For complex information, use tables or bullet points
- Include a Sources section at the end with numbered references

EXAMPLE RESPONSE FORMAT:
```
# Answer to Query

The information you requested is... [Source 1]

Additional details include... [Source 2]

## Sources
1. Document A - Confidence: 0.92
2. Document B (Section: Legal Requirements) - Confidence: 0.85
```
"""

# --- Example of how to update the final_synthesizer_agent definition ---

"""
final_synthesizer_agent = Agent(
    name="FinalSynthesizerAgent",
    instructions=ENHANCED_SYNTHESIZER_INSTRUCTIONS,
    model=COMPLETION_MODEL,
    output_type=EnhancedFinalAnswer
)
"""

# --- Example of how to modify extract_final_answer function ---

def enhanced_extract_final_answer(run_result):
    """Enhanced version of extract_final_answer that handles source attribution.
    
    This is a template for how to modify the existing function.
    The actual implementation would need to be integrated with the existing function.
    """
    try:
        final_output = run_result.final_output
        
        # Case 1: We got an EnhancedFinalAnswer instance
        if isinstance(final_output, EnhancedFinalAnswer):
            return final_output.format_with_sources()
            
        # Handle other cases as in the original function...
        
    except Exception as e:
        # Error handling as in the original function...
        pass
