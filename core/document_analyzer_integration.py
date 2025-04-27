"""
DocumentAnalyzerAgent Integration Module

This module provides integration functions to connect the DocumentAnalyzerAgent
with the existing app.py workflow, including support for large document handling.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

# Import the shared data model
from data.data_models import ExtractedData

# Import the required functions from DocumentAnalyzerAgent
from agents.document_analyzer_agent import extract_data_for_template_agent_based

# Setup logging
logger = logging.getLogger(__name__)

# Try to import the document sampling agent
try:
    from agents.document_sampling_agent import get_document_samples
    DOCUMENT_SAMPLING_AVAILABLE = True
    logger.info("Document sampling agent available for large document processing")
except ImportError:
    DOCUMENT_SAMPLING_AVAILABLE = False
    logger.warning("Document sampling agent not available. Large documents may not be processed efficiently.")

async def extract_data_for_template_integrated(ctx, context_sources, required_fields, document_analyses: Optional[List[Dict]] = None):
    """
    Integrated extraction function that properly handles the ExtractedData class
    and supports large document processing.

    Args:
        ctx: The run context
        context_sources: List of document contents
        required_fields: List of fields to extract
        document_analyses: Optional pre-computed document analyses

    Returns:
        ExtractedData object with the extracted data
    """
    logger.info(f"[Integration] extract_data_for_template_integrated called with {len(required_fields)} fields")

    if document_analyses:
        logger.info(f"[Integration] Document analyses provided: {len(document_analyses)}")

    try:
        # Call the agent-based implementation with document analyses if available
        extracted_data = await extract_data_for_template_agent_based(ctx, context_sources, required_fields, document_analyses=document_analyses)

        # Ensure all required fields exist, even if null
        for field in required_fields:
            if field not in extracted_data:
                extracted_data[field] = None

        # Return the proper ExtractedData object
        return ExtractedData(
            data=extracted_data,
            status="success",
            error_message=None
        )
    except Exception as e:
        logger.error(f"[Integration Error] extract_data_for_template_integrated failed: {e}", exc_info=True)
        return ExtractedData(
            data={field: None for field in required_fields},
            status="error",
            error_message=str(e)
        )

async def sample_document_if_needed(ctx, document_content: str, document_name: str, max_tokens: int = 8000) -> Tuple[str, float]:
    """
    Sample a document if it's too large, otherwise return the original content.

    Args:
        ctx: The run context
        document_content: The document content
        document_name: The name of the document
        max_tokens: Maximum tokens to include

    Returns:
        Tuple of (processed_content, coverage_ratio)
    """
    # If document is small enough, return as is
    if len(document_content) <= max_tokens or not DOCUMENT_SAMPLING_AVAILABLE:
        return document_content, 1.0

    try:
        # Use the document sampling agent to get representative samples
        logger.info(f"[Integration] Sampling large document: {document_name} ({len(document_content)} chars)")
        sampled_content, coverage = await get_document_samples(ctx, document_content, document_name, max_tokens)
        logger.info(f"[Integration] Document sampled successfully. Coverage: {coverage:.2f}")
        return sampled_content, coverage
    except Exception as e:
        logger.error(f"[Integration Error] Document sampling failed: {e}", exc_info=True)
        # Fall back to simple truncation
        truncated = document_content[:max_tokens]
        coverage = len(truncated) / len(document_content)
        logger.warning(f"[Integration] Falling back to simple truncation. Coverage: {coverage:.2f}")
        return truncated, coverage

# detect_required_fields_from_template_integrated removed - now calling detect_fields_from_template tool directly

async def analyze_document_type_during_upload(ctx, document_content: str, document_name: str) -> Dict[str, Any]:
    """
    Analyze a document during upload to determine its type and metadata.
    This function is used to override user-provided document type with agent-detected type.

    Args:
        ctx: The run context
        document_content: The document content
        document_name: The name of the document

    Returns:
        Dictionary with document type and metadata
    """
    logger.info(f"[Integration] Analyzing document type during upload: {document_name}")

    try:
        # Import the document analyzer function
        from agents.document_analyzer_agent import analyze_document_for_workflow

        # For large documents, use the document sampling agent
        document_size = len(document_content)
        max_tokens = 6000  # Maximum tokens for analysis

        if document_size > max_tokens and DOCUMENT_SAMPLING_AVAILABLE:
            logger.info(f"Document is large ({document_size} chars), using document sampling for type detection")
            try:
                # Get representative samples from the document
                sampled_content, coverage = await get_document_samples(ctx, document_content, document_name, max_tokens)

                # Use the sampled content for analysis
                logger.info(f"Using sampled document content for type detection. Coverage: {coverage:.2f}")
                analysis_content = sampled_content
            except Exception as sampling_error:
                logger.error(f"Error sampling document: {sampling_error}", exc_info=True)
                # Fall back to using the beginning of the document
                logger.warning(f"Falling back to using first {max_tokens} chars of document")
                analysis_content = document_content[:max_tokens]
        else:
            # For small documents, use the full content
            analysis_content = document_content

        # Analyze the document
        analysis_result = await analyze_document_for_workflow(ctx, analysis_content, document_name)

        # Extract the document type and metadata
        doc_type = analysis_result.doc_type
        confidence = analysis_result.confidence
        metadata = analysis_result.metadata

        logger.info(f"[Integration] Document type detected: {doc_type} (confidence: {confidence:.2f})")

        # Return the document type and metadata
        return {
            "document_type": doc_type,
            "confidence": confidence,
            "metadata": metadata,
            "agent_analyzed": True
        }
    except Exception as e:
        logger.error(f"[Integration Error] Document type analysis failed: {e}", exc_info=True)
        return {
            "document_type": "general",
            "confidence": 0.0,
            "metadata": {},
            "agent_analyzed": False,
            "error": str(e)
        }