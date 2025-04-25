"""
Data Extraction Tools

This module provides tools for extracting data from documents.
"""

import logging
from typing import List, Dict, Optional
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ExtractedData(BaseModel):
    """Structured data extracted from context sources."""
    extracted_fields: Dict[str, str]
    missing_fields: List[str] = []
    confidence: Dict[str, float] = {}
    notes: str = ""

@function_tool(strict_mode=False)
async def extract_data_for_template(ctx: RunContextWrapper, context_sources: List[str], required_fields: List[str], document_analyses: Optional[List[Dict]] = None) -> ExtractedData:
    """Extracts specific data fields required for a template from provided text context sources."""
    logger.info(f"[Tool Call] extract_data_for_template. Required: {required_fields}. Sources: {len(context_sources)} provided.")

    if document_analyses:
        logger.info(f"Document analyses provided for extraction: {len(document_analyses)}")

    # Import document_analyzer_integration from app.py
    try:
        from app import document_analyzer_integration
        # Call the integrated DocumentAnalyzerAgent implementation
        return await document_analyzer_integration.extract_data_for_template_integrated(ctx, context_sources, required_fields, document_analyses)
    except ImportError:
        logger.error("Could not import document_analyzer_integration from app.py")
        # Fallback implementation
        return ExtractedData(
            extracted_fields={},
            missing_fields=required_fields,
            confidence={},
            notes="Error: Document analyzer integration not available"
        )
