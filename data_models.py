"""
Shared data models for the RAG application.

This module contains Pydantic models that are shared between different
components of the application to avoid circular imports.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class ExtractedData(BaseModel):
    """Data extracted from documents with status information."""
    data: Dict[str, Optional[str]]  # Dictionary with string keys and optional string values
    status: str = "success"
    error_message: Optional[str] = None

class DocumentSection(BaseModel):
    """Section of a document with name and content."""
    name: str
    content: str

class DocumentAnalysis(BaseModel):
    """Analysis of a document's structure and content."""
    doc_type: str = Field(description="The type of document (e.g., employment_contract, invoice, general_document)")
    structure: List[DocumentSection] = Field(default_factory=list, description="List of document sections")
    key_sections: List[str] = Field(default_factory=list, description="List of key section names")
    confidence: float = Field(default=0.0, description="Confidence score for the document analysis")
    language: Optional[str] = Field(default=None, description="Detected language of the document")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata about the document")