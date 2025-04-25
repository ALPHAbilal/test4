"""
Temporary File Processor Tools

This module provides tools for processing temporary files.
"""

import os
import re
import logging
from typing import Union
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import extract_text_from_pdf from pymupdf_tools
try:
    from tools.pymupdf_tools import extract_text_from_pdf
except ImportError:
    try:
        from pymupdf_tools import extract_text_from_pdf
    except ImportError:
        # Define a fallback function if extract_text_from_pdf cannot be imported
        def extract_text_from_pdf(file_path):
            logger.error(f"extract_text_from_pdf not available, cannot process PDF: {file_path}")
            return "Error: PDF processing not available"

# Define models for return types
class RetrievalSuccess(BaseModel):
    content: str
    source_filename: str

class RetrievalError(BaseModel):
    error_message: str

@function_tool(strict_mode=False)
async def process_temporary_file(ctx: RunContextWrapper, filename: str) -> Union[RetrievalSuccess, RetrievalError]:
    """Reads and returns the text content of a previously uploaded temporary file for use as context."""
    logger.info(f"[Tool Call] process_temporary_file: filename='{filename}'")
    tool_context = ctx.context
    temp_file_info = tool_context.get("temp_file_info")
    if not temp_file_info or temp_file_info.get("filename") != filename:
        return RetrievalError(error_message=f"Temporary file '{filename}' not available.")
    file_path = temp_file_info.get("path")
    if not file_path or not os.path.exists(file_path):
        return RetrievalError(error_message=f"Temporary file path invalid for '{filename}'.")
    try:
        text_content = ""
        file_lower = filename.lower()
        if file_lower.endswith(".pdf"):
            # Use our robust PDF text extraction function
            text_content = extract_text_from_pdf(file_path)
        elif file_lower.endswith((".txt", ".md")):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        else:
            return RetrievalError(error_message=f"Unsupported temporary file type: {filename}")
        cleaned_content = re.sub(r'\s+', ' ', text_content).strip()
        logger.info(f"[Tool Result] Processed temporary file '{filename}'. Length: {len(cleaned_content)}")
        return RetrievalSuccess(content=cleaned_content, source_filename=f"Uploaded: {filename}")
    except Exception as e:
        logger.error(f"[Tool Error] Failed read/process temp file '{filename}': {e}", exc_info=True)
        return RetrievalError(error_message=f"Error processing temp file: {str(e)}")
