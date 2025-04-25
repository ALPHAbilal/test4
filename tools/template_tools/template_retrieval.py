"""
Template Retrieval Tools

This module provides tools for retrieving templates.
"""

import os
import re
import logging
from typing import Union
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel
from werkzeug.utils import secure_filename

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

# Import TEMPLATE_DIR from app.py
try:
    from app import TEMPLATE_DIR
except ImportError:
    # Default value if app.py cannot be imported
    TEMPLATE_DIR = os.path.join(os.getcwd(), "templates")
    logger.warning(f"Could not import TEMPLATE_DIR from app.py, using default: {TEMPLATE_DIR}")

# Define models for return types
class RetrievalSuccess(BaseModel):
    content: str
    source_filename: str

class RetrievalError(BaseModel):
    error_message: str

@function_tool(strict_mode=False)
async def retrieve_template_content(ctx: RunContextWrapper, template_name: str) -> Union[RetrievalSuccess, RetrievalError]:
    """Retrieves the text content of a specified document template (txt, md, pdf)."""
    logger.info(f"[Tool Call] retrieve_template_content: template_name='{template_name}'")
    try:
        # Use asyncio.to_thread to run the file operations asynchronously
        async def process_template():
            # Sanitize template_name but preserve extension
            original_filename = secure_filename(template_name)
            base_name, ext = os.path.splitext(original_filename)
            if not ext: ext = ".md" # Default to markdown if no extension provided

            # Check if extension is allowed for templates
            allowed_template_exts = {'.txt', '.md', '.pdf'}
            if ext.lower() not in allowed_template_exts:
                logger.error(f"Attempted to retrieve template with unsupported extension: {original_filename}")
                return RetrievalError(error_message=f"Unsupported template file type '{ext}'. Allowed: {', '.join(allowed_template_exts)}")

            final_filename = f"{base_name}{ext}" # Reconstruct potentially sanitized name
            template_path = os.path.join(TEMPLATE_DIR, final_filename)

            # Security check: Ensure path is still within TEMPLATE_DIR
            if not os.path.exists(template_path) or os.path.commonpath([TEMPLATE_DIR]) != os.path.commonpath([TEMPLATE_DIR, template_path]):
                logger.error(f"[Tool Error] Template file not found or invalid path: {template_path}")
                return RetrievalError(error_message=f"Template '{template_name}' not found.")

            # --- Extract content based on type ---
            content = ""
            logger.info(f"Reading template file: {template_path}")
            if ext.lower() == ".pdf":
                # Use our robust PDF text extraction function
                content = extract_text_from_pdf(template_path)
            elif ext.lower() in [".txt", ".md"]:
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            if not content:
                logger.warning(f"Extracted empty content from template: {final_filename}")
                return RetrievalError(error_message=f"Could not extract content from template '{final_filename}'.")

            logger.info(f"[Tool Result] Retrieved template '{final_filename}'. Length: {len(content)}")
            cleaned_content = re.sub(r'\s+', ' ', content).strip()
            return RetrievalSuccess(content=cleaned_content, source_filename=f"Template: {original_filename}")

        # Run the template processing function asynchronously
        return await process_template()

    except Exception as e:
         logger.error(f"[Tool Error] Error retrieving template '{template_name}': {e}", exc_info=True)
         return RetrievalError(error_message=f"Error retrieving template: {str(e)}")
