"""
DOCX Generation Tools

This module provides tools for generating DOCX files from markdown content.
"""

import asyncio
import logging
from typing import Union
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DOCXGenerationResult(BaseModel):
    status: str  # "success" or "error"
    file_path: str = None  # Path to the generated file
    file_name: str = None  # Filename for downloading
    message: str  # Success or error message

@function_tool(strict_mode=False)
async def generate_docx_from_markdown(ctx: RunContextWrapper, markdown_content: str, template_name: str) -> DOCXGenerationResult:
    """Converts markdown content into a professionally formatted DOCX file.

    Args:
        markdown_content: The populated markdown content from the template
        template_name: Name of the original template for reference

    Returns:
        Object containing the path to the generated DOCX file and status
    """
    try:
        # Import the docx_generator module
        try:
            from tools.docx_generator import markdown_to_docx
        except ImportError:
            try:
                import docx_generator
                markdown_to_docx = docx_generator.markdown_to_docx
            except ImportError:
                logger.error("Could not import docx_generator module")
                return DOCXGenerationResult(
                    status="error",
                    file_path=None,
                    file_name=None,
                    message="Error: DOCX generation module not available"
                )

        # Generate the DOCX file asynchronously
        file_path, file_name = await asyncio.to_thread(markdown_to_docx, markdown_content, template_name)

        # Return success result
        return DOCXGenerationResult(
            status="success",
            file_path=file_path,
            file_name=file_name,
            message="DOCX file successfully generated"
        )
    except Exception as e:
        logger.error(f"Error generating DOCX: {e}", exc_info=True)
        return DOCXGenerationResult(
            status="error",
            file_path=None,
            file_name=None,
            message=f"Error generating DOCX: {str(e)}"
        )
