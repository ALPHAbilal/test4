"""
DocumentAnalyzerAgent Module

This module provides a comprehensive agent for document analysis and data extraction
that replaces regex-based approaches with semantic understanding using LLMs.
"""

import json
import logging
import re
from typing import Dict, List, Optional
from pydantic import BaseModel

# Import shared data models
from data.data_models import DocumentAnalysis

from agents import Agent, Runner, function_tool, RunContextWrapper

# Setup logging
logger = logging.getLogger(__name__)

# --- Data Models ---
class DocumentSectionInternal(BaseModel):
    """Internal section of a document used for processing"""
    name: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    content: str

    model_config = {
        "extra": "forbid"
    }

class DocumentMetadata(BaseModel):
    """Metadata about a document"""
    # Define specific fields that might be in metadata
    # Using Optional fields with default values to handle any case
    document_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    # Add a field for any other metadata as a simple string
    additional_info: Optional[str] = None

    model_config = {
        "extra": "forbid"
    }

class DocumentStructure(BaseModel):
    """Document structure analysis result"""
    document_type: str
    sections: List[DocumentSectionInternal]
    metadata: DocumentMetadata

    model_config = {
        "extra": "forbid"
    }

class ExtractedField(BaseModel):
    """Single extracted field with metadata"""
    field_name: str
    value: Optional[str]
    confidence: float
    source_location: Optional[str]
    alternatives: Optional[List[str]]

    model_config = {
        "extra": "forbid"
    }

class ExtractedDataResult(BaseModel):
    """Collection of extracted fields with rich metadata"""
    fields: List[ExtractedField]
    document_type: str
    status: str = "success"
    error_message: Optional[str] = None

    model_config = {
        "extra": "forbid"
    }

    def to_simple_dict(self) -> Dict[str, Optional[str]]:
        """Convert to simple dictionary format for compatibility with existing code"""
        return {field.field_name: field.value for field in self.fields}

# --- Function Tools ---
@function_tool(strict_mode=False)
async def analyze_document_structure(ctx: RunContextWrapper, document_content: str, document_name: str) -> DocumentStructure:
    """
    Analyze document structure to identify sections, layout, and document type.

    Args:
        document_content: The text content of the document
        document_name: The name of the document (helpful for type inference)

    Returns:
        DocumentStructure object with document type and section information
    """
    logger.info(f"[Tool Call] analyze_document_structure for: {document_name}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        empty_metadata = DocumentMetadata(additional_info="Configuration error")
        return DocumentStructure(
            document_type="unknown",
            sections=[],
            metadata=empty_metadata
        )

    # For large documents, use the document sampling agent
    document_size = len(document_content)
    max_tokens = 4000  # Maximum tokens for analysis

    if document_size > max_tokens:
        logger.info(f"Document is large ({document_size} chars), using document sampling")
        try:
            # Import the document sampling function
            from agents.document_sampling_agent import get_document_samples

            # Get representative samples from the document
            sampled_content, coverage = await get_document_samples(ctx, document_content, document_name, max_tokens)

            # Use the sampled content for analysis
            logger.info(f"Using sampled document content. Coverage: {coverage:.2f}")
            analysis_content = sampled_content
        except Exception as sampling_error:
            logger.error(f"Error sampling document: {sampling_error}", exc_info=True)
            # Fall back to using the beginning of the document
            logger.warning(f"Falling back to using first {max_tokens} chars of document")
            analysis_content = document_content[:max_tokens]
    else:
        # For small documents, use the full content
        analysis_content = document_content

    # Use model to analyze document structure
    prompt = f"""
    Analyze the structure of this document and identify its type, sections, and organization.

    Document Name: {document_name}

    Document Content:
    {analysis_content}

    Provide a detailed analysis including:
    1. Document type (e.g., employment_contract, invoice, ID_card, general_document)
    2. Main sections and their boundaries
    3. Key metadata about the document

    Return your analysis as a JSON object with these keys:
    - document_type: string
    - sections: array of objects with {{"name": string, "start_index": number, "end_index": number, "content": string}}
    - metadata: object with any relevant document metadata
    """

    try:
        # Call the model - use asyncio.to_thread for synchronous client
        try:
            # Try async version first
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
        except TypeError:
            # Fall back to synchronous version with asyncio.to_thread
            import asyncio
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

        # Parse the response
        content = response.choices[0].message.content
        structure_data = json.loads(content)

        # Process sections
        sections = []
        for section_data in structure_data.get("sections", []):
            try:
                sections.append(DocumentSectionInternal(
                    name=section_data.get("name", "Unnamed Section"),
                    start_index=section_data.get("start_index"),
                    end_index=section_data.get("end_index"),
                    content=section_data.get("content", "")
                ))
            except Exception as section_error:
                logger.warning(f"Error processing section: {section_error}")
                # Add a simplified section if there's an error
                sections.append(DocumentSectionInternal(
                    name="Error Processing Section",
                    content=str(section_data)[:200]
                ))

        # Process metadata
        metadata_dict = structure_data.get("metadata", {})
        metadata = DocumentMetadata(
            document_date=metadata_dict.get("date"),
            author=metadata_dict.get("author"),
            title=metadata_dict.get("title"),
            language=metadata_dict.get("language"),
            additional_info=json.dumps(metadata_dict) if metadata_dict else None
        )

        # Create and return DocumentStructure
        return DocumentStructure(
            document_type=structure_data.get("document_type", "unknown"),
            sections=sections,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"[Tool Error] Document structure analysis failed: {e}", exc_info=True)
        error_metadata = DocumentMetadata(additional_info=f"Error: {str(e)}")
        return DocumentStructure(
            document_type="unknown",
            sections=[],
            metadata=error_metadata
        )

@function_tool(strict_mode=False)
async def extract_fields_from_document(
    ctx: RunContextWrapper,
    document_content: str,
    required_fields: List[str],
    document_structure: Optional[DocumentStructure] = None
) -> ExtractedDataResult:
    """
    Extract specific fields from document content using semantic understanding.

    Args:
        document_content: The text content of the document
        required_fields: List of field names to extract
        document_structure: Optional document structure from previous analysis

    Returns:
        ExtractedDataResult object with extracted fields and metadata
    """
    logger.info(f"[Tool Call] extract_fields_from_document. Required fields: {required_fields}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        return ExtractedDataResult(
            fields=[],
            document_type="unknown",
            status="error",
            error_message="Configuration error"
        )

    # Normalize field names
    normalized_fields = [field.lower().replace(" ", "_") for field in required_fields]

    # Prepare document type and structure information
    doc_type = "unknown"
    structure_info = ""
    if document_structure:
        doc_type = document_structure.document_type
        # Create a simplified representation of sections for the prompt
        sections_info = []
        for section in document_structure.sections:
            section_preview = section.content[:200] + "..." if len(section.content) > 200 else section.content
            sections_info.append({"name": section.name, "content": section_preview})

        # Create a simplified representation of metadata
        metadata_dict = {
            "document_date": document_structure.metadata.document_date,
            "author": document_structure.metadata.author,
            "title": document_structure.metadata.title,
            "language": document_structure.metadata.language,
            "additional_info": document_structure.metadata.additional_info
        }
        # Remove None values
        metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}

        structure_info = f"""
        Document Type: {document_structure.document_type}

        Document Sections:
        {json.dumps(sections_info, indent=2)}

        Document Metadata:
        {json.dumps(metadata_dict, indent=2)}
        """

    # For large documents, use the document sampling agent
    document_size = len(document_content)
    max_tokens = 6000  # Maximum tokens for extraction

    if document_size > max_tokens:
        logger.info(f"Document is large ({document_size} chars), using document sampling for field extraction")
        try:
            # Import the document sampling function
            from agents.document_sampling_agent import get_document_samples

            # Get representative samples from the document
            document_name = document_structure.metadata.title if document_structure and document_structure.metadata.title else "Unknown Document"
            sampled_content, coverage = await get_document_samples(ctx, document_content, document_name, max_tokens)

            # Use the sampled content for extraction
            logger.info(f"Using sampled document content for field extraction. Coverage: {coverage:.2f}")
            analysis_content = sampled_content

            # Add sampling information to structure info
            structure_info += f"\n\nNote: Document has been sampled for analysis. Coverage: {coverage:.2f} of original content."
        except Exception as sampling_error:
            logger.error(f"Error sampling document: {sampling_error}", exc_info=True)
            # Fall back to using the beginning of the document
            logger.warning(f"Falling back to using first {max_tokens} chars of document")
            analysis_content = document_content[:max_tokens]
    else:
        # For small documents, use the full content
        analysis_content = document_content

    # Create field-specific guidelines based on document type
    field_guidelines = []
    for field in normalized_fields:
        if 'date' in field:
            field_guidelines.append(f"- {field}: Look for dates in various formats (DD/MM/YYYY, Month DD, YYYY)")
        elif 'name' in field or 'nom' in field:
            field_guidelines.append(f"- {field}: Look for person names, typically in formats like 'First Last' or 'Last, First'")
        elif 'address' in field or 'adresse' in field:
            field_guidelines.append(f"- {field}: Look for physical addresses, which may span multiple lines")
        elif 'amount' in field or 'montant' in field or 'salary' in field or 'salaire' in field:
            field_guidelines.append(f"- {field}: Look for monetary amounts, possibly with currency symbols")
        elif 'id' in field or 'number' in field or 'numéro' in field:
            field_guidelines.append(f"- {field}: Look for identification numbers, which may have specific formats")
        elif 'employer' in field or 'employeur' in field or 'company' in field or 'société' in field:
            field_guidelines.append(f"- {field}: Look for company or organization names")
        elif 'job' in field or 'title' in field or 'poste' in field or 'fonction' in field:
            field_guidelines.append(f"- {field}: Look for job titles or positions")
        elif 'duration' in field or 'durée' in field or 'period' in field or 'période' in field:
            field_guidelines.append(f"- {field}: Look for time periods or durations")
        else:
            field_guidelines.append(f"- {field}: Extract this field based on context")

    field_guidelines_text = "\n".join(field_guidelines)

    # Get the user query from the context if available
    user_query = ctx.context.get("current_query", "") if hasattr(ctx, "context") else ""

    # Create the prompt
    prompt = f"""
    Extract the following fields from this document using semantic understanding rather than pattern matching.

    Required Fields:
    {field_guidelines_text}

    {structure_info}

    Document Content:
    {analysis_content}

    User Query: {user_query}

    For each field:
    1. Extract the most likely value
    2. Provide a confidence score (0.0-1.0)
    3. Note where in the document you found it
    4. List alternative values if applicable

    Return your extraction as a JSON array of objects with these keys:
    - field_name: string (normalized field name)
    - value: string or null
    - confidence: number (0.0-1.0)
    - source_location: string (description of where in the document)
    - alternatives: array of strings (alternative values) or null
    """

    try:
        # Call the model - use asyncio.to_thread for synchronous client
        try:
            # Try async version first
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
        except TypeError:
            # Fall back to synchronous version with asyncio.to_thread
            import asyncio
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

        # Parse the response
        content = response.choices[0].message.content
        extraction_data = json.loads(content)

        # Process the extracted fields
        extracted_fields = []
        if isinstance(extraction_data, list):
            for field_data in extraction_data:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field_data.get("field_name", "unknown"),
                        value=field_data.get("value"),
                        confidence=field_data.get("confidence", 0.0),
                        source_location=field_data.get("source_location"),
                        alternatives=field_data.get("alternatives")
                    )
                )
        elif isinstance(extraction_data, dict) and "fields" in extraction_data:
            # Handle case where model returns a wrapper object
            for field_data in extraction_data["fields"]:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field_data.get("field_name", "unknown"),
                        value=field_data.get("value"),
                        confidence=field_data.get("confidence", 0.0),
                        source_location=field_data.get("source_location"),
                        alternatives=field_data.get("alternatives")
                    )
                )

        # Ensure all required fields are included
        existing_fields = {field.field_name for field in extracted_fields}
        for field in normalized_fields:
            if field not in existing_fields:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field,
                        value=None,
                        confidence=0.0,
                        source_location=None,
                        alternatives=None
                    )
                )

        # If we used document sampling, adjust confidence scores
        if document_size > max_tokens and 'coverage' in locals():
            # Reduce confidence proportionally to coverage
            for field in extracted_fields:
                # Only reduce if confidence is already high
                if field.confidence > 0.7:
                    # Adjust confidence based on coverage, but don't reduce below 0.5
                    field.confidence = max(0.5, field.confidence * coverage)

        # Create and return ExtractedDataResult
        return ExtractedDataResult(
            fields=extracted_fields,
            document_type=doc_type,
            status="success"
        )
    except Exception as e:
        logger.error(f"[Tool Error] Field extraction failed: {e}", exc_info=True)
        # Create empty fields for all required fields
        empty_fields = [
            ExtractedField(
                field_name=field,
                value=None,
                confidence=0.0,
                source_location=None,
                alternatives=None
            )
            for field in normalized_fields
        ]
        return ExtractedDataResult(
            fields=empty_fields,
            document_type=doc_type if doc_type else "unknown",
            status="error",
            error_message=str(e)
        )

@function_tool(strict_mode=False)
async def detect_fields_from_template(ctx: RunContextWrapper, input_data: Dict) -> List[str]:
    """
    Detect required fields from a template using semantic understanding.

    Args:
        input_data: Dictionary containing template_content and template_name

    Returns:
        List of detected field names
    """
    # Extract template_content and template_name from the input dictionary
    template_content = input_data.get("template_content", "")
    template_name = input_data.get("template_name", "Unknown Template")

    logger.info(f"[Tool Call] detect_fields_from_template for: {template_name}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        return []

    # Create the prompt
    prompt = f"""
    Analyze this document template and identify all fields that need to be filled.

    Template Name: {template_name}

    Template Content:
    {template_content[:4000]}  # Limit content length

    Look for:
    1. Explicit placeholders like [Field Name], {{Field Name}}, <Field Name>, etc.
    2. Implied fields based on context (e.g., "The employee, _____, agrees to...")
    3. Standard fields expected in this type of document

    Return a JSON array of field names, normalized to lowercase with underscores instead of spaces.
    """

    try:
        # Call the model - use asyncio.to_thread for synchronous client
        try:
            # Try async version first
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
        except TypeError:
            # Fall back to synchronous version with asyncio.to_thread
            import asyncio
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

        # Parse the response
        content = response.choices[0].message.content
        fields_data = json.loads(content)

        # Process the fields
        if isinstance(fields_data, list):
            detected_fields = fields_data
        elif isinstance(fields_data, dict) and "fields" in fields_data:
            detected_fields = fields_data["fields"]
        else:
            detected_fields = []
            logger.warning(f"Unexpected fields data format: {fields_data}")

        # Normalize field names
        normalized_fields = [
            field.lower().replace(" ", "_") if isinstance(field, str) else str(field)
            for field in detected_fields
        ]

        # Remove duplicates while preserving order
        unique_fields = []
        for field in normalized_fields:
            if field not in unique_fields:
                unique_fields.append(field)

        logger.info(f"[Tool Result] Detected fields from template: {unique_fields}")
        return unique_fields
    except Exception as e:
        logger.error(f"[Tool Error] Field detection failed: {e}", exc_info=True)
        return []

# --- Integration Function ---
async def extract_data_for_template_agent_based(ctx: RunContextWrapper, context_sources: List[str], required_fields: List[str], document_analyses: Optional[List[Dict]] = None) -> Dict[str, Optional[str]]:
    """
    Extract data for a template using the document analyzer agent.

    Args:
        ctx: The run context wrapper
        context_sources: List of document contents
        required_fields: List of fields to extract
        document_analyses: Optional pre-computed document analyses

    Returns:
        Dictionary of extracted fields
    """
    logger.info(f"Extracting data for template with {len(context_sources)} sources and {len(required_fields)} required fields")

    # Initialize all fields to None
    extracted_fields = {field: None for field in required_fields}

    # If no context sources, return empty values
    if not context_sources or len(context_sources) == 0:
        logger.warning("No context provided for extraction")
        return extracted_fields

    try:
        # For large documents, use the document sampling agent
        combined_content = "\n\n".join(context_sources)
        document_size = len(combined_content)
        max_tokens = 4000  # Maximum tokens for extraction

        if document_size > max_tokens:
            logger.info(f"Combined content is large ({document_size} chars), using document sampling")
            try:
                # Import the document sampling function
                from agents.document_sampling_agent import get_document_samples

                # Get representative samples from the document
                sampled_content, coverage = await get_document_samples(ctx, combined_content, "Template Data", max_tokens)

                # Use the sampled content for extraction
                logger.info(f"Using sampled content for template data extraction. Coverage: {coverage:.2f}")
                analysis_content = sampled_content
            except Exception as sampling_error:
                logger.error(f"Error sampling document: {sampling_error}", exc_info=True)
                # Fall back to using the beginning of the document
                logger.warning(f"Falling back to using first {max_tokens} chars of document")
                analysis_content = combined_content[:max_tokens]
        else:
            # For small documents, use the full content
            analysis_content = combined_content

        # Use document analyses if provided
        doc_analysis_info = ""
        if document_analyses:
            logger.info(f"Using {len(document_analyses)} document analyses to guide extraction")
            doc_analysis_info = "\n\nDocument Analysis Information:\n"
            for i, analysis in enumerate(document_analyses):
                doc_analysis_info += f"\nDocument {i+1}:\n"
                doc_analysis_info += f"- Type: {analysis.get('doc_type', 'unknown')}\n"

                # Add key sections if available
                if 'key_sections' in analysis and analysis['key_sections']:
                    doc_analysis_info += f"- Key Sections: {', '.join(analysis['key_sections'])}\n"

        # Extract fields using the extract_fields_from_document function
        result = await extract_fields_from_document(
            ctx=ctx,
            document_content=analysis_content,
            required_fields=required_fields
        )

        # Convert the result to a simple dictionary
        if result and result.fields:
            for field in result.fields:
                if field.value is not None:
                    extracted_fields[field.field_name] = field.value

        logger.info(f"Extracted {len([f for f in extracted_fields.values() if f is not None])}/{len(required_fields)} fields")

    except Exception as e:
        logger.error(f"Error in template extraction: {e}", exc_info=True)
        # Fall back to simple extraction
        for field in required_fields:
            for source in context_sources:
                if isinstance(source, str) and field.lower() in source.lower():
                    extracted_fields[field] = f"Extracted {field} from context"
                    break

    return extracted_fields

# --- New Document Analysis Function ---
@function_tool(strict_mode=False)
async def analyze_document_for_workflow(ctx: RunContextWrapper, document_content: str, document_name: str) -> DocumentAnalysis:
    """
    Analyze document structure and return a standardized DocumentAnalysis object for workflow use.

    Args:
        document_content: The text content of the document
        document_name: The name of the document (helpful for type inference)

    Returns:
        DocumentAnalysis object with document type, structure, and key sections
    """
    logger.info(f"[Tool Call] analyze_document_for_workflow for: {document_name}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        return DocumentAnalysis(
            doc_type="unknown",
            confidence=0.0,
            key_sections=[],
            structure=[],
            metadata={"error": "Configuration error"}
        )

    # For large documents, use the document sampling agent
    document_size = len(document_content)
    max_tokens = 6000  # Maximum tokens for analysis

    if document_size > max_tokens:
        logger.info(f"Document is large ({document_size} chars), using document sampling")
        try:
            # Import the document sampling function
            from agents.document_sampling_agent import get_document_samples

            # Get representative samples from the document
            sampled_content, coverage = await get_document_samples(ctx, document_content, document_name, max_tokens)

            # Use the sampled content for analysis
            logger.info(f"Using sampled document content. Coverage: {coverage:.2f}")
            analysis_content = sampled_content
        except Exception as sampling_error:
            logger.error(f"Error sampling document: {sampling_error}", exc_info=True)
            # Fall back to using the beginning of the document
            logger.warning(f"Falling back to using first {max_tokens} chars of document")
            analysis_content = document_content[:max_tokens]
    else:
        # For small documents, use the full content
        analysis_content = document_content

    # Create a prompt that explicitly requests JSON output matching our model
    prompt = f"""
    Analyze the structure of this document and identify its type, sections, and organization.

    Document Name: {document_name}

    Document Content:
    {analysis_content}

    Provide a detailed analysis as a JSON object with these exact keys:
    - doc_type: The type of document (e.g., employment_contract, invoice, general_document)
    - structure: Array of objects with {{"name": string, "content": string}} representing document sections
    - key_sections: Array of strings with names of the most important sections
    - confidence: Number between 0 and 1 indicating confidence in the analysis
    - language: String indicating the detected language (or null if unknown)
    - metadata: Object with any additional relevant metadata as string values

    Example response format:
    {{"doc_type": "employment_contract", "structure": [{{"name": "Introduction", "content": "This agreement..."}}, {{"name": "Compensation", "content": "The employee shall receive..."}}], "key_sections": ["Introduction", "Compensation"], "confidence": 0.85, "language": "en", "metadata": {{"date": "2023-01-15", "parties": "Company X and Employee Y"}}}}

    IMPORTANT: Return ONLY the JSON object with no additional text, explanations, or markdown formatting.
    """

    try:
        # Call the model with explicit JSON response format - use asyncio.to_thread for synchronous client
        try:
            # Try async version first
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
        except TypeError:
            # Fall back to synchronous version with asyncio.to_thread
            import asyncio
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",  # Use appropriate model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

        # Parse the response
        content = response.choices[0].message.content
        logger.debug(f"Raw LLM response: {content}")

        # Handle potential JSON parsing issues
        try:
            analysis_data = json.loads(content)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {json_err}. Raw content: {content[:200]}...")
            # Try to extract JSON if it's wrapped in other text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.error(f"Failed to extract valid JSON from match: {json_match.group(0)[:200]}...")
                    raise
            else:
                raise ValueError(f"Could not find JSON object in response: {content[:200]}...")

        # Process structure sections
        structure_sections = []
        if "structure" in analysis_data and isinstance(analysis_data["structure"], list):
            for section_data in analysis_data["structure"]:
                if isinstance(section_data, dict) and "name" in section_data and "content" in section_data:
                    # Create a dictionary instead of a DocumentSection object
                    structure_sections.append({
                        "name": section_data["name"],
                        "content": section_data["content"]
                    })

        # Process key sections
        key_sections = []
        if "key_sections" in analysis_data and isinstance(analysis_data["key_sections"], list):
            key_sections = [str(section) for section in analysis_data["key_sections"] if section]

        # Process metadata
        metadata = {}
        if "metadata" in analysis_data and isinstance(analysis_data["metadata"], dict):
            metadata = {str(k): str(v) for k, v in analysis_data["metadata"].items() if k and v}

        # If we used document sampling, add that information to metadata
        if document_size > max_tokens:
            metadata["sampling_applied"] = "true"
            metadata["document_coverage"] = f"{coverage:.2f}" if 'coverage' in locals() else "unknown"

        # Create and return DocumentAnalysis
        return DocumentAnalysis(
            doc_type=analysis_data.get("doc_type", "unknown"),
            structure=structure_sections,
            key_sections=key_sections,
            confidence=float(analysis_data.get("confidence", 0.0)),
            language=analysis_data.get("language"),
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"[Tool Error] Document analysis failed: {e}", exc_info=True)
        return DocumentAnalysis(
            doc_type="unknown",
            confidence=0.0,
            key_sections=[],
            structure=[],
            metadata={"error": str(e)}
        )

# --- Compatibility Functions for Existing Code ---

async def extract_data_for_template_agent_based(ctx: RunContextWrapper, context_sources: List[str], required_fields: List[str], template_structure: Optional[List[Dict]] = None, document_analyses: Optional[List[Dict]] = None) -> Dict[str, Optional[str]]:
    """
    Agent-based replacement for the regex-based extract_data_for_template function.

    Args:
        context_sources: List of text content from various sources
        required_fields: List of field names to extract
        template_structure: Optional structure of the template (sections, etc.)
        document_analyses: Optional list of document analyses to guide extraction

    Returns:
        Dictionary mapping field names to extracted values
    """
    logger.info(f"[Agent Call] extract_data_for_template_agent_based. Required: {required_fields}. Sources: {len(context_sources)} provided.")

    # Prepare document analysis information if available
    doc_analysis_info = ""
    if document_analyses:
        doc_analysis_info = "\n\nDocument Analysis Information:\n"
        for i, analysis in enumerate(document_analyses):
            doc_analysis_info += f"\nDocument {i+1}:\n"
            doc_analysis_info += f"- Type: {analysis.get('doc_type', 'unknown')}\n"
            doc_analysis_info += f"- Confidence: {analysis.get('confidence', 0.0)}\n"

            # Add language if available
            if 'language' in analysis and analysis['language']:
                doc_analysis_info += f"- Language: {analysis['language']}\n"

            # Add key sections if available
            if 'key_sections' in analysis and analysis['key_sections']:
                doc_analysis_info += f"- Key Sections: {', '.join(analysis['key_sections'])}\n"

                # Add hints for where to find specific fields based on document type
                doc_type = analysis.get('doc_type', '').lower()
                if 'contract' in doc_type or 'agreement' in doc_type:
                    doc_analysis_info += "- Field Location Hints:\n"
                    doc_analysis_info += "  - Names are typically found in the introduction or signature sections\n"
                    doc_analysis_info += "  - Dates are often in the header or near signatures\n"
                    doc_analysis_info += "  - Monetary values are usually in compensation or payment sections\n"
                elif 'invoice' in doc_type or 'receipt' in doc_type:
                    doc_analysis_info += "- Field Location Hints:\n"
                    doc_analysis_info += "  - Amounts are typically in line items or totals sections\n"
                    doc_analysis_info += "  - Dates are often in the header\n"
                    doc_analysis_info += "  - Company information is usually in the header or footer\n"

            # Add metadata if available
            if 'metadata' in analysis and analysis['metadata']:
                doc_analysis_info += "- Metadata:\n"
                for key, value in analysis['metadata'].items():
                    doc_analysis_info += f"  - {key}: {value}\n"

    # Combine context sources
    combined_context = "\n\n".join(context_sources)
    if not combined_context:
        logger.warning("No context provided for extraction")
        return {field: None for field in required_fields}

    # For large documents, use the document sampling agent
    document_size = len(combined_context)
    max_tokens = 4000  # Maximum tokens for extraction

    if document_size > max_tokens:
        logger.info(f"Combined context is large ({document_size} chars), using document sampling")
        try:
            # Import the document sampling function
            from agents.document_sampling_agent import get_document_samples

            # Get representative samples from the document
            sampled_content, coverage = await get_document_samples(ctx, combined_context, "Template Data", max_tokens)

            # Use the sampled content for extraction
            logger.info(f"Using sampled content for template data extraction. Coverage: {coverage:.2f}")
            analysis_content = sampled_content

            # Add sampling information to doc_analysis_info
            doc_analysis_info += f"\n\nNote: Document has been sampled for analysis. Coverage: {coverage:.2f} of original content."
        except Exception as sampling_error:
            logger.error(f"Error sampling document: {sampling_error}", exc_info=True)
            # Fall back to using the beginning of the document
            logger.warning(f"Falling back to using first {max_tokens} chars of document")
            analysis_content = combined_context[:max_tokens]
    else:
        # For small documents, use the full content
        analysis_content = combined_context

    try:
        # Create a more sophisticated extraction agent with detailed instructions
        extraction_agent = Agent(
            name="DocumentFieldExtractor",
            instructions=f"""You are an expert document field extractor with advanced semantic understanding.

            Your task is to extract the following fields from the provided document: {', '.join(required_fields)}.

            When extracting fields:
            1. Use semantic understanding rather than pattern matching
            2. Consider document type and structure when looking for fields
            3. Pay attention to document analysis information when provided
            4. Look in the appropriate sections for specific field types
            5. Handle different languages appropriately
            6. Normalize dates, names, and monetary values to standard formats

            For each field, provide the value if found in the document, or indicate if it's not found.
            Format your response as a list of field:value pairs, one per line.
            """,
            model="gpt-4o-mini"
        )

        # Create a more detailed prompt with the document content and analysis info
        prompt = f"""Here is the document content to analyze:

        {analysis_content}

        {doc_analysis_info if doc_analysis_info else ''}

        {"Template structure information:" + json.dumps(template_structure, indent=2) if template_structure else ''}

        Please extract values for these fields: {', '.join(required_fields)}

        EXTRACTION GUIDELINES:
        1. Use the document analysis information to understand the document type and structure
        2. Focus on the key sections identified in the document analysis when looking for field values
        3. For each field, provide the most accurate value found in the document
        4. If a field is not found, indicate with "not found" or "null"
        5. For dates, normalize to YYYY-MM-DD format when possible
        6. For monetary values, include the currency symbol/code when available
        7. For names, use the full name as it appears in the document
        8. If multiple values could apply to a field, choose the most appropriate one
        9. If the document is in a language other than English, translate field values when appropriate

        FORMAT YOUR RESPONSE AS:
        field_name1: extracted_value1
        field_name2: extracted_value2
        ...
        """

        # Run the agent using the Runner class (static method)
        result = await Runner.run(extraction_agent, input=prompt, context=ctx.context)

        # Extract the data from the agent's response
        extracted_data = {field: None for field in required_fields}

        # Get the final output from the result
        # The final_output might be a string, dict, or other type
        final_output = result.final_output

        # Convert the final output to a string for pattern matching
        response_text = ""
        if isinstance(final_output, str):
            response_text = final_output
        elif hasattr(final_output, 'markdown_response'):
            response_text = final_output.markdown_response
        elif isinstance(final_output, dict):
            response_text = str(final_output)
        else:
            # Try to convert to string as a fallback
            response_text = str(final_output)

        # Try to find any extracted fields in the agent's response
        for field in required_fields:
            # Look for patterns like "field: value" in the agent's response
            field_pattern = re.compile(rf"{field}:\s*([^\n]+)")
            match = field_pattern.search(response_text)
            if match:
                extracted_data[field] = match.group(1).strip()

        # Log the results
        logger.info(f"[Agent Result] Extracted data: {json.dumps(extracted_data, ensure_ascii=False)}")

        return extracted_data
    except Exception as e:
        logger.error(f"[Agent Error] Document analysis failed: {e}", exc_info=True)
        return {field: None for field in required_fields}

async def detect_required_fields_agent_based(template_content: str, template_name: str) -> List[str]:
    """
    Agent-based replacement for the regex-based detect_required_fields_from_template function.

    Args:
        template_content: The text content of the template
        template_name: The name of the template

    Returns:
        List of detected field names
    """
    from app import get_openai_client

    logger.info(f"[Agent Call] detect_required_fields_agent_based for: {template_name}")

    try:
        # Create a context with the OpenAI client
        context = {"client": get_openai_client()}

        # Create a field detection agent
        field_detection_agent = Agent(
            name="TemplateFieldDetector",
            instructions="""Analyze document templates and identify all fields that need to be filled.
            Return a list of field names, normalized to lowercase with underscores instead of spaces.
            Format your response as a bulleted list, with one field per line.
            """,
            model="gpt-4o-mini"
        )

        # Create a prompt for the agent
        prompt = f"""Analyze this document template and identify all fields that need to be filled.

        Template Name: {template_name}

        Template Content:
        {template_content[:4000]}  # Limit content length

        Return a list of field names, normalized to lowercase with underscores instead of spaces.
        """

        # Run the agent using the Runner class (static method)
        result = await Runner.run(field_detection_agent, input=prompt, context=context)

        # Extract the field names from the agent's response
        # Look for a list or JSON structure in the response
        import json
        detected_fields = []

        # Get the final output from the result
        # The final_output might be a string, dict, or other type
        final_output = result.final_output

        # Convert the final output to a string for pattern matching
        response_text = ""
        if isinstance(final_output, str):
            response_text = final_output
        elif hasattr(final_output, 'markdown_response'):
            response_text = final_output.markdown_response
        elif isinstance(final_output, dict):
            response_text = str(final_output)
        else:
            # Try to convert to string as a fallback
            response_text = str(final_output)

        # Try to find a JSON array in the response
        json_pattern = re.compile(r'\[.*\]')
        json_match = json_pattern.search(response_text)
        if json_match:
            try:
                detected_fields = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # If no JSON found, look for field names in the text
        if not detected_fields:
            # Look for patterns like "- field_name" or "* field_name" in the agent's response
            field_pattern = re.compile(r'[\-\*]\s+([a-z0-9_]+)')
            matches = field_pattern.findall(response_text)
            if matches:
                detected_fields = matches

        # Normalize field names
        normalized_fields = [
            field.lower().replace(" ", "_") if isinstance(field, str) else str(field)
            for field in detected_fields
        ]

        # Remove duplicates while preserving order
        unique_fields = []
        for field in normalized_fields:
            if field not in unique_fields:
                unique_fields.append(field)

        logger.info(f"[Agent Result] Detected fields from template: {unique_fields}")
        return unique_fields
    except Exception as e:
        logger.error(f"[Agent Error] Field detection failed: {e}", exc_info=True)
        # Fall back to regex-based detection if agent-based detection fails
        from app import detect_required_fields_from_template
        logger.info("Falling back to regex-based field detection")
        return detect_required_fields_from_template(template_content, template_name)

# --- Import the DocumentSamplingAgent ---
try:
    from agents.document_sampling_agent import sample_document
    SAMPLING_AGENT_AVAILABLE = True
except ImportError:
    logger.warning("DocumentSamplingAgent not available. Large documents may not be processed efficiently.")
    SAMPLING_AGENT_AVAILABLE = False

# --- Define the DocumentAnalyzerAgent ---
document_analyzer_agent = Agent(
    name="DocumentAnalyzerAgent",
    instructions="""You are a specialized document analyzer agent that understands document structure and extracts information using semantic understanding rather than pattern matching.

You have five main capabilities:
1. Analyzing document structure to identify sections, layout, and document type
2. Extracting specific fields from documents based on semantic understanding
3. Detecting required fields from templates
4. Analyzing documents for workflow integration with standardized output
5. Intelligently sampling large documents to extract the most representative content

When analyzing documents:
- Focus on understanding the document's semantic structure, not just looking for patterns
- Consider the document type when extracting information
- Provide confidence scores for extracted values
- Suggest alternative values when appropriate
- For large documents, use intelligent sampling to focus on the most important content

When detecting fields from templates:
- Look for explicit placeholders in various formats
- Identify implied fields based on context
- Consider standard fields expected in the document type

Use the appropriate tool based on the task:
- analyze_document_structure: To understand document organization (legacy format)
- analyze_document_for_workflow: To analyze documents with standardized output for workflow integration
- extract_fields_from_document: To extract specific fields
- detect_fields_from_template: To identify fields in a template
- sample_document: To intelligently sample large documents

Always return structured data according to the tool's output format.
""",
    tools=[
        analyze_document_structure,
        analyze_document_for_workflow,
        extract_fields_from_document,
        detect_fields_from_template,
        sample_document if SAMPLING_AGENT_AVAILABLE else None
    ],
    model="gpt-4o-mini"  # Use appropriate model
)