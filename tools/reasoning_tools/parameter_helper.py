"""
Parameter Helper Tool

This module provides tools for helping agents determine the best parameters for tool calls.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ParameterSuggestion(BaseModel):
    """Parameter suggestion for a tool call."""
    parameter_name: str
    suggested_value: Any
    confidence: float
    reasoning: str

class ParameterHelperResult(BaseModel):
    """Result of a parameter helper operation."""
    success: bool
    message: str
    tool_name: str
    parameter_suggestions: List[ParameterSuggestion]
    context_analysis: Optional[Dict[str, Any]] = None

@function_tool(strict_mode=False)
async def suggest_tool_parameters(ctx: RunContextWrapper, tool_name: str, query: str) -> ParameterHelperResult:
    """
    Analyze the context and suggest parameters for a tool call.
    
    Args:
        tool_name: Name of the tool to suggest parameters for
        query: The user query or task description
        
    Returns:
        Parameter suggestions for the tool
    """
    logger.info(f"[Tool Call] suggest_tool_parameters: tool_name='{tool_name}', query='{query}'")
    
    try:
        # Get the workflow context
        workflow_context = ctx.context or {}
        
        # Get tool registry
        from tools.registry import tool_registry
        
        # Get the tool definition
        tool_def = None
        for tool in tool_registry.tools:
            if tool.name == tool_name:
                tool_def = tool
                break
                
        if not tool_def:
            return ParameterHelperResult(
                success=False,
                message=f"Tool '{tool_name}' not found in registry",
                tool_name=tool_name,
                parameter_suggestions=[]
            )
            
        # Extract parameter information from the tool
        parameter_info = _extract_parameter_info(tool_def)
        
        # Analyze the context to suggest parameters
        parameter_suggestions, context_analysis = _analyze_context_for_parameters(
            tool_name, 
            parameter_info, 
            query, 
            workflow_context
        )
        
        return ParameterHelperResult(
            success=True,
            message=f"Generated parameter suggestions for {tool_name}",
            tool_name=tool_name,
            parameter_suggestions=parameter_suggestions,
            context_analysis=context_analysis
        )
    except Exception as e:
        logger.error(f"Error suggesting parameters for tool {tool_name}: {e}")
        return ParameterHelperResult(
            success=False,
            message=f"Error suggesting parameters: {str(e)}",
            tool_name=tool_name,
            parameter_suggestions=[]
        )

def _extract_parameter_info(tool_def: Any) -> Dict[str, Any]:
    """
    Extract parameter information from a tool definition.
    
    Args:
        tool_def: Tool definition
        
    Returns:
        Parameter information
    """
    parameter_info = {}
    
    # Try to extract parameter info from the tool's schema
    try:
        if hasattr(tool_def, 'schema') and callable(getattr(tool_def, 'schema')):
            schema = tool_def.schema()
            if 'parameters' in schema and 'properties' in schema['parameters']:
                for param_name, param_def in schema['parameters']['properties'].items():
                    parameter_info[param_name] = {
                        'type': param_def.get('type', 'string'),
                        'description': param_def.get('description', ''),
                        'required': param_name in schema['parameters'].get('required', [])
                    }
    except Exception as e:
        logger.warning(f"Error extracting parameter info from schema: {e}")
    
    # If we couldn't extract from schema, try to infer from function signature
    if not parameter_info and hasattr(tool_def, 'function'):
        try:
            import inspect
            sig = inspect.signature(tool_def.function)
            for param_name, param in sig.parameters.items():
                if param_name != 'ctx':  # Skip the context parameter
                    parameter_info[param_name] = {
                        'type': 'unknown',
                        'description': '',
                        'required': param.default == inspect.Parameter.empty
                    }
        except Exception as e:
            logger.warning(f"Error extracting parameter info from function signature: {e}")
    
    return parameter_info

def _analyze_context_for_parameters(
    tool_name: str, 
    parameter_info: Dict[str, Any], 
    query: str, 
    workflow_context: Dict[str, Any]
) -> tuple[List[ParameterSuggestion], Dict[str, Any]]:
    """
    Analyze the context to suggest parameters for a tool call.
    
    Args:
        tool_name: Name of the tool
        parameter_info: Parameter information
        query: The user query or task description
        workflow_context: The workflow context
        
    Returns:
        Parameter suggestions and context analysis
    """
    parameter_suggestions = []
    context_analysis = {
        "query_analysis": {},
        "context_elements_used": [],
        "reasoning_steps": []
    }
    
    # Extract relevant information from the context
    relevant_context = _extract_relevant_context(tool_name, parameter_info, workflow_context)
    context_analysis["context_elements_used"] = list(relevant_context.keys())
    
    # Add reasoning step
    context_analysis["reasoning_steps"].append(
        f"Extracted relevant context for {tool_name}: {list(relevant_context.keys())}"
    )
    
    # Analyze the query
    query_entities = _extract_entities_from_query(query)
    context_analysis["query_analysis"] = {
        "entities": query_entities,
        "query_type": _determine_query_type(query)
    }
    
    # Add reasoning step
    context_analysis["reasoning_steps"].append(
        f"Analyzed query and extracted entities: {query_entities}"
    )
    
    # Generate parameter suggestions based on the tool
    if tool_name == "get_kb_document_content":
        suggestions = _suggest_kb_document_parameters(parameter_info, query, query_entities, relevant_context)
        parameter_suggestions.extend(suggestions)
        
    elif tool_name == "retrieve_template_content":
        suggestions = _suggest_template_parameters(parameter_info, query, query_entities, relevant_context)
        parameter_suggestions.extend(suggestions)
        
    elif tool_name == "process_temp_file":
        suggestions = _suggest_temp_file_parameters(parameter_info, query, query_entities, relevant_context)
        parameter_suggestions.extend(suggestions)
        
    elif tool_name == "extract_data_for_template":
        suggestions = _suggest_extraction_parameters(parameter_info, query, query_entities, relevant_context)
        parameter_suggestions.extend(suggestions)
        
    elif tool_name == "generate_docx_from_markdown":
        suggestions = _suggest_docx_parameters(parameter_info, query, query_entities, relevant_context)
        parameter_suggestions.extend(suggestions)
        
    # Add reasoning step
    context_analysis["reasoning_steps"].append(
        f"Generated {len(parameter_suggestions)} parameter suggestions for {tool_name}"
    )
    
    return parameter_suggestions, context_analysis

def _extract_relevant_context(
    tool_name: str, 
    parameter_info: Dict[str, Any], 
    workflow_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract relevant information from the context for a specific tool.
    
    Args:
        tool_name: Name of the tool
        parameter_info: Parameter information
        workflow_context: The workflow context
        
    Returns:
        Relevant context information
    """
    relevant_context = {}
    
    # Extract common relevant context
    if "current_query" in workflow_context:
        relevant_context["current_query"] = workflow_context["current_query"]
        
    if "chat_id" in workflow_context:
        relevant_context["chat_id"] = workflow_context["chat_id"]
        
    if "history" in workflow_context:
        # Only include the last few messages to avoid context bloat
        history = workflow_context["history"]
        if isinstance(history, list) and len(history) > 0:
            relevant_context["recent_history"] = history[-3:]  # Last 3 messages
    
    # Extract tool-specific relevant context
    if tool_name == "get_kb_document_content":
        # Include knowledge base information
        if "kb_files" in workflow_context:
            relevant_context["kb_files"] = workflow_context["kb_files"]
            
        if "last_kb_query" in workflow_context:
            relevant_context["last_kb_query"] = workflow_context["last_kb_query"]
            
    elif tool_name == "retrieve_template_content":
        # Include template information
        if "template_to_populate" in workflow_context:
            relevant_context["template_to_populate"] = workflow_context["template_to_populate"]
            
        if "available_templates" in workflow_context:
            relevant_context["available_templates"] = workflow_context["available_templates"]
            
    elif tool_name == "process_temp_file":
        # Include temporary file information
        if "temp_files_info" in workflow_context:
            relevant_context["temp_files_info"] = workflow_context["temp_files_info"]
            
    elif tool_name == "extract_data_for_template":
        # Include template and context information
        if "template_content" in workflow_context:
            relevant_context["template_content"] = workflow_context["template_content"]
            
        if "kb_content" in workflow_context:
            relevant_context["kb_content"] = workflow_context["kb_content"]
            
        if "temp_file_content" in workflow_context:
            relevant_context["temp_file_content"] = workflow_context["temp_file_content"]
            
    elif tool_name == "generate_docx_from_markdown":
        # Include markdown content
        if "markdown_content" in workflow_context:
            relevant_context["markdown_content"] = workflow_context["markdown_content"]
            
        if "populated_template" in workflow_context:
            relevant_context["populated_template"] = workflow_context["populated_template"]
    
    return relevant_context

def _extract_entities_from_query(query: str) -> Dict[str, Any]:
    """
    Extract entities from a query.
    
    Args:
        query: The query to extract entities from
        
    Returns:
        Extracted entities
    """
    entities = {
        "document_references": [],
        "template_references": [],
        "person_names": [],
        "date_references": [],
        "topic_references": []
    }
    
    # Extract document references
    doc_patterns = [
        r'document[s]?\s+(?:called|named|titled)\s+["\']?([^"\'.,;:!?]+)["\']?',
        r'["\']([^"\']+\.(pdf|docx|txt|md))["\']',
        r'([a-zA-Z0-9_-]+\.(pdf|docx|txt|md))',
        r'the\s+([a-zA-Z0-9_-]+)\s+document'
    ]
    
    for pattern in doc_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    entities["document_references"].append(match[0])
                else:
                    entities["document_references"].append(match)
    
    # Extract template references
    template_patterns = [
        r'template[s]?\s+(?:called|named|titled)\s+["\']?([^"\'.,;:!?]+)["\']?',
        r'["\']([^"\']+template)["\']',
        r'([a-zA-Z0-9_-]+)\s+template',
        r'template\s+for\s+([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in template_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            entities["template_references"].extend(matches)
    
    # Extract person names (simplified)
    name_patterns = [
        r'for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|should|needs)'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, query)
        if matches:
            entities["person_names"].extend(matches)
    
    # Extract date references (simplified)
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, query)
        if matches:
            entities["date_references"].extend(matches)
    
    # Extract topic references
    topic_patterns = [
        r'about\s+([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+){0,3})',
        r'related\s+to\s+([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+){0,3})',
        r'regarding\s+([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+){0,3})'
    ]
    
    for pattern in topic_patterns:
        matches = re.findall(pattern, query)
        if matches:
            entities["topic_references"].extend(matches)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def _determine_query_type(query: str) -> str:
    """
    Determine the type of query.
    
    Args:
        query: The query to analyze
        
    Returns:
        Query type
    """
    query_lower = query.lower()
    
    if any(x in query_lower for x in ["generate", "create", "make", "produce", "draft"]):
        return "generation"
    elif any(x in query_lower for x in ["compare", "analyze", "evaluate", "assess"]):
        return "analysis"
    elif any(x in query_lower for x in ["find", "search", "look for", "get", "retrieve"]):
        return "retrieval"
    elif any(x in query_lower for x in ["extract", "pull", "identify"]):
        return "extraction"
    else:
        return "general"

def _suggest_kb_document_parameters(
    parameter_info: Dict[str, Any],
    query: str,
    query_entities: Dict[str, Any],
    relevant_context: Dict[str, Any]
) -> List[ParameterSuggestion]:
    """
    Suggest parameters for the get_kb_document_content tool.
    
    Args:
        parameter_info: Parameter information
        query: The user query
        query_entities: Entities extracted from the query
        relevant_context: Relevant context information
        
    Returns:
        Parameter suggestions
    """
    suggestions = []
    
    # Suggest query parameter
    query_suggestion = None
    
    # Check if there are document references in the query
    if query_entities["document_references"]:
        doc_ref = query_entities["document_references"][0]
        query_suggestion = ParameterSuggestion(
            parameter_name="query",
            suggested_value=doc_ref,
            confidence=0.8,
            reasoning=f"Document reference '{doc_ref}' found in the query"
        )
    # Check if there are topic references in the query
    elif query_entities["topic_references"]:
        topic_ref = query_entities["topic_references"][0]
        query_suggestion = ParameterSuggestion(
            parameter_name="query",
            suggested_value=topic_ref,
            confidence=0.7,
            reasoning=f"Topic reference '{topic_ref}' found in the query"
        )
    # Use the current query as a fallback
    else:
        current_query = relevant_context.get("current_query", "")
        if current_query:
            # Remove common prefixes that aren't relevant to the KB search
            cleaned_query = re.sub(r'^(find|get|retrieve|tell me about|what does|how does|can you|please)\s+', '', current_query, flags=re.IGNORECASE)
            query_suggestion = ParameterSuggestion(
                parameter_name="query",
                suggested_value=cleaned_query,
                confidence=0.5,
                reasoning="Using cleaned version of the current query as no specific document or topic reference was found"
            )
    
    if query_suggestion:
        suggestions.append(query_suggestion)
    
    # Suggest document_type parameter
    doc_type_suggestion = None
    
    # Check if there are document references with extensions
    for doc_ref in query_entities["document_references"]:
        if "." in doc_ref:
            ext = doc_ref.split(".")[-1].lower()
            if ext in ["pdf", "docx", "txt", "md"]:
                doc_type_suggestion = ParameterSuggestion(
                    parameter_name="document_type",
                    suggested_value=ext,
                    confidence=0.9,
                    reasoning=f"Document extension '.{ext}' found in reference '{doc_ref}'"
                )
                break
    
    # If no document type found, suggest a default
    if not doc_type_suggestion:
        doc_type_suggestion = ParameterSuggestion(
            parameter_name="document_type",
            suggested_value="all",
            confidence=0.6,
            reasoning="No specific document type found, using 'all' as default"
        )
    
    suggestions.append(doc_type_suggestion)
    
    return suggestions

def _suggest_template_parameters(
    parameter_info: Dict[str, Any],
    query: str,
    query_entities: Dict[str, Any],
    relevant_context: Dict[str, Any]
) -> List[ParameterSuggestion]:
    """
    Suggest parameters for the retrieve_template_content tool.
    
    Args:
        parameter_info: Parameter information
        query: The user query
        query_entities: Entities extracted from the query
        relevant_context: Relevant context information
        
    Returns:
        Parameter suggestions
    """
    suggestions = []
    
    # Suggest template_name parameter
    template_name_suggestion = None
    
    # Check if there's a template to populate in the context
    if "template_to_populate" in relevant_context:
        template_name = relevant_context["template_to_populate"]
        if template_name:
            template_name_suggestion = ParameterSuggestion(
                parameter_name="template_name",
                suggested_value=template_name,
                confidence=0.9,
                reasoning=f"Template name '{template_name}' found in context"
            )
    
    # Check if there are template references in the query
    if not template_name_suggestion and query_entities["template_references"]:
        template_ref = query_entities["template_references"][0]
        template_name_suggestion = ParameterSuggestion(
            parameter_name="template_name",
            suggested_value=template_ref,
            confidence=0.8,
            reasoning=f"Template reference '{template_ref}' found in the query"
        )
    
    # Check if there are document references in the query that might be templates
    if not template_name_suggestion and query_entities["document_references"]:
        doc_ref = query_entities["document_references"][0]
        if doc_ref.lower().endswith((".md", ".txt")):
            template_name_suggestion = ParameterSuggestion(
                parameter_name="template_name",
                suggested_value=doc_ref,
                confidence=0.7,
                reasoning=f"Document reference '{doc_ref}' found in the query and has a template-compatible extension"
            )
    
    if template_name_suggestion:
        suggestions.append(template_name_suggestion)
    
    return suggestions

def _suggest_temp_file_parameters(
    parameter_info: Dict[str, Any],
    query: str,
    query_entities: Dict[str, Any],
    relevant_context: Dict[str, Any]
) -> List[ParameterSuggestion]:
    """
    Suggest parameters for the process_temp_file tool.
    
    Args:
        parameter_info: Parameter information
        query: The user query
        query_entities: Entities extracted from the query
        relevant_context: Relevant context information
        
    Returns:
        Parameter suggestions
    """
    suggestions = []
    
    # Suggest file_id parameter
    file_id_suggestion = None
    
    # Check if there are temporary files in the context
    if "temp_files_info" in relevant_context:
        temp_files = relevant_context["temp_files_info"]
        if temp_files and isinstance(temp_files, list) and len(temp_files) > 0:
            # If there's only one temp file, use it
            if len(temp_files) == 1:
                file_info = temp_files[0]
                file_id = file_info.get("id") or file_info.get("file_id")
                if file_id:
                    file_id_suggestion = ParameterSuggestion(
                        parameter_name="file_id",
                        suggested_value=file_id,
                        confidence=0.9,
                        reasoning=f"Only one temporary file found with ID '{file_id}'"
                    )
            # If there are multiple temp files, try to find the most relevant one
            else:
                # Check if any of the document references in the query match a temp file name
                if query_entities["document_references"]:
                    for doc_ref in query_entities["document_references"]:
                        for file_info in temp_files:
                            filename = file_info.get("filename") or file_info.get("name") or ""
                            if doc_ref.lower() in filename.lower():
                                file_id = file_info.get("id") or file_info.get("file_id")
                                if file_id:
                                    file_id_suggestion = ParameterSuggestion(
                                        parameter_name="file_id",
                                        suggested_value=file_id,
                                        confidence=0.8,
                                        reasoning=f"Document reference '{doc_ref}' matches temporary file '{filename}' with ID '{file_id}'"
                                    )
                                    break
                        if file_id_suggestion:
                            break
                
                # If no match found, use the most recently uploaded file
                if not file_id_suggestion:
                    # Assume the last file in the list is the most recent
                    file_info = temp_files[-1]
                    file_id = file_info.get("id") or file_info.get("file_id")
                    if file_id:
                        file_id_suggestion = ParameterSuggestion(
                            parameter_name="file_id",
                            suggested_value=file_id,
                            confidence=0.6,
                            reasoning=f"Using most recently uploaded temporary file with ID '{file_id}'"
                        )
    
    if file_id_suggestion:
        suggestions.append(file_id_suggestion)
    
    return suggestions

def _suggest_extraction_parameters(
    parameter_info: Dict[str, Any],
    query: str,
    query_entities: Dict[str, Any],
    relevant_context: Dict[str, Any]
) -> List[ParameterSuggestion]:
    """
    Suggest parameters for the extract_data_for_template tool.
    
    Args:
        parameter_info: Parameter information
        query: The user query
        query_entities: Entities extracted from the query
        relevant_context: Relevant context information
        
    Returns:
        Parameter suggestions
    """
    suggestions = []
    
    # Suggest context_sources parameter
    context_sources = []
    
    # Add template content if available
    if "template_content" in relevant_context:
        context_sources.append(relevant_context["template_content"])
    
    # Add KB content if available
    if "kb_content" in relevant_context:
        context_sources.append(relevant_context["kb_content"])
    
    # Add temp file content if available
    if "temp_file_content" in relevant_context:
        context_sources.append(relevant_context["temp_file_content"])
    
    if context_sources:
        suggestions.append(ParameterSuggestion(
            parameter_name="context_sources",
            suggested_value=context_sources,
            confidence=0.8,
            reasoning=f"Using {len(context_sources)} available context sources from workflow context"
        ))
    
    # Suggest required_fields parameter
    required_fields = []
    
    # Extract potential fields from template content
    if "template_content" in relevant_context:
        template_content = relevant_context["template_content"]
        if isinstance(template_content, str):
            # Look for placeholders in the template
            placeholder_matches = re.findall(r'\{\{([^}]+)\}\}', template_content)
            if placeholder_matches:
                required_fields.extend(placeholder_matches)
            
            # Look for field labels in the template
            field_label_matches = re.findall(r'^([A-Z][a-zA-Z_]+):', template_content, re.MULTILINE)
            if field_label_matches:
                required_fields.extend(field_label_matches)
    
    # Add common fields based on query type
    query_lower = query.lower()
    if "contract" in query_lower or "agreement" in query_lower:
        common_fields = ["employee_name", "employer_name", "start_date", "salary", "position"]
        for field in common_fields:
            if field not in required_fields:
                required_fields.append(field)
    elif "invoice" in query_lower:
        common_fields = ["client_name", "invoice_date", "due_date", "items", "total_amount"]
        for field in common_fields:
            if field not in required_fields:
                required_fields.append(field)
    
    # Add person names from query as potential fields
    if query_entities["person_names"]:
        for name in query_entities["person_names"]:
            required_fields.append(f"{name.lower().replace(' ', '_')}_details")
    
    if required_fields:
        suggestions.append(ParameterSuggestion(
            parameter_name="required_fields",
            suggested_value=required_fields,
            confidence=0.7,
            reasoning=f"Extracted {len(required_fields)} required fields from template content and query analysis"
        ))
    
    return suggestions

def _suggest_docx_parameters(
    parameter_info: Dict[str, Any],
    query: str,
    query_entities: Dict[str, Any],
    relevant_context: Dict[str, Any]
) -> List[ParameterSuggestion]:
    """
    Suggest parameters for the generate_docx_from_markdown tool.
    
    Args:
        parameter_info: Parameter information
        query: The user query
        query_entities: Entities extracted from the query
        relevant_context: Relevant context information
        
    Returns:
        Parameter suggestions
    """
    suggestions = []
    
    # Suggest markdown_content parameter
    if "populated_template" in relevant_context:
        suggestions.append(ParameterSuggestion(
            parameter_name="markdown_content",
            suggested_value=relevant_context["populated_template"],
            confidence=0.9,
            reasoning="Using populated template content from workflow context"
        ))
    elif "markdown_content" in relevant_context:
        suggestions.append(ParameterSuggestion(
            parameter_name="markdown_content",
            suggested_value=relevant_context["markdown_content"],
            confidence=0.9,
            reasoning="Using markdown content from workflow context"
        ))
    
    # Suggest output_filename parameter
    output_filename = None
    
    # Check if there are document references in the query
    if query_entities["document_references"]:
        doc_ref = query_entities["document_references"][0]
        # Convert to .docx if it's not already
        if not doc_ref.lower().endswith(".docx"):
            base_name = doc_ref.split(".")[0] if "." in doc_ref else doc_ref
            output_filename = f"{base_name}.docx"
        else:
            output_filename = doc_ref
    
    # Check if there are person names in the query
    elif query_entities["person_names"]:
        name = query_entities["person_names"][0]
        # Determine document type from query
        doc_type = "document"
        query_lower = query.lower()
        if "contract" in query_lower:
            doc_type = "contract"
        elif "invoice" in query_lower:
            doc_type = "invoice"
        elif "report" in query_lower:
            doc_type = "report"
        elif "letter" in query_lower:
            doc_type = "letter"
        
        output_filename = f"{name.replace(' ', '_')}_{doc_type}.docx"
    
    # Default filename
    else:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"generated_document_{timestamp}.docx"
    
    if output_filename:
        suggestions.append(ParameterSuggestion(
            parameter_name="output_filename",
            suggested_value=output_filename,
            confidence=0.8,
            reasoning=f"Generated output filename '{output_filename}' based on query analysis"
        ))
    
    return suggestions
