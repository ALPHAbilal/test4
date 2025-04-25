"""
Outcome Analyzer Tool

This module provides tools for analyzing the outcomes of tool calls and agent actions.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from agents import function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ActionRecommendation(BaseModel):
    """Recommendation for the next action."""
    action_type: str  # "call_agent", "call_tool", "return_to_user", "needs_input", "error"
    details: Dict[str, Any]
    confidence: float
    reasoning: str

class OutcomeAnalysisResult(BaseModel):
    """Result of an outcome analysis operation."""
    success: bool
    message: str
    outcome_type: str  # "success", "partial_success", "failure", "error", "ambiguous"
    outcome_details: Dict[str, Any]
    recommendations: List[ActionRecommendation]
    analysis_summary: str

@function_tool(strict_mode=False)
async def analyze_tool_outcome(ctx: RunContextWrapper, tool_name: str, outcome: Any) -> OutcomeAnalysisResult:
    """
    Analyze the outcome of a tool call and suggest next actions.
    
    Args:
        tool_name: Name of the tool that was called
        outcome: The outcome of the tool call
        
    Returns:
        Analysis of the outcome and recommendations for next actions
    """
    logger.info(f"[Tool Call] analyze_tool_outcome: tool_name='{tool_name}'")
    
    try:
        # Get the workflow context
        workflow_context = ctx.context or {}
        
        # Convert outcome to string if it's not already
        outcome_str = json.dumps(outcome) if not isinstance(outcome, str) else outcome
        
        # Analyze the outcome
        outcome_type, outcome_details = _analyze_outcome(tool_name, outcome, outcome_str)
        
        # Generate recommendations
        recommendations = _generate_recommendations(tool_name, outcome_type, outcome_details, workflow_context)
        
        # Generate summary
        summary = _generate_analysis_summary(tool_name, outcome_type, outcome_details, recommendations)
        
        return OutcomeAnalysisResult(
            success=True,
            message=f"Analyzed outcome of {tool_name}",
            outcome_type=outcome_type,
            outcome_details=outcome_details,
            recommendations=recommendations,
            analysis_summary=summary
        )
    except Exception as e:
        logger.error(f"Error analyzing outcome for tool {tool_name}: {e}")
        return OutcomeAnalysisResult(
            success=False,
            message=f"Error analyzing outcome: {str(e)}",
            outcome_type="error",
            outcome_details={"error": str(e)},
            recommendations=[],
            analysis_summary=f"Error analyzing outcome: {str(e)}"
        )

@function_tool(strict_mode=False)
async def analyze_agent_outcome(ctx: RunContextWrapper, agent_name: str, outcome: Any) -> OutcomeAnalysisResult:
    """
    Analyze the outcome of an agent call and suggest next actions.
    
    Args:
        agent_name: Name of the agent that was called
        outcome: The outcome of the agent call
        
    Returns:
        Analysis of the outcome and recommendations for next actions
    """
    logger.info(f"[Tool Call] analyze_agent_outcome: agent_name='{agent_name}'")
    
    try:
        # Get the workflow context
        workflow_context = ctx.context or {}
        
        # Convert outcome to string if it's not already
        outcome_str = json.dumps(outcome) if not isinstance(outcome, str) else outcome
        
        # Analyze the outcome
        outcome_type, outcome_details = _analyze_agent_outcome(agent_name, outcome, outcome_str)
        
        # Generate recommendations
        recommendations = _generate_agent_recommendations(agent_name, outcome_type, outcome_details, workflow_context)
        
        # Generate summary
        summary = _generate_analysis_summary(agent_name, outcome_type, outcome_details, recommendations)
        
        return OutcomeAnalysisResult(
            success=True,
            message=f"Analyzed outcome of {agent_name}",
            outcome_type=outcome_type,
            outcome_details=outcome_details,
            recommendations=recommendations,
            analysis_summary=summary
        )
    except Exception as e:
        logger.error(f"Error analyzing outcome for agent {agent_name}: {e}")
        return OutcomeAnalysisResult(
            success=False,
            message=f"Error analyzing outcome: {str(e)}",
            outcome_type="error",
            outcome_details={"error": str(e)},
            recommendations=[],
            analysis_summary=f"Error analyzing outcome: {str(e)}"
        )

def _analyze_outcome(tool_name: str, outcome: Any, outcome_str: str) -> tuple[str, Dict[str, Any]]:
    """
    Analyze the outcome of a tool call.
    
    Args:
        tool_name: Name of the tool
        outcome: The outcome object
        outcome_str: String representation of the outcome
        
    Returns:
        Outcome type and details
    """
    outcome_type = "ambiguous"
    outcome_details = {}
    
    # Check for common error patterns
    error_patterns = [
        r'error',
        r'exception',
        r'failed',
        r'not found',
        r'invalid',
        r'missing',
        r'unauthorized',
        r'forbidden',
        r'timeout',
        r'unavailable'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, outcome_str, re.IGNORECASE):
            outcome_type = "failure"
            outcome_details["error_pattern"] = pattern
            break
    
    # Check for empty or null results
    if outcome is None or outcome == "" or outcome == {} or outcome == []:
        outcome_type = "failure"
        outcome_details["empty_result"] = True
    
    # Check for success indicators
    success_patterns = [
        r'success',
        r'completed',
        r'found',
        r'retrieved',
        r'generated'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, outcome_str, re.IGNORECASE):
            outcome_type = "success"
            outcome_details["success_pattern"] = pattern
            break
    
    # Tool-specific analysis
    if tool_name == "get_kb_document_content":
        if isinstance(outcome, dict) and "content" in outcome and outcome["content"]:
            outcome_type = "success"
            outcome_details["content_length"] = len(outcome["content"])
            outcome_details["source"] = outcome.get("source", "unknown")
        elif isinstance(outcome, dict) and "error" in outcome:
            outcome_type = "failure"
            outcome_details["error_message"] = outcome["error"]
    
    elif tool_name == "retrieve_template_content":
        if isinstance(outcome, dict) and "content" in outcome and outcome["content"]:
            outcome_type = "success"
            outcome_details["content_length"] = len(outcome["content"])
            outcome_details["template_name"] = outcome.get("source_filename", "unknown")
        elif isinstance(outcome, dict) and "error" in outcome:
            outcome_type = "failure"
            outcome_details["error_message"] = outcome["error"]
    
    elif tool_name == "process_temp_file":
        if isinstance(outcome, dict) and "content" in outcome and outcome["content"]:
            outcome_type = "success"
            outcome_details["content_length"] = len(outcome["content"])
            outcome_details["file_name"] = outcome.get("file_name", "unknown")
        elif isinstance(outcome, dict) and "error" in outcome:
            outcome_type = "failure"
            outcome_details["error_message"] = outcome["error"]
    
    elif tool_name == "extract_data_for_template":
        if isinstance(outcome, dict) and "extracted_fields" in outcome:
            extracted_fields = outcome["extracted_fields"]
            missing_fields = outcome.get("missing_fields", [])
            
            if extracted_fields and not missing_fields:
                outcome_type = "success"
                outcome_details["extracted_field_count"] = len(extracted_fields)
            elif extracted_fields and missing_fields:
                outcome_type = "partial_success"
                outcome_details["extracted_field_count"] = len(extracted_fields)
                outcome_details["missing_field_count"] = len(missing_fields)
                outcome_details["missing_fields"] = missing_fields
            else:
                outcome_type = "failure"
                outcome_details["error_message"] = "No fields extracted"
    
    elif tool_name == "generate_docx_from_markdown":
        if isinstance(outcome, dict) and "file_path" in outcome:
            outcome_type = "success"
            outcome_details["file_path"] = outcome["file_path"]
        elif isinstance(outcome, dict) and "error" in outcome:
            outcome_type = "failure"
            outcome_details["error_message"] = outcome["error"]
    
    return outcome_type, outcome_details

def _analyze_agent_outcome(agent_name: str, outcome: Any, outcome_str: str) -> tuple[str, Dict[str, Any]]:
    """
    Analyze the outcome of an agent call.
    
    Args:
        agent_name: Name of the agent
        outcome: The outcome object
        outcome_str: String representation of the outcome
        
    Returns:
        Outcome type and details
    """
    outcome_type = "ambiguous"
    outcome_details = {}
    
    # Check for common error patterns
    error_patterns = [
        r'error',
        r'exception',
        r'failed',
        r'not found',
        r'invalid',
        r'missing',
        r'unauthorized',
        r'forbidden',
        r'timeout',
        r'unavailable'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, outcome_str, re.IGNORECASE):
            outcome_type = "failure"
            outcome_details["error_pattern"] = pattern
            break
    
    # Check for empty or null results
    if outcome is None or outcome == "" or outcome == {} or outcome == []:
        outcome_type = "failure"
        outcome_details["empty_result"] = True
    
    # Check for success indicators
    success_patterns = [
        r'success',
        r'completed',
        r'found',
        r'retrieved',
        r'generated'
    ]
    
    for pattern in success_patterns:
        if re.search(pattern, outcome_str, re.IGNORECASE):
            outcome_type = "success"
            outcome_details["success_pattern"] = pattern
            break
    
    # Agent-specific analysis
    if agent_name == "DataGatheringAgentMinimal":
        if isinstance(outcome, dict) and "data" in outcome and outcome["data"]:
            outcome_type = "success"
            outcome_details["data_length"] = len(json.dumps(outcome["data"]))
            outcome_details["data_type"] = type(outcome["data"]).__name__
        elif isinstance(outcome, dict) and "error" in outcome:
            outcome_type = "failure"
            outcome_details["error_message"] = outcome["error"]
    
    elif agent_name == "QueryAnalyzerAgent":
        if isinstance(outcome, dict) and "intent" in outcome:
            outcome_type = "success"
            outcome_details["intent"] = outcome["intent"]
            outcome_details["confidence"] = outcome.get("confidence", 0)
        elif isinstance(outcome, dict) and "error" in outcome:
            outcome_type = "failure"
            outcome_details["error_message"] = outcome["error"]
    
    return outcome_type, outcome_details

def _generate_recommendations(
    tool_name: str, 
    outcome_type: str, 
    outcome_details: Dict[str, Any], 
    workflow_context: Dict[str, Any]
) -> List[ActionRecommendation]:
    """
    Generate recommendations based on the outcome analysis.
    
    Args:
        tool_name: Name of the tool
        outcome_type: Type of outcome
        outcome_details: Details of the outcome
        workflow_context: The workflow context
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Handle success outcomes
    if outcome_type == "success":
        if tool_name == "get_kb_document_content":
            # Recommend returning to user if this was a direct KB query
            recommendations.append(ActionRecommendation(
                action_type="return_to_user",
                details={
                    "final_response": f"Here is the information I found in the knowledge base about your query:\n\n{outcome_details.get('content', 'Content not available')}"
                },
                confidence=0.8,
                reasoning="Successfully retrieved content from knowledge base"
            ))
            
            # Recommend calling an agent to analyze the content
            recommendations.append(ActionRecommendation(
                action_type="call_agent",
                details={
                    "agent_name": "FinalSynthesizerAgent",
                    "input": {
                        "query": workflow_context.get("current_query", ""),
                        "kb_content": outcome_details.get("content", ""),
                        "source": outcome_details.get("source", "")
                    }
                },
                confidence=0.7,
                reasoning="Retrieved content may need further analysis or synthesis"
            ))
        
        elif tool_name == "retrieve_template_content":
            # Recommend extracting data for the template
            recommendations.append(ActionRecommendation(
                action_type="call_tool",
                details={
                    "tool_name": "extract_data_for_template",
                    "parameters": {
                        "context_sources": [outcome_details.get("content", "")],
                        "required_fields": []  # This would need to be determined from the template
                    }
                },
                confidence=0.8,
                reasoning="Successfully retrieved template content, now need to extract data to populate it"
            ))
        
        elif tool_name == "process_temp_file":
            # Recommend returning the content to the user
            recommendations.append(ActionRecommendation(
                action_type="return_to_user",
                details={
                    "final_response": f"Here is the content of the file:\n\n{outcome_details.get('content', 'Content not available')}"
                },
                confidence=0.7,
                reasoning="Successfully processed temporary file"
            ))
            
            # Recommend analyzing the content
            recommendations.append(ActionRecommendation(
                action_type="call_agent",
                details={
                    "agent_name": "DocumentAnalyzerAgent",
                    "input": {
                        "query": workflow_context.get("current_query", ""),
                        "document_content": outcome_details.get("content", ""),
                        "file_name": outcome_details.get("file_name", "")
                    }
                },
                confidence=0.8,
                reasoning="Temporary file content may need further analysis"
            ))
        
        elif tool_name == "extract_data_for_template":
            # Recommend generating a document from the extracted data
            recommendations.append(ActionRecommendation(
                action_type="call_agent",
                details={
                    "agent_name": "TemplatePopulatorAgent",
                    "input": {
                        "template_content": workflow_context.get("template_content", ""),
                        "extracted_fields": outcome_details.get("extracted_fields", {})
                    }
                },
                confidence=0.9,
                reasoning="Successfully extracted data, now need to populate the template"
            ))
        
        elif tool_name == "generate_docx_from_markdown":
            # Recommend returning to user with the generated document
            recommendations.append(ActionRecommendation(
                action_type="return_to_user",
                details={
                    "final_response": f"I've generated the document for you. You can download it here: {outcome_details.get('file_path', 'File path not available')}"
                },
                confidence=0.9,
                reasoning="Successfully generated document"
            ))
    
    # Handle partial success outcomes
    elif outcome_type == "partial_success":
        if tool_name == "extract_data_for_template":
            # Recommend asking the user for missing fields
            missing_fields = outcome_details.get("missing_fields", [])
            if missing_fields:
                missing_fields_str = ", ".join(missing_fields)
                recommendations.append(ActionRecommendation(
                    action_type="needs_input",
                    details={
                        "prompt": f"I need some additional information to complete the template. Please provide the following: {missing_fields_str}"
                    },
                    confidence=0.8,
                    reasoning=f"Missing {len(missing_fields)} required fields for template"
                ))
            
            # Recommend proceeding with partial data
            recommendations.append(ActionRecommendation(
                action_type="call_agent",
                details={
                    "agent_name": "TemplatePopulatorAgent",
                    "input": {
                        "template_content": workflow_context.get("template_content", ""),
                        "extracted_fields": outcome_details.get("extracted_fields", {}),
                        "missing_fields": outcome_details.get("missing_fields", [])
                    }
                },
                confidence=0.6,
                reasoning="Partially extracted data, can proceed with available fields"
            ))
    
    # Handle failure outcomes
    elif outcome_type == "failure":
        error_message = outcome_details.get("error_message", "Unknown error")
        
        if "not found" in error_message.lower():
            # Recommend asking the user for clarification
            recommendations.append(ActionRecommendation(
                action_type="needs_input",
                details={
                    "prompt": f"I couldn't find what you're looking for. Could you provide more details or try a different query?"
                },
                confidence=0.8,
                reasoning=f"Resource not found: {error_message}"
            ))
        elif "permission" in error_message.lower() or "unauthorized" in error_message.lower():
            # Recommend informing the user about permission issues
            recommendations.append(ActionRecommendation(
                action_type="return_to_user",
                details={
                    "final_response": f"I don't have permission to access the requested resource. {error_message}"
                },
                confidence=0.9,
                reasoning=f"Permission error: {error_message}"
            ))
        else:
            # Generic error handling
            recommendations.append(ActionRecommendation(
                action_type="error",
                details={
                    "message": f"Error when calling {tool_name}: {error_message}"
                },
                confidence=0.9,
                reasoning=f"Tool execution failed: {error_message}"
            ))
            
            # Recommend trying an alternative approach
            if tool_name == "get_kb_document_content":
                recommendations.append(ActionRecommendation(
                    action_type="call_tool",
                    details={
                        "tool_name": "list_knowledge_base_files",
                        "parameters": {}
                    },
                    confidence=0.7,
                    reasoning="Failed to get specific KB content, try listing available files instead"
                ))
            elif tool_name == "retrieve_template_content":
                recommendations.append(ActionRecommendation(
                    action_type="needs_input",
                    details={
                        "prompt": "I couldn't find the template you're looking for. Could you specify which template you'd like to use?"
                    },
                    confidence=0.7,
                    reasoning="Failed to retrieve template, ask for clarification"
                ))
    
    # Handle ambiguous outcomes
    elif outcome_type == "ambiguous":
        # Recommend asking the user for clarification
        recommendations.append(ActionRecommendation(
            action_type="needs_input",
            details={
                "prompt": "I'm not sure how to proceed with your request. Could you provide more details or clarify what you're looking for?"
            },
            confidence=0.7,
            reasoning="Ambiguous outcome, need clarification"
        ))
    
    # Handle error outcomes
    elif outcome_type == "error":
        # Recommend reporting the error
        recommendations.append(ActionRecommendation(
            action_type="error",
            details={
                "message": f"An error occurred when calling {tool_name}: {outcome_details.get('error', 'Unknown error')}"
            },
            confidence=0.9,
            reasoning="Error in tool execution"
        ))
    
    return recommendations

def _generate_agent_recommendations(
    agent_name: str, 
    outcome_type: str, 
    outcome_details: Dict[str, Any], 
    workflow_context: Dict[str, Any]
) -> List[ActionRecommendation]:
    """
    Generate recommendations based on the agent outcome analysis.
    
    Args:
        agent_name: Name of the agent
        outcome_type: Type of outcome
        outcome_details: Details of the outcome
        workflow_context: The workflow context
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Handle success outcomes
    if outcome_type == "success":
        if agent_name == "DataGatheringAgentMinimal":
            # Recommend proceeding with the gathered data
            recommendations.append(ActionRecommendation(
                action_type="call_agent",
                details={
                    "agent_name": "FinalSynthesizerAgent",
                    "input": {
                        "query": workflow_context.get("current_query", ""),
                        "gathered_data": outcome_details.get("data", {})
                    }
                },
                confidence=0.8,
                reasoning="Successfully gathered data, now need to synthesize it"
            ))
        
        elif agent_name == "QueryAnalyzerAgent":
            intent = outcome_details.get("intent", "")
            confidence = outcome_details.get("confidence", 0)
            
            if intent == "kb_query":
                # Recommend calling the KB tool
                recommendations.append(ActionRecommendation(
                    action_type="call_tool",
                    details={
                        "tool_name": "get_kb_document_content",
                        "parameters": {
                            "query": workflow_context.get("current_query", ""),
                            "document_type": "all"
                        }
                    },
                    confidence=confidence,
                    reasoning=f"Query intent is KB query with confidence {confidence}"
                ))
            elif intent == "populate_template":
                # Recommend retrieving the template
                recommendations.append(ActionRecommendation(
                    action_type="call_tool",
                    details={
                        "tool_name": "retrieve_template_content",
                        "parameters": {
                            "template_name": workflow_context.get("template_to_populate", "")
                        }
                    },
                    confidence=confidence,
                    reasoning=f"Query intent is template population with confidence {confidence}"
                ))
    
    # Handle failure outcomes
    elif outcome_type == "failure":
        error_message = outcome_details.get("error_message", "Unknown error")
        
        # Recommend reporting the error
        recommendations.append(ActionRecommendation(
            action_type="error",
            details={
                "message": f"Error when calling {agent_name}: {error_message}"
            },
            confidence=0.9,
            reasoning=f"Agent execution failed: {error_message}"
        ))
        
        # Recommend trying an alternative approach
        if agent_name == "DataGatheringAgentMinimal":
            recommendations.append(ActionRecommendation(
                action_type="call_tool",
                details={
                    "tool_name": "list_knowledge_base_files",
                    "parameters": {}
                },
                confidence=0.7,
                reasoning="Failed to gather data, try listing available files instead"
            ))
    
    # Handle ambiguous outcomes
    elif outcome_type == "ambiguous":
        # Recommend asking the user for clarification
        recommendations.append(ActionRecommendation(
            action_type="needs_input",
            details={
                "prompt": "I'm not sure how to proceed with your request. Could you provide more details or clarify what you're looking for?"
            },
            confidence=0.7,
            reasoning="Ambiguous outcome from agent, need clarification"
        ))
    
    return recommendations

def _generate_analysis_summary(
    name: str, 
    outcome_type: str, 
    outcome_details: Dict[str, Any], 
    recommendations: List[ActionRecommendation]
) -> str:
    """
    Generate a summary of the analysis.
    
    Args:
        name: Name of the tool or agent
        outcome_type: Type of outcome
        outcome_details: Details of the outcome
        recommendations: List of recommendations
        
    Returns:
        Summary of the analysis
    """
    summary_parts = []
    
    # Add outcome type
    summary_parts.append(f"Outcome type: {outcome_type}")
    
    # Add key details
    if outcome_type == "success":
        if "content_length" in outcome_details:
            summary_parts.append(f"Content length: {outcome_details['content_length']} characters")
        if "source" in outcome_details:
            summary_parts.append(f"Source: {outcome_details['source']}")
        if "file_path" in outcome_details:
            summary_parts.append(f"File path: {outcome_details['file_path']}")
    elif outcome_type == "partial_success":
        if "extracted_field_count" in outcome_details:
            summary_parts.append(f"Extracted {outcome_details['extracted_field_count']} fields")
        if "missing_field_count" in outcome_details:
            summary_parts.append(f"Missing {outcome_details['missing_field_count']} fields")
    elif outcome_type == "failure":
        if "error_message" in outcome_details:
            summary_parts.append(f"Error: {outcome_details['error_message']}")
    
    # Add recommendations
    if recommendations:
        summary_parts.append(f"Top recommendation: {recommendations[0].action_type} ({recommendations[0].confidence:.2f} confidence)")
        summary_parts.append(f"Reasoning: {recommendations[0].reasoning}")
    
    return " | ".join(summary_parts)
