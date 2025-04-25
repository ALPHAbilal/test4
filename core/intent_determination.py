"""
Enhanced Intent Determination Module

This module provides sophisticated intent determination capabilities
by leveraging LLM-powered agents rather than regex patterns or hardcoded fallbacks.
"""

import json
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional

# Import DocumentAnalysis model
from data_models import DocumentAnalysis

# Setup logging
logger = logging.getLogger(__name__)

# Helper function for intent clarification
def intent_to_action(intent: str) -> str:
    """Convert an intent to a user-friendly action description."""
    intent_actions = {
        "kb_query": "search the knowledge base",
        "populate_template": "fill out a template",
        "analyze_template": "analyze a template",
        "temp_context_query": "answer questions about your uploaded files",
        "kb_query_with_temp_context": "search the knowledge base with context from your files",
        "rag": "get information from the knowledge base"
    }
    return intent_actions.get(intent, "process your request")

async def get_semantic_intent_scores(user_query: str, client, model: str) -> Dict[str, float]:
    """
    Calculate semantic similarity between query and different intents
    using the LLM's understanding rather than keyword matching.
    """
    prompt = f"""
    Analyze the following user query and rate how strongly it matches each of these intents on a scale of 0.0 to 1.0:

    User Query: "{user_query}"

    Intents to rate:
    1. populate_template: User wants to fill, create, or generate a document based on a template
    2. analyze_template: User wants to analyze, compare, or explain aspects of a template
    3. kb_query: User wants information from the knowledge base
    4. temp_context_query: User wants information from uploaded documents
    5. kb_query_with_temp_context: User wants to combine knowledge base information with uploaded documents

    For each intent, provide a score between 0.0 (not matching at all) and 1.0 (perfect match).
    Explain your reasoning briefly, then provide the scores in JSON format.

    Output JSON format:
    {{"populate_template": score, "analyze_template": score, "kb_query": score, "temp_context_query": score, "kb_query_with_temp_context": score}}
    """

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        content = response.choices[0].message.content
        scores = json.loads(content)

        # Validate scores
        valid_scores = {}
        for intent, score in scores.items():
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                valid_scores[intent] = float(score)
            else:
                logger.warning(f"Invalid score for {intent}: {score}, defaulting to 0.5")
                valid_scores[intent] = 0.5

        return valid_scores
    except Exception as e:
        logger.error(f"Error getting semantic intent scores: {e}")
        # Return default scores if there's an error
        return {
            "populate_template": 0.5,
            "analyze_template": 0.5,
            "kb_query": 0.5,
            "temp_context_query": 0.5,
            "kb_query_with_temp_context": 0.5
        }

async def analyze_template_context(template_name: str, template_content: str, user_query: str, client, model: str) -> Dict[str, Any]:
    """
    Analyze template to determine document type and required fields using LLM
    rather than regex patterns.
    """
    prompt = f"""
    Analyze the following template and user query to determine:
    1. The document type (e.g., employment_contract, invoice, general_document)
    2. Required fields that need to be filled in this template
    3. How likely the user wants to populate vs. analyze this template (score 0-1 for each)

    Template Name: {template_name}

    Template Content:
    {template_content[:2000]}  # Limit content length

    User Query: "{user_query}"

    Provide your analysis in JSON format with these keys:
    - document_type: string
    - required_fields: array of strings (field names)
    - populate_score: float (0-1)
    - analyze_score: float (0-1)
    """

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        content = response.choices[0].message.content
        analysis = json.loads(content)

        # Ensure required fields are normalized
        if "required_fields" in analysis and isinstance(analysis["required_fields"], list):
            analysis["required_fields"] = [
                field.lower().replace(" ", "_") if isinstance(field, str) else str(field)
                for field in analysis["required_fields"]
            ]

        return analysis
    except Exception as e:
        logger.error(f"Error analyzing template context: {e}")
        # Return minimal default analysis
        return {
            "document_type": "unknown",
            "required_fields": [],
            "populate_score": 0.5,
            "analyze_score": 0.5
        }

async def analyze_conversation_context(history: List[Dict[str, str]], user_query: str, client, model: str) -> Dict[str, float]:
    """
    Analyze conversation history to determine intent continuity and context
    """
    # Extract the last few messages for context
    recent_messages = history[-5:] if len(history) > 5 else history

    # Format messages for the prompt
    formatted_history = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}..."
        for msg in recent_messages
    ])

    # Check if the last assistant message contains a KB file listing
    kb_file_listing_detected = False
    kb_meta_query_detected = False
    last_assistant_msg = None

    # Find the last assistant message
    for i in range(len(recent_messages) - 1, -1, -1):
        msg = recent_messages[i]
        if msg.get('role') == 'assistant':
            last_assistant_msg = msg.get('content', '')
            break

    # Check for KB file listing in the last assistant message
    kb_list_markers = [
        "# Knowledge Base Contents",
        "The knowledge base contains",
        "files in the knowledge base",
        "documents in the knowledge base",
        "# Files Available"
    ]
    last_msg_contains_kb_list = last_assistant_msg and any(marker in last_assistant_msg for marker in kb_list_markers)

    # Check for KB meta-query in recent user messages
    for i in range(len(recent_messages) - 1, -1, -1):
        msg = recent_messages[i]
        if msg.get('role') == 'user' and any(kw in msg.get('content', '').lower() for kw in ['what files', 'what documents', 'what\'s in the kb', 'knowledge base contents', 'list files', 'show files']):
            kb_meta_query_detected = True
            logger.info("Detected KB meta-query in recent conversation history")
            break
        if msg.get('role') == 'assistant' and any(kw in msg.get('content', '').lower() for kw in ['knowledge base contents', '# knowledge base', 'files in the knowledge base', 'documents in the knowledge base']):
            kb_file_listing_detected = True
            logger.info("Detected KB file listing in recent assistant message")
            break

    prompt = f"""
    Analyze this conversation history and the new user query to determine:
    1. If the user is continuing a previous intent
    2. If the user is referring to specific items or content from previous messages (especially the LAST assistant message)
    3. How the conversation context affects the likely true intent

    Conversation History (most recent last):
    {formatted_history}

    New User Query: "{user_query}"

    {'IMPORTANT NOTE: The IMMEDIATELY PRECEDING ASSISTANT MESSAGE contained a list of Knowledge Base files.' if last_msg_contains_kb_list else ''}
    {'IMPORTANT NOTE: The conversation history shows that the assistant recently provided a list of knowledge base files to the user.' if kb_file_listing_detected else ''}
    {'IMPORTANT NOTE: The conversation history shows that the user recently asked about the contents of the knowledge base.' if kb_meta_query_detected else ''}

    Is the New User Query making a follow-up question or statement specifically about the content, files, or items mentioned or listed in the IMMEDIATELY PRECEDING ASSISTANT MESSAGE? Rate this likelihood (0.0 to 1.0):
    - refers_to_previous_assistant_output: [Score]

    If the IMMEDIATELY PRECEDING ASSISTANT MESSAGE contained a list of Knowledge Base files, and the New User Query refers to "those" or "these files" or uses other pronouns like "they", "them", strongly consider a 'kb_query' intent bias, and strongly decrease the bias for 'temp_context_query' unless temporary files are explicitly mentioned in the new query.

    Pay special attention to pronouns like "those", "these", "they", "them" in the user query that may refer to previously mentioned items, especially if they refer to knowledge base files that were just listed.

    Rate how the conversation history suggests each intent (0.0 to 1.0):
    - populate_template: User wants to fill a template
    - analyze_template: User wants to analyze a template (consider boosting if user asks to analyze KB files after listing)
    - kb_query: User wants knowledge base information (consider boosting if user refers to listed KB files)
    - temp_context_query: User wants information from uploaded documents (decrease score if user refers to listed KB files but no temp files)
    - kb_query_with_temp_context: User wants combined information

    Additionally, if the user query contains pronouns (like "those", "these", "they") that seem to refer to previously listed knowledge base files, add a field "kb_file_reference_score" with a value between 0.0 and 1.0 indicating how likely the user is referring to KB files.

    Provide your analysis as a JSON object with these intents as keys and scores as values, plus the 'refers_to_previous_assistant_output' score.
    """

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        content = response.choices[0].message.content
        bias_scores = json.loads(content)

        # Validate scores
        valid_scores = {}
        for key, score in bias_scores.items():
            # Handle both intent scores and the special refers_to_previous_assistant_output score
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                valid_scores[key] = float(score)
            else:
                # Default to 0.0 for invalid scores
                valid_scores[key] = 0.0
                logger.warning(f"Invalid score for {key}: {score}, defaulting to 0.0")

        # Ensure we have a refers_to_previous_assistant_output score
        if 'refers_to_previous_assistant_output' not in valid_scores:
            # If not provided, default to 0.0
            valid_scores['refers_to_previous_assistant_output'] = 0.0
            logger.info("No refers_to_previous_assistant_output score provided, defaulting to 0.0")
        else:
            logger.info(f"refers_to_previous_assistant_output score: {valid_scores['refers_to_previous_assistant_output']}")

        return valid_scores
    except Exception as e:
        logger.error(f"Error analyzing conversation context: {e}")
        # Return neutral bias if there's an error
        return {
            "populate_template": 0.0,
            "analyze_template": 0.0,
            "kb_query": 0.0,
            "temp_context_query": 0.0,
            "kb_query_with_temp_context": 0.0,
            "refers_to_previous_assistant_output": 0.0,
            "kb_file_reference_score": 0.0
        }

async def estimate_analyzer_confidence(analyzer_output: Any) -> float:
    """
    Estimate the confidence of the QueryAnalyzerAgent's intent determination
    """
    # If the output is a string containing "confidence" or "score", try to extract it
    if isinstance(analyzer_output, str) and ("confidence" in analyzer_output.lower() or "score" in analyzer_output.lower()):
        try:
            # Try to find a confidence value in the string
            import re
            confidence_matches = re.findall(r'confidence["\s:]+([0-9.]+)', analyzer_output.lower())
            score_matches = re.findall(r'score["\s:]+([0-9.]+)', analyzer_output.lower())

            matches = confidence_matches + score_matches
            if matches:
                # Use the first match
                confidence = float(matches[0])
                # Ensure it's in the range [0, 1]
                return max(0.0, min(1.0, confidence))
        except Exception:
            pass

    # Default confidence based on output type
    if isinstance(analyzer_output, dict) and "intent" in analyzer_output:
        # Structured output suggests higher confidence
        return 0.8
    elif hasattr(analyzer_output, "intent") and hasattr(analyzer_output, "details"):
        # Object with proper attributes suggests higher confidence
        return 0.8
    elif isinstance(analyzer_output, str) and len(analyzer_output) > 0:
        # String output suggests medium confidence
        return 0.6
    else:
        # Unknown output suggests low confidence
        return 0.4

async def determine_final_intent(
    analyzer_intent: str,
    analyzer_details: Dict[str, Any],
    user_query: str,
    template_name: Optional[str] = None,
    template_content: Optional[str] = None,
    temp_files_info: Optional[List[Dict]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    client = None,
    model: str = "gpt-4o-mini",
    document_analyses: Optional[List[DocumentAnalysis]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Determine the final intent using a sophisticated approach that leverages LLM capabilities
    rather than regex patterns or hardcoded fallbacks.
    """
    logger.info("Starting enhanced intent determination")

    # Initialize scores with the analyzer's intent having a base value
    intent_scores = {
        "populate_template": 0.0,
        "analyze_template": 0.0,
        "kb_query": 0.0,
        "temp_context_query": 0.0,
        "kb_query_with_temp_context": 0.0
    }

    # 1. Get semantic intent scores directly from LLM
    semantic_scores = await get_semantic_intent_scores(user_query, client, model)
    logger.info(f"Semantic intent scores: {semantic_scores}")

    # Add semantic scores to our total scores
    for intent, score in semantic_scores.items():
        intent_scores[intent] += score

    # 2. Consider template context if available
    template_analysis = None
    if template_name and template_content:
        template_analysis = await analyze_template_context(
            template_name, template_content, user_query, client, model
        )
        logger.info(f"Template analysis: {template_analysis}")

        # Adjust scores based on template analysis
        if "populate_score" in template_analysis:
            intent_scores["populate_template"] += template_analysis["populate_score"]

        if "analyze_score" in template_analysis:
            intent_scores["analyze_template"] += template_analysis["analyze_score"]

    # 3. Consider conversation history if available
    kb_file_reference_detected = False
    refers_to_previous_output = False
    has_temp_files = bool(temp_files_info)

    if history:
        conversation_bias = await analyze_conversation_context(
            history, user_query, client, model
        )
        logger.info(f"Conversation context bias: {conversation_bias}")

        # Extract special scores
        refers_score = 0.0
        kb_file_reference_score = 0.0

        if 'refers_to_previous_assistant_output' in conversation_bias:
            refers_score = conversation_bias.pop('refers_to_previous_assistant_output')
            logger.info(f"Extracted refers_to_previous_assistant_output score: {refers_score:.4f}")

        if 'kb_file_reference_score' in conversation_bias:
            kb_file_reference_score = conversation_bias.pop('kb_file_reference_score')
            logger.info(f"Extracted kb_file_reference_score: {kb_file_reference_score:.4f}")

        # Check if the user is referring to previous assistant output
        if refers_score > 0.7:  # High likelihood threshold
            refers_to_previous_output = True
            logger.info(f"High likelihood of referring to previous assistant output: {refers_score:.4f}")

            # Check if the previous assistant message was a KB file list
            last_assistant_msg = None
            for msg in reversed(history):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg.get('content', '')
                    break

            kb_list_markers = [
                "# Knowledge Base Contents",
                "The knowledge base contains",
                "files in the knowledge base",
                "documents in the knowledge base",
                "# Files Available"
            ]
            kb_list_in_last_msg = last_assistant_msg and any(marker in last_assistant_msg for marker in kb_list_markers)

            if kb_list_in_last_msg:
                logger.info("Previous assistant message appears to be KB list. Biasing towards KB query/analysis.")
                # Strong boost for kb_query
                intent_scores["kb_query"] += refers_score * 1.5

                # Check if this is a comparison query
                comparison_keywords = ["identical", "compare", "similar", "same", "difference", "different"]
                is_comparison_query = any(keyword in user_query.lower() for keyword in comparison_keywords)

                if is_comparison_query:
                    # Boost analyze_template for comparison queries
                    intent_scores["analyze_template"] += refers_score * 1.0
                    logger.info("Query appears to be comparing KB files. Boosting analyze_template score.")

                # If no temp files are present, strongly suppress temp_context_query
                if not has_temp_files:
                    intent_scores["temp_context_query"] *= (1.0 - refers_score)
                    logger.info(f"No temp files present. Reducing temp_context_query score to {intent_scores['temp_context_query']:.4f}")

                # Set the KB file reference flag
                kb_file_reference_detected = True

        # Also check the kb_file_reference_score as a backup
        elif kb_file_reference_score > 0.5:  # Threshold for considering it a reference
            kb_file_reference_detected = True
            logger.info(f"Detected reference to KB files with score: {kb_file_reference_score:.4f}")
            # Boost kb_query score significantly for file references
            intent_scores["kb_query"] += kb_file_reference_score * 1.0

        # Add conversation bias to our scores
        for intent, bias in conversation_bias.items():
            intent_scores[intent] += bias * 0.5  # Weight conversation context less

    # 4. Consider the analyzer's intent with high weight
    analyzer_confidence = await estimate_analyzer_confidence(analyzer_intent)
    logger.info(f"Analyzer confidence for {analyzer_intent}: {analyzer_confidence:.4f}")

    # Give the analyzer's intent a significant boost
    if analyzer_intent in intent_scores:
        intent_scores[analyzer_intent] += analyzer_confidence * 1.5
        logger.info(f"Boosted {analyzer_intent} score based on analyzer confidence")

    # 5. Consider document analyses if available
    if document_analyses:
        logger.info(f"Using {len(document_analyses)} document analyses for intent determination")

        # Calculate average confidence across all analyses
        avg_confidence = sum(analysis.confidence for analysis in document_analyses) / len(document_analyses)
        logger.info(f"Average document analysis confidence: {avg_confidence:.4f}")

        # Check document types
        doc_types = [analysis.doc_type for analysis in document_analyses]
        logger.info(f"Document types detected: {doc_types}")

        # Adjust scores based on document types and confidence
        if any("contract" in doc_type.lower() for doc_type in doc_types):
            # Contracts are likely to be templates that need population
            intent_scores["populate_template"] += 0.3 * avg_confidence
            intent_scores["analyze_template"] += 0.2 * avg_confidence
            intent_scores["kb_query_with_temp_context"] += 0.1 * avg_confidence

        if any("form" in doc_type.lower() for doc_type in doc_types):
            # Forms are likely to need data extraction
            intent_scores["temp_context_query"] += 0.3 * avg_confidence
            intent_scores["populate_template"] += 0.2 * avg_confidence

        # If we have multiple documents, they're more likely to be for context queries
        if len(document_analyses) > 1:
            intent_scores["temp_context_query"] += 0.2
            intent_scores["kb_query_with_temp_context"] += 0.2

    # 6. Consider UI state (template selection, temp files)
    if template_name:
        # Having a template selected slightly increases template-related intents
        intent_scores["populate_template"] += 0.2
        intent_scores["analyze_template"] += 0.1

    if temp_files_info:
        # Having temp files slightly increases temp file-related intents
        intent_scores["temp_context_query"] += 0.2
        intent_scores["kb_query_with_temp_context"] += 0.1

    # 6. Make final decision
    final_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
    highest_score = intent_scores[final_intent]
    logger.info(f"Final determined intent: {final_intent} (score: {highest_score:.4f})")

    # Check if the confidence is too low
    confidence_threshold = 0.6  # Adjust this threshold as needed
    needs_clarification = highest_score < confidence_threshold

    # If confidence is low, set a flag in final_details
    if needs_clarification:
        logger.warning(f"Low confidence intent detection: {final_intent} (score: {highest_score:.4f}). May need clarification.")

        # Get the second highest intent for reference
        second_intent = None
        second_score = 0.0
        for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
            if intent != final_intent:
                second_intent = intent
                second_score = score
                break

        if second_intent:
            logger.info(f"Alternative intent: {second_intent} (score: {second_score:.4f})")

    # 7. Prepare detailed intent information
    final_details = analyzer_details.copy()

    # Add confidence and clarification information
    final_details["confidence"] = highest_score
    final_details["needs_clarification"] = needs_clarification

    if needs_clarification and second_intent:
        final_details["alternative_intent"] = second_intent
        final_details["alternative_score"] = second_score
        final_details["clarification_message"] = f"I'm not entirely sure if you want to {intent_to_action(final_intent)} or {intent_to_action(second_intent)}. Could you please clarify?"

    # Add document analyses to final details if available
    if document_analyses:
        # Convert document analyses to serializable format
        doc_analyses_data = []
        for analysis in document_analyses:
            doc_analysis_dict = {
                "doc_type": analysis.doc_type,
                "confidence": analysis.confidence,
                "key_sections": analysis.key_sections,
                "language": analysis.language,
                "metadata": analysis.metadata
            }
            doc_analyses_data.append(doc_analysis_dict)

        final_details["document_analyses"] = doc_analyses_data

    if final_intent == "populate_template" and template_analysis:
        # Use template analysis for required fields
        if "required_fields" in template_analysis and template_analysis["required_fields"]:
            final_details["required_fields"] = template_analysis["required_fields"]
            final_details["document_type"] = template_analysis.get("document_type", "unknown")

        # If we have document analyses, use them to enhance template population
        if document_analyses:
            # Extract document types and key sections for template population
            doc_types = [analysis.doc_type for analysis in document_analyses]
            all_key_sections = [section for analysis in document_analyses for section in analysis.key_sections]

            # Add to final details
            final_details["document_types"] = doc_types
            final_details["key_sections"] = all_key_sections

    elif final_intent == "analyze_template" and template_name:
        final_details["template_name"] = template_name
        final_details["needs_kb_lookup"] = True
        final_details["needs_temp_files"] = bool(temp_files_info)

        # If we have document analyses, use them to enhance template analysis
        if document_analyses:
            # Extract document types for template analysis
            doc_types = [analysis.doc_type for analysis in document_analyses]
            final_details["document_types"] = doc_types

    elif final_intent == "kb_query_with_temp_context" and temp_files_info:
        final_details["query_topic"] = user_query
        final_details["temp_file_names"] = [f['filename'] for f in temp_files_info]

        # If we have document analyses, use them to refine the KB query
        if document_analyses:
            # Extract key sections and metadata for KB query refinement
            all_key_sections = [section for analysis in document_analyses for section in analysis.key_sections]
            final_details["key_sections"] = all_key_sections

    elif final_intent == "temp_context_query" and temp_files_info:
        final_details["query"] = user_query
        final_details["temp_filenames"] = [f['filename'] for f in temp_files_info]

        # If we have document analyses, use them to enhance temp context query
        if document_analyses:
            # Extract document types and key sections for temp context query
            doc_types = [analysis.doc_type for analysis in document_analyses]
            all_key_sections = [section for analysis in document_analyses for section in analysis.key_sections]

            # Add to final details
            final_details["document_types"] = doc_types
            final_details["key_sections"] = all_key_sections

    # Ensure query is always included in details
    if "query" not in final_details:
        final_details["query"] = user_query

    # Handle KB file reference detection
    if final_intent == "kb_query" and kb_file_reference_detected:
        # Set kb_query_type to file_analysis for queries about previously listed KB files
        final_details["kb_query_type"] = "file_analysis"
        logger.info("Setting kb_query_type to file_analysis based on detected reference to KB files")

        # Add analysis type based on query content
        if any(word in user_query.lower() for word in ["identical", "same", "similar", "compare", "difference", "different"]):
            final_details["analysis_type"] = "comparison"
        elif any(word in user_query.lower() for word in ["size", "largest", "smallest", "bigger", "smaller"]):
            final_details["analysis_type"] = "size_analysis"
        elif any(word in user_query.lower() for word in ["date", "created", "modified", "when", "time", "latest", "newest", "oldest"]):
            final_details["analysis_type"] = "date_analysis"
        elif any(word in user_query.lower() for word in ["content", "contain", "about", "topic", "subject", "information"]):
            final_details["analysis_type"] = "content_analysis"
        else:
            final_details["analysis_type"] = "general_analysis"

        # Add a flag indicating that this is a follow-up query
        final_details["is_followup_query"] = True

        # Log the detection of a follow-up query about KB files
        logger.info(f"Detected follow-up query about KB files with analysis_type: {final_details['analysis_type']}")

    return final_intent, final_details

async def record_intent_determination(
    user_query: str,
    determined_intent: str,
    analyzer_intent: str,
    intent_scores: Dict[str, float],
    final_workflow: Optional[str] = None
) -> None:
    """
    Record intent determination for analysis and improvement
    """
    # This would typically write to a database, but for now we'll just log it
    logger.info(f"Intent Determination Record: Query='{user_query[:50]}...', "
                f"Determined={determined_intent}, Analyzer={analyzer_intent}, "
                f"Scores={intent_scores}, Final Workflow={final_workflow}")

    # In a real implementation, we would store this in a database for later analysis
    # intent_db.insert({
    #     "timestamp": time.time(),
    #     "user_query": user_query,
    #     "determined_intent": determined_intent,
    #     "analyzer_intent": analyzer_intent,
    #     "intent_scores": intent_scores,
    #     "final_workflow": final_workflow
    # })