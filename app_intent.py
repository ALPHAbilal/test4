import json
import re
import logging

# Set up logger
logger = logging.getLogger(__name__)

def determine_final_intent(analyzer_output):
    """Extracts the final intent and details from the analyzer output.
    Handles different output formats from the analyzer agent."""
    try:
        # Case 1: Output is a dictionary with intent and details
        if isinstance(analyzer_output, dict) and "intent" in analyzer_output:
            intent = analyzer_output.get("intent", "kb_query")
            details = analyzer_output.get("details", {})

            # Check for kb_query_type in details for meta-queries
            if intent == "kb_query" and isinstance(details, dict) and details.get("kb_query_type") == "meta":
                logger.info("Detected meta-query about KB contents from analyzer output")

            return intent, details

        # Case 2: Output is a string that can be parsed as JSON
        if isinstance(analyzer_output, str):
            try:
                parsed = json.loads(analyzer_output)
                if isinstance(parsed, dict) and "intent" in parsed:
                    intent = parsed.get("intent", "kb_query")
                    details = parsed.get("details", {})

                    # Check for kb_query_type in details for meta-queries
                    if intent == "kb_query" and isinstance(details, dict) and details.get("kb_query_type") == "meta":
                        logger.info("Detected meta-query about KB contents from analyzer output (JSON string)")

                    return intent, details
            except json.JSONDecodeError:
                # Not valid JSON, continue to other cases
                pass

        # Case 3: Output is a string with intent mentioned
        if isinstance(analyzer_output, str):
            # Extract intent using regex
            intent_match = re.search(r'intent[\"\\\':\\s]+(\\w+)', analyzer_output, re.IGNORECASE)
            if intent_match:
                intent = intent_match.group(1).lower()
                # Try to extract details as well
                details_match = re.search(r'details[\"\\\':\\s]+({.+})', analyzer_output, re.IGNORECASE)
                details = {}
                if details_match:
                    try:
                        details_str = details_match.group(1)
                        details = json.loads(details_str)

                        # Check for kb_query_type in details for meta-queries
                        if intent == "kb_query" and isinstance(details, dict) and details.get("kb_query_type") == "meta":
                            logger.info("Detected meta-query about KB contents from analyzer output (regex)")

                    except json.JSONDecodeError:
                        # Couldn't parse details, use empty dict
                        pass

                # Check for kb_query_type in the string itself if not found in details
                if intent == "kb_query" and "kb_query_type" in analyzer_output and "meta" in analyzer_output:
                    logger.info("Detected possible meta-query about KB contents from analyzer output text")
                    if "kb_query_type" not in details:
                        details["kb_query_type"] = "meta"

                return intent, details

        # Default fallback
        return "kb_query", {}
    except Exception as e:
        logger.error(f"Error determining intent: {e}")
        return "kb_query", {}
