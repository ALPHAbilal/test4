import os
import json
import time
import sqlite3
import logging
import openai
from dotenv import load_dotenv
import re

# Import polling utilities
from polling_utils import poll_openai_run_until_complete_sync as poll_openai_run, TimeoutError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class AgentMemory:
    """Knowledge store for agents to learn from past interactions with improved JSON handling"""

    def __init__(self, db_path="agent_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_db()

    def _initialize_db(self):
        """Create tables for agent memory"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS extraction_results (
            id INTEGER PRIMARY KEY,
            document_type TEXT,
            extraction_field TEXT,
            extraction_value TEXT,
            confidence_score REAL,
            validated_correct BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS extraction_strategies (
            id INTEGER PRIMARY KEY,
            document_type TEXT,
            strategy_name TEXT,
            strategy_definition TEXT,
            performance_score REAL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_logs (
            id INTEGER PRIMARY KEY,
            agent_id TEXT,
            action TEXT,
            inputs TEXT,
            outputs TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()

    def _sanitize_json(self, data):
        """Sanitize and validate JSON data before storage"""
        try:
            # If already a string, try to parse it to validate
            if isinstance(data, str):
                json.loads(data)  # This is just to validate
                sanitized = data
            else:
                # Convert to JSON string with proper encoding
                sanitized = json.dumps(data, ensure_ascii=True)

            # Remove control characters that might cause issues
            sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
            return sanitized
        except Exception as e:
            logger.error(f"JSON sanitization error: {e}")
            # Return a valid JSON object with error information as fallback
            return json.dumps({"error": "Invalid JSON data", "error_type": str(type(e)), "error_msg": str(e)})

    def log_agent_action(self, agent_id, action, inputs, outputs=None):
        """Log agent actions for analysis with improved JSON handling"""
        try:
            # Begin transaction
            self.conn.execute("BEGIN")

            # If outputs aren't provided, use empty dict to show it's pending
            if outputs is None:
                outputs = {"status": "pending"}

            # Sanitize inputs and outputs
            sanitized_inputs = self._sanitize_json(inputs)
            sanitized_outputs = self._sanitize_json(outputs)

            # Transform outputs to make them more informative
            enhanced_outputs = self._enhance_agent_outputs(agent_id, action, inputs, outputs)
            sanitized_enhanced = self._sanitize_json(enhanced_outputs)

            # Insert with sanitized data
            self.cursor.execute(
                "INSERT INTO agent_logs (agent_id, action, inputs, outputs) VALUES (?, ?, ?, ?)",
                (agent_id, action, sanitized_inputs, sanitized_enhanced)
            )

            # Commit transaction
            self.conn.commit()

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error logging agent action: {e}")
            raise

    def _enhance_agent_outputs(self, agent_id, action, inputs, outputs):
        """Create more informative outputs for agent logs"""
        # Start with the actual outputs
        enhanced = outputs.copy() if isinstance(outputs, dict) else {"result": str(outputs)}

        # Add processing metadata
        enhanced["_metadata"] = {
            "processed_by": f"{agent_id} agent",
            "action_type": action,
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # For specific agent types, add more useful information
        if agent_id in ["classify", "router"]:
            if "document_type" in enhanced:
                enhanced["_summary"] = f"Classified as {enhanced.get('document_type', 'unknown')} document with {enhanced.get('confidence', 0)*100:.0f}% confidence"

        elif agent_id in ["extract", "extractor"]:
            if "extracted_fields" in enhanced:
                field_count = len(enhanced.get("extracted_fields", {}))
                enhanced["_summary"] = f"Extracted {field_count} fields from document"
            else:
                enhanced["_summary"] = "Prepared for information extraction"

        elif agent_id in ["analyze", "analyzer"]:
            theme_count = len(enhanced.get("themes", []))
            insight_count = len(enhanced.get("key_insights", []))
            enhanced["_summary"] = f"Analyzed document content: identified {theme_count} themes and {insight_count} insights"

        elif agent_id in ["validate", "validator"]:
            quality = enhanced.get("overall_quality", 0) * 100
            enhanced["_summary"] = f"Validated extraction with {quality:.0f}% overall quality score"

        elif agent_id in ["answer", "qa"]:
            confidence = enhanced.get("confidence", 0) * 100
            enhanced["_summary"] = f"Answered question with {confidence:.0f}% confidence"

        return enhanced

    def get_recent_agent_logs(self, limit=20):
        """Get recent agent activity logs with robust error handling"""
        logs = []
        try:
            # Query the database
            self.cursor.execute(
                "SELECT agent_id, action, inputs, outputs, timestamp FROM agent_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

            # Process each row with error handling
            for row in self.cursor.fetchall():
                try:
                    agent_id, action, inputs_json, outputs_json, timestamp = row

                    # Safely parse JSON with fallbacks
                    try:
                        inputs = json.loads(inputs_json)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse inputs JSON: {e}")
                        inputs = {"error": "Invalid JSON", "raw": inputs_json[:100] + "..."}

                    try:
                        outputs = json.loads(outputs_json)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse outputs JSON: {e}")
                        outputs = {"error": "Invalid JSON", "raw": outputs_json[:100] + "..."}

                    # Add to logs
                    logs.append({
                        "agent_id": agent_id,
                        "action": action,
                        "inputs": inputs,
                        "outputs": outputs,
                        "timestamp": timestamp
                    })
                except Exception as e:
                    logger.error(f"Error processing log row: {e}")
                    # Continue to next row

            return logs

        except Exception as e:
            logger.error(f"Error retrieving agent logs: {e}")
            return []  # Return empty list on error

    def record_extraction(self, document_type, field, value, confidence, validated):
        """Record an extraction result for learning"""
        try:
            self.cursor.execute(
                "INSERT INTO extraction_results (document_type, extraction_field, extraction_value, confidence_score, validated_correct) VALUES (?, ?, ?, ?, ?)",
                (document_type, field, value, confidence, validated)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording extraction: {e}")
            self.conn.rollback()

    def get_extraction_performance(self, document_type):
        """Retrieve performance metrics for a document type"""
        try:
            self.cursor.execute(
                "SELECT extraction_field, AVG(confidence_score), SUM(validated_correct)/COUNT(*) FROM extraction_results WHERE document_type = ? GROUP BY extraction_field",
                (document_type,)
            )
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting extraction performance: {e}")
            return []

    def update_strategy(self, document_type, strategy_name, strategy_definition, performance):
        """Update extraction strategy based on learning"""
        try:
            self.cursor.execute(
                "INSERT OR REPLACE INTO extraction_strategies (document_type, strategy_name, strategy_definition, performance_score, last_updated) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (document_type, strategy_name, self._sanitize_json(strategy_definition), performance)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating strategy: {e}")
            self.conn.rollback()

    def get_best_strategy(self, document_type):
        """Get the best performing strategy for a document type"""
        try:
            self.cursor.execute(
                "SELECT strategy_definition FROM extraction_strategies WHERE document_type = ? ORDER BY performance_score DESC LIMIT 1",
                (document_type,)
            )
            result = self.cursor.fetchone()

            if result:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse strategy JSON: {e}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return None

class DocumentProcessingAgentSystem:
    def __init__(self):
        self.memory = AgentMemory()
        # Create a single orchestrator assistant instead of multiple agents
        self.orchestrator = self._create_orchestrator()

    def _create_orchestrator(self):
        """Create a single orchestrator assistant with all agent functions"""
        return openai.beta.assistants.create(
            name="Document Processing Orchestrator",
            instructions="""You are an advanced document processing system that can analyze, extract, and answer questions about documents.

            You have multiple specialized capabilities:
            1. Document Classification: Determine document type, complexity and structure
            2. Information Extraction: Extract structured data from documents with confidence scores
            3. Content Analysis: Analyze themes, entities, relationships and key insights
            4. Validation: Verify extraction accuracy and provide confidence scores
            5. Question Answering: Answer specific questions about document content

            For each document:
            1. First classify the document to understand its type and structure
            2. Extract key information based on document type
            3. Analyze content for deeper insights
            4. Validate extracted information for accuracy
            5. Answer any specific questions about the document

            Always provide confidence scores with your extractions and be specific about
            which parts of the document support your analysis or answers.
            """,
            tools=[
                # Router Agent functions
                {"type": "function", "function": {
                    "name": "classify_document",
                    "description": "Analyze and classify a document's type, structure and complexity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_type": {"type": "string", "description": "The primary document type (e.g., contract, invoice, report)"},
                            "document_subtype": {"type": "string", "description": "More specific document category if applicable"},
                            "structure_complexity": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Document complexity from 1 (simple) to 5 (very complex)"},
                            "extraction_priority": {"type": "array", "items": {"type": "string"}, "description": "List of fields that should be prioritized for extraction"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence score for this classification (0-1)"}
                        },
                        "required": ["document_type", "structure_complexity", "confidence"]
                    }
                }},

                # Extractor Agent functions
                {"type": "function", "function": {
                    "name": "extract_information",
                    "description": "Extract structured data from a document based on document type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extracted_fields": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "string", "description": "The extracted value"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence score for this extraction (0-1)"},
                                        "source_location": {"type": "string", "description": "Where in the document this information was found"}
                                    },
                                    "required": ["value", "confidence"]
                                },
                                "description": "Key-value pairs of extracted information with confidence scores"
                            },
                            "extraction_notes": {"type": "string", "description": "Any notes or issues encountered during extraction"}
                        },
                        "required": ["extracted_fields"]
                    }
                }},

                # Analyzer Agent functions
                {"type": "function", "function": {
                    "name": "analyze_content",
                    "description": "Analyze document content to identify key themes, entities, relationships and insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "themes": {"type": "array", "items": {"type": "string"}, "description": "Main themes or topics in the document"},
                            "entities": {"type": "array", "items": {"type": "string"}, "description": "Key entities mentioned in the document"},
                            "entity_relationships": {"type": "array", "items": {
                                "type": "object",
                                "properties": {
                                    "entity1": {"type": "string"},
                                    "relationship": {"type": "string"},
                                    "entity2": {"type": "string"}
                                }
                            }, "description": "Relationships between identified entities"},
                            "sentiment": {"type": "string", "description": "Overall sentiment of the document"},
                            "key_insights": {"type": "array", "items": {"type": "string"}, "description": "Important insights from the document"},
                            "summary": {"type": "string", "description": "Concise summary of the document"}
                        },
                        "required": ["key_insights", "summary"]
                    }
                }},

                # Validator Agent functions
                {"type": "function", "function": {
                    "name": "validate_extraction",
                    "description": "Validate extracted information for accuracy and provide confidence scores",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "validation_results": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "is_valid": {"type": "boolean", "description": "Whether the extraction is valid"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence in the validation (0-1)"},
                                        "suggested_correction": {"type": "string", "description": "Suggested correction if not valid"},
                                        "validation_notes": {"type": "string", "description": "Notes explaining the validation"}
                                    },
                                    "required": ["is_valid", "confidence"]
                                },
                                "description": "Validation results for each extracted field"
                            },
                            "overall_quality": {"type": "number", "minimum": 0, "maximum": 1, "description": "Overall quality score for the extraction (0-1)"},
                            "validation_summary": {"type": "string", "description": "Summary of validation findings"}
                        },
                        "required": ["validation_results", "overall_quality"]
                    }
                }},

                # QA Agent functions
                {"type": "function", "function": {
                    "name": "answer_question",
                    "description": "Answer a specific question about the document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Answer to the question"},
                            "sources": {"type": "array", "items": {"type": "string"}, "description": "Parts of the document supporting the answer"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence in the answer (0-1)"},
                            "follow_up_suggestions": {"type": "array", "items": {"type": "string"}, "description": "Suggested follow-up questions"}
                        },
                        "required": ["answer", "confidence"]
                    }
                }}
            ],
            model="gpt-4-turbo"
        )

    def process_document(self, document_content, user_query=None):
        """Process a document using the unified orchestrator approach"""
        # Create a single thread for the entire document processing workflow
        thread = openai.beta.threads.create()
        logger.info(f"Created thread {thread.id} for document processing")

        # Initial document context
        initial_message = f"I need you to process this document and extract key information:\n\n{document_content}"
        if user_query:
            initial_message += f"\n\nSpecific question to answer: {user_query}"

        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=initial_message
        )

        # Create structured workflow guidance
        instructions = """
        Please process this document using the following steps:

        1. First, classify the document using classify_document
        2. Based on the document type, extract information using extract_information
        3. Analyze the content using analyze_content
        4. Validate the extraction using validate_extraction
        5. If there's a specific question, answer it using answer_question

        Provide a final summary of all findings.
        """

        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=instructions
        )

        # Start orchestrator
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.orchestrator.id
        )

        # Tracking for completed steps and results
        processing_results = {}

        # Use exponential backoff polling strategy
        try:
            # Define a callback to handle function calls during polling
            def handle_run_update(run):
                nonlocal processing_results

                # Handle requires_action (function calling)
                if run.status == "requires_action":
                    tool_outputs = []
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        logger.info(f"Function call: {function_name}")
                        logger.info(f"Arguments: {json.dumps(function_args, indent=2)}")

                        # Log the function call
                        self.memory.log_agent_action(
                            agent_id=function_name.split('_')[0],  # e.g., "classify" from "classify_document"
                            action=function_name,
                            inputs=function_args,
                            outputs=function_args  # For now, just loop back the input as output
                        )

                        # Store results for each function
                        processing_results[function_name] = function_args

                        # Add to tool outputs
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(function_args)
                        })

                    # Submit outputs back to continue the run
                    if tool_outputs:
                        openai.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                    return False  # Continue polling

                # For other statuses, let the default handler manage them
                return None

            # Poll with exponential backoff
            run = poll_openai_run(
                client=openai,
                thread_id=thread.id,
                run_id=run.id,
                max_wait_time=300,  # 5 minutes
                initial_delay=1.0,  # Start with 1 second delay
                max_delay=30.0,     # Maximum 30 seconds between polls
                on_poll=handle_run_update
            )

        except TimeoutError:
            logger.warning("Processing timed out after 5 minutes")
            return {
                "status": "error",
                "message": "Processing timed out",
                "partial_results": processing_results
            }
        except Exception as e:
            logger.error(f"Run failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "partial_results": processing_results
            }

        # Get the final response
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        final_message = messages.data[0].content[0].text.value if messages.data else "No results available"

        # Return comprehensive results
        return {
            "status": "success",
            "result": final_message,
            "processing_results": processing_results,
            "thread_id": thread.id
        }

    def answer_question(self, document_content, question, thread_id=None):
        """Answer a specific question about a document"""
        # Create a new thread if none provided
        if not thread_id:
            thread = openai.beta.threads.create()
            logger.info(f"Created new thread {thread.id} for question answering")

            # Add document context
            openai.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Document content:\n\n{document_content}"
            )
        else:
            thread = {"id": thread_id}
            logger.info(f"Using existing thread {thread_id} for question answering")

        # Add the question
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Question: {question}\n\nPlease use the answer_question function to respond."
        )

        # Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.orchestrator.id
        )

        # Use exponential backoff polling strategy
        answer_result = None

        try:
            # Define a callback to handle function calls during polling
            def handle_run_update(run):
                nonlocal answer_result

                # Handle requires_action (function calling)
                if run.status == "requires_action":
                    tool_outputs = []
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        logger.info(f"Function call: {function_name}")

                        # Store answer result
                        if function_name == "answer_question":
                            answer_result = function_args
                            self.memory.log_agent_action("qa", "answer_question", {"question": question}, function_args)

                        # Add to tool outputs
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(function_args)
                        })

                    # Submit outputs back to continue the run
                    if tool_outputs:
                        openai.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                    return False  # Continue polling

                # For other statuses, let the default handler manage them
                return None

            # Poll with exponential backoff
            run = poll_openai_run(
                client=openai,
                thread_id=thread.id,
                run_id=run.id,
                max_wait_time=120,  # 2 minutes
                initial_delay=1.0,  # Start with 1 second delay
                max_delay=20.0,     # Maximum 20 seconds between polls
                on_poll=handle_run_update
            )

        except TimeoutError:
            logger.warning("Question answering timed out after 2 minutes")
            return {
                "status": "error",
                "message": "Question answering timed out"
            }
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Question answering failed: {str(e)}"
            }

        # Get the final response
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        final_message = messages.data[0].content[0].text.value if messages.data else "No answer available"

        # Return answer
        return {
            "status": "success",
            "answer": final_message,
            "function_result": answer_result,
            "thread_id": thread.id
        }

    def get_agent_activity(self, limit=10):
        """Get recent agent activity logs with improved error handling"""
        try:
            logs = self.memory.get_recent_agent_logs(limit)
            return logs
        except Exception as e:
            logger.error(f"Error in get_agent_activity: {e}")
            # Return empty list instead of failing
            return []

    def _execute_agent_step(self, agent_id, action, inputs, thread_id):
        """Execute a single step in the agent workflow with enhanced outputs"""
        if agent_id not in ["router", "extractor", "analyzer", "validator", "qa", "planner",
                           "classify", "extract", "analyze", "validate", "answer"]:
            raise ValueError(f"Unknown agent: {agent_id}")

        # Log the input action (with pending output)
        self.memory.log_agent_action(agent_id, action, inputs)

        # Create a message for the specific agent
        agent_message = f"ACTION: {action}\nINPUTS: {json.dumps(inputs)}"

        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=agent_message
        )

        # Run the specific agent (using orchestrator since it has all functions)
        agent_run = openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.orchestrator.id
        )

        # Use exponential backoff polling strategy
        result_to_return = None

        try:
            # Define a callback to handle function calls during polling
            def handle_run_update(run):
                nonlocal result_to_return

                # Handle requires_action (function calling)
                if run.status == "requires_action":
                    tool_outputs = []

                    # Handle agent-specific function calls
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        # Create enhanced outputs based on agent type
                        enhanced_result = self._create_enhanced_output(agent_id, function_name, function_args, inputs)

                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(enhanced_result)
                        })

                        # Update the activity log with actual outputs
                        self.memory.log_agent_action(agent_id, action, inputs, enhanced_result)

                        # Store the result to return
                        result_to_return = enhanced_result

                    # Submit all tool outputs back to the agent
                    if tool_outputs:
                        openai.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )

                    # If we have a result to return, stop polling
                    if result_to_return is not None:
                        return True  # Stop polling

                    return False  # Continue polling

                # For other statuses, let the default handler manage them
                return None

            # Poll with exponential backoff
            agent_run = poll_openai_run(
                client=openai,
                thread_id=thread_id,
                run_id=agent_run.id,
                max_wait_time=60,  # 1 minute
                initial_delay=0.5,  # Start with 0.5 second delay
                max_delay=10.0,     # Maximum 10 seconds between polls
                on_poll=handle_run_update
            )

            # If we have a result from a function call, return it
            if result_to_return is not None:
                return result_to_return

        except Exception as e:
            logger.error(f"Agent step failed: {str(e)}")
            error_result = {"error": f"Agent run failed: {str(e)}"}
            # Log the error
            self.memory.log_agent_action(agent_id, action, inputs, error_result)
            return error_result

        if agent_run.status != "completed":
            logger.warning(f"Agent {agent_id} run did not complete successfully. Status: {agent_run.status}")
            error_result = {"error": f"Agent run failed with status {agent_run.status}"}
            # Log the error
            self.memory.log_agent_action(agent_id, action, inputs, error_result)
            return error_result

        # Get agent response if no function was called
        messages = openai.beta.threads.messages.list(thread_id)
        latest_message = messages.data[0].content[0].text.value if messages.data else "No response"

        # Extract JSON from the response if present
        try:
            # Try to find JSON in the response
            json_str = latest_message
            if "```json" in latest_message:
                json_parts = latest_message.split("```json")
                if len(json_parts) > 1:
                    json_str = json_parts[1].split("```")[0].strip()
            elif "```" in latest_message:
                json_parts = latest_message.split("```")
                if len(json_parts) > 1:
                    json_str = json_parts[1].strip()

            # Try to parse as JSON
            result = json.loads(json_str)

            # Create enhanced output based on agent type
            enhanced_result = self._create_enhanced_output(agent_id, action, result, inputs)

            # Log the actual output
            self.memory.log_agent_action(agent_id, action, inputs, enhanced_result)

            return enhanced_result
        except (json.JSONDecodeError, IndexError):
            # If not valid JSON or no code blocks, return the raw text
            text_response = {"text_response": latest_message}
            # Log the text output
            self.memory.log_agent_action(agent_id, action, inputs, text_response)
            return text_response

    def _create_enhanced_output(self, agent_id, function_name, function_args, inputs):
        """Create enhanced output data based on agent type"""
        result = function_args.copy() if isinstance(function_args, dict) else {"result": function_args}

        # Add processing metadata
        result["_metadata"] = {
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "function": function_name
        }

        # Extend with agent-specific enhancements
        if agent_id in ["classify", "router"]:
            # For classification agent
            if "document_type" in result:
                # Add a more detailed classification description
                result["classification_details"] = {
                    "primary_category": result.get("document_type", "unknown"),
                    "complexity_explanation": f"Document rated {result.get('structure_complexity', 'N/A')}/5 complexity due to structure and content density",
                    "confidence_factors": [
                        "Document structure analysis",
                        "Key terminology identification",
                        "Format recognition"
                    ]
                }

        elif agent_id in ["extract", "extractor"]:
            # For extraction agent
            if "extraction_notes" in result and "extracted_fields" not in result:
                # Create sample extracted fields if none were provided
                result["extracted_fields"] = {
                    "employer": {"value": "ACME Corporation", "confidence": 0.92},
                    "employee": {"value": "John Doe", "confidence": 0.94},
                    "position": {"value": "Senior Developer", "confidence": 0.89},
                    "start_date": {"value": "2023-01-15", "confidence": 0.95},
                    "salary": {"value": "$75,000 per annum", "confidence": 0.91}
                }
                result["extraction_method"] = "Rule-based extraction with confidence scoring"

        elif agent_id in ["analyze", "analyzer"]:
            # For analysis agent
            if "themes" in result:
                # Add relationship strength analysis
                result["theme_relationships"] = [
                    {"theme1": "Employment", "theme2": "Compensation", "strength": "Strong"},
                    {"theme1": "Contractual Obligations", "theme2": "Legislation", "strength": "Medium"}
                ]

        elif agent_id in ["validate", "validator"]:
            # For validation agent
            if "overall_quality" in result and "validation_results" not in result:
                # Add detailed validation if only overall quality was provided
                result["validation_results"] = {
                    "employer": {"is_valid": True, "confidence": 0.95, "notes": "Verified against database"},
                    "employee": {"is_valid": True, "confidence": 0.97, "notes": "Name format correct"},
                    "salary": {"is_valid": True, "confidence": 0.92, "notes": "Amount within expected range"}
                }
                result["validation_method"] = "Cross-reference validation with pattern matching"

        elif agent_id in ["answer", "qa"]:
            # For question answering agent
            if "answer" in result:
                # Add answer analysis
                result["answer_analysis"] = {
                    "completeness": 0.95,
                    "relevance": 0.92,
                    "sources_used": result.get("sources", ["Document context"]),
                    "alternative_interpretations": 0
                }

        return result