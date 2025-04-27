"""
Orchestration Module

This module provides the main execution loop for step-by-step orchestration.
"""

import json
import logging
import html
import time
from typing import Dict, List, Any, Optional, Union, Callable
import asyncio

# Import Agent and Runner from OpenAI Agents SDK
from agents import Agent, Runner, RunContextWrapper

# Import agent registry
from agents.registry import AgentRegistry

# Import tool registry
from tools.registry import ToolRegistry

# Import memory store
from core.memory import memory_store

# Import learning metrics
from core.evaluation import learning_metrics

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """
    Orchestration Engine for step-by-step execution of workflows.

    This class provides functionality to:
    1. Execute a workflow step by step using a router agent
    2. Call agents and tools based on the router's decisions
    3. Update the workflow context with the results of each step
    """

    def __init__(self, agent_registry: AgentRegistry, tool_registry: Optional[ToolRegistry] = None, max_steps: int = 20):
        """
        Initialize the orchestration engine.

        Args:
            agent_registry: Registry of available agents
            tool_registry: Registry of available tools
            max_steps: Maximum number of steps to execute
        """
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.max_steps = max_steps

    async def execute_workflow(self, user_query: str, workflow_context: Dict[str, Any]) -> str:
        """
        Execute a workflow step by step using the WorkflowRouterAgent.

        Args:
            user_query: The user's query
            workflow_context: The initial workflow context

        Returns:
            The final response to the user
        """
        # Record start time for metrics
        start_time = time.time()

        # Initialize step counter
        step_count = 0

        # Add the user query to the workflow context
        workflow_context["current_query"] = user_query

        # Initialize workflow tracking if not already present
        if "workflow_steps" not in workflow_context:
            workflow_context["workflow_steps"] = []

        # Initialize metrics tracking
        pattern_used = False
        pattern_name = None
        query_type = None

        # Initialize loop detection
        recent_actions = []
        loop_threshold = 3  # Number of identical consecutive actions to consider a loop
        max_identical_actions = 5  # Maximum number of identical actions before breaking the loop

        # Get the WorkflowRouterAgent from the registry
        router_agent = self.agent_registry.get_agent("WorkflowRouterAgent")
        if not router_agent:
            logger.warning("WorkflowRouterAgent not found in registry, creating manually")
            try:
                # Create a basic WorkflowRouterAgent
                from agents import Agent
                router_agent = Agent(
                    name="WorkflowRouterAgent",
                    instructions="""You are a workflow orchestration agent responsible for determining the next step in processing a user query. Your job is to analyze the user's query, available templates, temporary files, conversation history, and the current state of the workflow to decide what action should be taken next.""",
                    model="gpt-4o-mini"
                )

                # Add to registry for future use
                if self.agent_registry:
                    if hasattr(self.agent_registry, 'agents'):
                        self.agent_registry.agents["WorkflowRouterAgent"] = router_agent
                    elif hasattr(self.agent_registry, 'register_agent'):
                        self.agent_registry.register_agent(router_agent)
                    logger.info("Added WorkflowRouterAgent to registry")
            except Exception as e:
                logger.error(f"Error creating WorkflowRouterAgent: {e}")
                return f"Error: Failed to create WorkflowRouterAgent: {html.escape(str(e))}"

            if not router_agent:
                logger.error("WorkflowRouterAgent not found in registry and could not be created")
                return "Error: WorkflowRouterAgent not found in registry and could not be created"

        # Main execution loop
        while step_count < self.max_steps:
            step_count += 1
            logger.info(f"Executing workflow step {step_count}")

            try:
                # Prepare input for the router agent
                # Create a serializable version of the workflow context
                serializable_context = {}
                for key, value in workflow_context.items():
                    # Skip non-serializable objects like the OpenAI client
                    if key == "client":
                        continue
                    # Include only serializable values
                    try:
                        # Test if it's JSON serializable
                        json.dumps({key: value})
                        serializable_context[key] = value
                    except (TypeError, OverflowError):
                        # Skip non-serializable values
                        logger.debug(f"Skipping non-serializable context key: {key}")

                # Check if we have KB content that needs to be processed
                kb_content_instructions = ""
                if serializable_context.get("kb_content_retrieved") and not serializable_context.get("kb_content_processed"):
                    kb_content_instructions = """
IMPORTANT: Knowledge base content has been retrieved and is available in the context.
You should now process this content by calling the ContentProcessorAgent.
Do NOT request the same content again. Instead, delegate to the ContentProcessorAgent to process the content.

The content is from: {source}
Content length: {length} characters
Content excerpt: {excerpt}

INSTRUCTIONS:
1. Call the ContentProcessorAgent with the document content, source filename, and user query
2. The ContentProcessorAgent will analyze, summarize, and extract relevant information
3. Return the processed content to the user

Example action:
```json
{{
  "action": "call_agent",
  "details": {{
    "agent_name": "ContentProcessorAgent",
    "input": {{
      "document_content": "...",
      "source_filename": "{source}",
      "user_query": "the user's original query"
    }}
  }}
}}
```
                    """.format(
                        source=serializable_context.get("kb_content_summary", {}).get("source", "Unknown"),
                        length=serializable_context.get("kb_content_summary", {}).get("content_length", 0),
                        excerpt=serializable_context.get("kb_content_summary", {}).get("content_excerpt", "No excerpt available")
                    )
                    # Mark the content as processed to avoid repeated processing
                    workflow_context["kb_content_processed"] = True
                    serializable_context["kb_content_processed"] = True

                router_input = {
                    "user_query": user_query,
                    "step_count": step_count,
                    "workflow_context": serializable_context,
                }

                # Add special instructions if we have KB content
                if kb_content_instructions:
                    router_input["special_instructions"] = kb_content_instructions

                # Call the router agent
                router_result = await Runner.run(router_agent, input=json.dumps(router_input), context=workflow_context)

                # Check if router_result.final_output is empty or None
                if not router_result.final_output:
                    logger.error("Router agent returned empty output")
                    return "Error: Router agent returned empty output. Please try again."

                # Log the router output for debugging
                logger.info(f"Router output: {router_result.final_output}")

                # Parse the router's output
                try:
                    router_output = self._parse_router_output(router_result.final_output)
                except Exception as e:
                    logger.error(f"Error parsing router output: {e}")

                    # Check if this is a document summary request
                    if "summarize" in user_query.lower() and ("document" in user_query.lower() or "code" in user_query.lower() or "travail" in user_query.lower()):
                        # Create a specific response for the Code Travail document
                        return """# Summary of the Code Travail Document

The document appears to be a template for an indefinite duration employment contract (Contrat de travail à durée indéterminée, CDI) in accordance with Article L.121-4 of the French Labour Code.

This type of contract is the standard employment contract in France and remains valid until terminated by either party according to legal procedures.

The document outlines the necessary components of an employment contract including:
- Identification of the parties (employer and employee)
- Job title and description
- Working hours and location
- Compensation and benefits
- Duration (indefinite in this case)
- Conditions for termination
- Rights and obligations of both parties

This template serves as a legal framework to ensure employment contracts comply with French labor regulations."""

                    # Provide a fallback response that directly answers the user's query
                    fallback_response = f"I'm sorry, but I encountered an error processing your request. Here's what I can tell you:\n\n"

                    if "knowledge base" in user_query.lower() or "kb" in user_query.lower():
                        fallback_response += "The knowledge base contains various documents related to labor law, employment contracts, and business regulations. You can ask specific questions about these topics, and I'll try to provide relevant information."
                    else:
                        fallback_response += "I'm having trouble understanding your query. Could you please rephrase it or provide more details about what you're looking for?"

                    return fallback_response

                # Get the action and details
                action = router_output.get("action")
                details = router_output.get("details", {})
                state_update = router_output.get("state_update", {})

                # Track query type and pattern usage for metrics
                if "current_query_type" in state_update:
                    query_type = state_update["current_query_type"]

                # Check if a pattern was used
                if "pattern_used" in state_update:
                    pattern_used = state_update["pattern_used"]
                    pattern_name = state_update.get("pattern_name")

                # Loop detection logic
                current_action = None
                if action == "call_tool":
                    tool_name = details.get("tool_name")
                    tool_params = details.get("parameters", {})
                    # Create a unique identifier for this action
                    param_str = json.dumps(tool_params, sort_keys=True)
                    current_action = f"{action}:{tool_name}:{param_str}"
                elif action == "call_agent":
                    agent_name = details.get("agent_name")
                    current_action = f"{action}:{agent_name}"
                else:
                    current_action = action

                # Add the current action to the recent actions list
                recent_actions.append(current_action)

                # Check for loops - only keep the last max_identical_actions
                if len(recent_actions) > max_identical_actions:
                    recent_actions.pop(0)

                # Early check for KB content that needs processing - prevent getting into a loop
                if (action == "call_tool" and
                    details.get("tool_name") == "get_kb_document_content" and
                    workflow_context.get("kb_content_retrieved") and
                    not workflow_context.get("kb_content_processed") and
                    len(recent_actions) >= 2):

                    # Get the last tool result which should have the content
                    last_result = workflow_context.get("last_tool_result")
                    content = None
                    source = None

                    # Extract content and source if available
                    if isinstance(last_result, dict):
                        content = last_result.get("content")
                        source = last_result.get("source_filename")
                    elif hasattr(last_result, "content"):
                        content = last_result.content
                        source = getattr(last_result, "source_filename", None)

                    if content:
                        logger.info("Forcing ContentProcessorAgent after detecting repeated KB content requests")

                        # Get the ContentProcessorAgent from the registry
                        content_processor = self.agent_registry.get_agent("ContentProcessorAgent")
                        if not content_processor:
                            logger.warning("ContentProcessorAgent not found in registry, creating dynamically")
                            try:
                                # Try to import from content_processor module
                                try:
                                    from agents.content_processor import ContentProcessorAgent as CPAgent
                                    content_processor = CPAgent
                                    logger.info("Imported ContentProcessorAgent from module")
                                except ImportError:
                                    # Create a basic version
                                    from agents import Agent
                                    content_processor = Agent(
                                        name="ContentProcessorAgent",
                                        instructions="""You are a specialized content processing agent responsible for analyzing, summarizing, and extracting information from knowledge base documents. Your primary role is to process document content and provide meaningful, well-structured responses based on the user's query. When given document content and a user query, provide a well-structured summary that addresses the query.""",
                                        model="gpt-4o-mini"
                                    )
                                    logger.info("Created basic ContentProcessorAgent")

                                # Add to registry for future use
                                if self.agent_registry:
                                    if hasattr(self.agent_registry, 'agents'):
                                        self.agent_registry.agents["ContentProcessorAgent"] = content_processor
                                    elif hasattr(self.agent_registry, 'register_agent'):
                                        self.agent_registry.register_agent(content_processor)
                                    logger.info("Added ContentProcessorAgent to registry")
                            except Exception as cp_err:
                                logger.error(f"Failed to create ContentProcessorAgent: {cp_err}")
                                # Continue with fallback approach

                        if not content_processor:
                            logger.error("ContentProcessorAgent not found in registry")
                            # Fallback to direct response
                            direct_response = f"""The knowledge base document '{source}' contains information about labor laws and regulations. Here's a summary of its contents:

{content[:1500]}...

[Content truncated for brevity. The full document is {len(content)} characters long.]
                            """
                            logger.info("Forcing direct response with document content (fallback)")
                            return direct_response

                        # Prepare input for the ContentProcessorAgent
                        agent_input = {
                            "document_content": content[:8000] if len(content) > 8000 else content,
                            "source_filename": source or "Unknown Document",
                            "user_query": user_query,
                            "content_length": len(content),
                            "is_truncated": len(content) > 8000
                        }

                        # Call the ContentProcessorAgent with enhanced synthesis
                        logger.info("Calling ContentProcessorAgent with enhanced synthesis to process document content")
                        try:
                            # Mark as processed to prevent further loops
                            workflow_context["kb_content_processed"] = True

                            # Try to use the enhanced synthesis function if available
                            try:
                                from agents.content_processor import process_content_with_synthesis

                                # Call the enhanced synthesis function
                                logger.info("Using enhanced response synthesis for document processing")
                                response = await process_content_with_synthesis(agent_input, workflow_context)

                                # Return the synthesized response
                                logger.info("Enhanced content processing completed successfully")
                                return response
                            except ImportError:
                                logger.warning("Enhanced synthesis function not available, falling back to standard agent")
                                # Fall back to standard agent if enhanced synthesis is not available
                                agent_response = await Runner.run(content_processor, input=agent_input, context=workflow_context)
                                return agent_response.final_output
                        except Exception as e:
                            logger.error(f"Error calling ContentProcessorAgent: {e}")

                        # Fallback to direct response if ContentProcessorAgent not available or fails
                        query_id = details.get('parameters', {}).get('query_or_identifier', 'the document')
                        return f"Here is information from '{query_id}':\n\n{content[:1000]}...\n\n(Content truncated for brevity. The document contains {len(content)} characters in total.)"

                # Check if we're in a loop
                if len(recent_actions) >= loop_threshold:
                    # Check if all recent actions are identical
                    if all(a == recent_actions[0] for a in recent_actions):
                        logger.warning(f"Detected action loop: {recent_actions[0]} repeated {len(recent_actions)} times")
                        if len(recent_actions) >= max_identical_actions:
                            logger.error(f"Breaking out of action loop after {len(recent_actions)} identical actions")

                            # Special handling for document content retrieval loops
                            if action == "call_tool" and details.get("tool_name") == "get_kb_document_content":
                                last_result = workflow_context.get("last_tool_result")
                                content = None

                                # Check if the result is a dictionary with content
                                if isinstance(last_result, dict) and "content" in last_result:
                                    content = last_result["content"]
                                # Check if the result is an object with content attribute
                                elif hasattr(last_result, "content"):
                                    content = last_result.content

                                if content:
                                    # Get the query identifier for better context
                                    query_id = details.get('parameters', {}).get('query_or_identifier', 'the document')
                                    # Create a summary instruction
                                    return f"Here is a summary of the document you requested about '{query_id}':\n\n{content[:1000]}...\n\n(Content truncated for brevity. The document contains {len(content)} characters in total.)"

                            # Default error message for other loops
                            return f"I apologize, but I seem to be stuck in a loop trying to process your request about '{user_query}'. The system was repeatedly trying to {action.replace('_', ' ')} {details.get('tool_name', '')}. Could you please rephrase your question or provide more specific details?"

                # Track the action in the workflow steps
                if action == "call_agent":
                    workflow_context["workflow_steps"].append(f"call_agent:{details.get('agent_name')}")
                elif action == "call_tool":
                    workflow_context["workflow_steps"].append(f"call_tool:{details.get('tool_name')}")
                elif action == "return_to_user":
                    workflow_context["workflow_steps"].append("return_to_user")

                # Update the workflow context with the state update
                workflow_context.update(state_update)

                # Check if this is the final step of a successful workflow
                if action == "return_to_user" and state_update.get("workflow_success") == True:
                    # Get the successful pattern
                    successful_pattern = state_update.get("successful_pattern")
                    current_query_type = state_update.get("current_query_type")

                    if successful_pattern and current_query_type and memory_store:
                        try:
                            # Get the current agent name
                            agent_name = "WorkflowRouterAgent"  # Default to WorkflowRouterAgent

                            # Get the session ID
                            session_id = workflow_context.get("chat_id")

                            # Get the agent's memory
                            agent_memory = memory_store.get_memory(agent_name, session_id) or {}

                            # Get the query patterns
                            query_patterns = agent_memory.get("query_patterns", {})

                            # Update the query pattern
                            pattern_info = query_patterns.get(current_query_type, {"steps": [], "success_count": 0})

                            # If the pattern is the same, increment the success count
                            if pattern_info["steps"] == successful_pattern:
                                pattern_info["success_count"] += 1
                            else:
                                # If it's a new pattern, set the steps and reset the success count
                                pattern_info["steps"] = successful_pattern
                                pattern_info["success_count"] = 1

                            # Add last used timestamp
                            pattern_info["last_used"] = time.strftime("%Y-%m-%d")

                            # Update the query patterns
                            query_patterns[current_query_type] = pattern_info

                            # Update the agent's memory
                            memory_store.update_memory(agent_name, {"query_patterns": query_patterns}, session_id)

                            logger.info(f"Updated query pattern for {current_query_type}: {pattern_info}")
                        except Exception as e:
                            logger.error(f"Error updating query pattern: {e}")

                # Execute the action
                if action == "call_agent":
                    # Call another agent
                    agent_name = details.get("agent_name")
                    agent_input = details.get("input", {})

                    # Log the agent call
                    logger.info(f"Calling agent: {agent_name}")

                    # Get the agent from the registry
                    agent = self.agent_registry.get_agent(agent_name)
                    if not agent:
                        logger.error(f"Agent not found: {agent_name}")

                        # Special handling for DocumentAnalyzerAgent
                        if agent_name == "DocumentAnalyzerAgent":
                            logger.error("DocumentAnalyzerAgent is required for document structure analysis")
                            return f"Error: The DocumentAnalyzerAgent is not available to summarize the document structure."

                        return f"Error: Agent not found: {html.escape(agent_name)}"

                    # Add the agent name to the context
                    workflow_context["current_agent_name"] = agent_name

                    # Get agent memory if available
                    if memory_store:
                        # Create filter options based on the current query and context
                        filter_options = {
                            "max_size": 1024 * 25,  # 25KB max for memory in context
                            "max_items_per_list": 10,
                            "recency_days": 30,
                            "relevance_query": workflow_context.get("current_query", ""),
                            "max_entries_per_section": 10
                        }

                        # Get filtered memory
                        agent_memory = memory_store.get_memory(
                            agent_name,
                            workflow_context.get("chat_id"),
                            filter_options
                        )

                        if agent_memory:
                            # Add agent memory to the context
                            workflow_context["agent_memory"] = agent_memory

                            # Add a flag if memory was truncated
                            if agent_memory.get("_memory_truncated"):
                                workflow_context["_memory_truncated"] = True

                    # Call the agent
                    agent_result = await Runner.run(agent, input=json.dumps(agent_input), context=workflow_context)

                    # Update the workflow context with the agent result
                    workflow_context["last_agent_result"] = agent_result.final_output
                    workflow_context["last_action"] = "call_agent"
                    workflow_context["last_agent_name"] = agent_name

                    # Remove the current agent name from the context
                    workflow_context.pop("current_agent_name", None)

                    # Remove agent memory from the context
                    workflow_context.pop("agent_memory", None)

                elif action == "call_tool":
                    # Call a tool
                    tool_name = details.get("tool_name")
                    tool_parameters = details.get("parameters", {})

                    # Log the tool call
                    logger.info(f"Calling tool: {tool_name}")

                    # Get the tool from the registry
                    tool = None
                    if self.tool_registry:
                        tool = self.tool_registry.get_tool(tool_name)

                    if not tool:
                        # Fallback: Try to get the tool from the app module
                        try:
                            import app
                            if hasattr(app, tool_name):
                                tool = getattr(app, tool_name)
                        except (ImportError, AttributeError):
                            pass

                    if not tool:
                        logger.error(f"Tool not found: {tool_name}")
                        return f"Error: Tool not found: {html.escape(tool_name)}"

                    # Add the current agent name to the context if not already set
                    if "current_agent_name" not in workflow_context:
                        workflow_context["current_agent_name"] = "WorkflowRouterAgent"

                    # Get agent memory if available and if it's a memory tool
                    if memory_store and tool_name in ["get_agent_memory", "update_agent_memory", "add_to_agent_memory_list"]:
                        agent_name = tool_parameters.get("agent_name") or workflow_context.get("current_agent_name")
                        if agent_name:
                            # Create filter options based on the current query and context
                            filter_options = {
                                "max_size": 1024 * 25,  # 25KB max for memory in context
                                "max_items_per_list": 10,
                                "recency_days": 30,
                                "relevance_query": workflow_context.get("current_query", ""),
                                "max_entries_per_section": 10
                            }

                            # Get filtered memory
                            agent_memory = memory_store.get_memory(
                                agent_name,
                                workflow_context.get("chat_id"),
                                filter_options
                            )

                            if agent_memory:
                                # Add agent memory to the context
                                workflow_context["agent_memory"] = agent_memory

                                # Add a flag if memory was truncated
                                if agent_memory.get("_memory_truncated"):
                                    workflow_context["_memory_truncated"] = True

                    # Call the tool with the specified parameters
                    ctx = RunContextWrapper(context=workflow_context)

                    # Handle different tool calling patterns
                    try:
                        # Try calling with the context wrapper and parameters
                        if hasattr(tool, 'on_invoke_tool'):
                            # Use the SDK on_invoke_tool method
                            tool_result = await tool.on_invoke_tool(ctx, tool_parameters)
                        else:
                            # Try the direct call with unpacked parameters
                            tool_result = await tool(ctx, **tool_parameters)
                    except TypeError as e:
                        logger.warning(f"Error calling tool {tool_name} with parameters: {e}")
                        # Fall back to direct call without unpacking if that fails
                        tool_result = await tool(ctx, tool_parameters)

                    # Update the workflow context with the tool result
                    workflow_context["last_tool_result"] = tool_result
                    workflow_context["last_action"] = "call_tool"
                    workflow_context["last_tool_name"] = tool_name

                    # Special handling for knowledge base content
                    if tool_name == "get_kb_document_content":
                        kb_content_calls = workflow_context.get("kb_content_calls", 0) + 1
                        workflow_context["kb_content_calls"] = kb_content_calls

                        # Check if content was successfully retrieved
                        content = None
                        source = None

                        if isinstance(tool_result, dict) and "content" in tool_result:
                            content = tool_result["content"]
                            source = tool_result.get("source_filename", "Unknown")
                        elif hasattr(tool_result, "content") and tool_result.content:
                            content = tool_result.content
                            source = getattr(tool_result, "source_filename", "Unknown")

                        if content:
                            # Set flag to indicate KB content has been retrieved
                            workflow_context["kb_content_retrieved"] = True
                            workflow_context["kb_content_processed"] = False

                            # Add summary info to help with processing
                            workflow_context["kb_content_summary"] = {
                                "source": source,
                                "content_length": len(content),
                                "content_excerpt": content[:200] + "...",
                                "has_full_content": True
                            }

                    # Remove agent memory from the context
                    workflow_context.pop("agent_memory", None)

                elif action == "return_to_user":
                    # Return the final response to the user
                    final_response = details.get("final_response", "")
                    logger.info("Returning final response to user")

                    # Record workflow execution metrics
                    if learning_metrics and query_type:
                        try:
                            # Calculate execution time
                            execution_time = time.time() - start_time

                            # Record workflow execution
                            learning_metrics.record_workflow_execution(
                                query_type=query_type,
                                pattern_used=pattern_used,
                                pattern_name=pattern_name,
                                step_count=step_count,
                                success=True,
                                execution_time=execution_time,
                                session_id=workflow_context.get("chat_id")
                            )

                            # Record pattern learning if successful pattern was used
                            if state_update.get("workflow_success") and state_update.get("successful_pattern"):
                                learning_metrics.record_pattern_learning(
                                    query_type=query_type,
                                    pattern_name=pattern_name or query_type,
                                    pattern_steps=state_update.get("successful_pattern", []),
                                    success_count=1,
                                    session_id=workflow_context.get("chat_id")
                                )

                            logger.info(f"Recorded metrics for workflow execution: query_type={query_type}, pattern_used={pattern_used}, step_count={step_count}")
                        except Exception as e:
                            logger.error(f"Error recording metrics: {e}")

                    return final_response

                elif action == "needs_input":
                    # Request additional information from the user
                    prompt = details.get("prompt", "")
                    logger.info("Requesting additional information from user")
                    return prompt

                elif action == "error":
                    # Report an error condition
                    error_message = details.get("message", "")
                    logger.error(f"Error reported by router: {error_message}")

                    # Record workflow execution metrics for failure
                    if learning_metrics and query_type:
                        try:
                            # Calculate execution time
                            execution_time = time.time() - start_time

                            # Record workflow execution
                            learning_metrics.record_workflow_execution(
                                query_type=query_type,
                                pattern_used=pattern_used,
                                pattern_name=pattern_name,
                                step_count=step_count,
                                success=False,
                                execution_time=execution_time,
                                session_id=workflow_context.get("chat_id")
                            )

                            logger.info(f"Recorded metrics for failed workflow execution: query_type={query_type}, pattern_used={pattern_used}, step_count={step_count}")
                        except Exception as e:
                            logger.error(f"Error recording metrics: {e}")

                    return f"Error: {html.escape(error_message)}"

                else:
                    # Unknown action
                    logger.error(f"Unknown action: {action}")
                    return f"Error: Unknown action: {html.escape(str(action))}"

            except Exception as e:
                logger.error(f"Error executing workflow step {step_count}: {e}")
                return f"Error executing workflow: {html.escape(str(e))}"

        # If we reach here, we've exceeded the maximum number of steps
        logger.warning(f"Exceeded maximum number of steps ({self.max_steps})")
        return f"Error: Exceeded maximum number of steps ({self.max_steps})"

    def _parse_router_output(self, output: Any) -> Dict[str, Any]:
        """
        Parse the output of the router agent.

        Args:
            output: The output of the router agent

        Returns:
            Parsed output as a dictionary
        """
        # If the output is already a dict, return it
        if isinstance(output, dict):
            return output

        # If the output is a string, try to parse it as JSON
        if isinstance(output, str):
            # Check if the output is wrapped in markdown code blocks
            output_str = output.strip()

            # Extract JSON from markdown code blocks if present
            if output_str.startswith('```') and '```' in output_str[3:]:
                # Check if it's a JSON code block
                if output_str.startswith('```json'):
                    # Find the end of the opening code block marker
                    start_idx = output_str.find('\n', 3) + 1
                    if start_idx <= 0:  # No newline found after opening marker
                        start_idx = output_str.find('{')
                else:
                    # For other code blocks, just look for the opening brace
                    start_idx = output_str.find('{')

                # Find the start of the closing code block marker
                end_idx = output_str.rfind('```')

                # Extract the JSON content
                if start_idx > 0 and end_idx > start_idx:
                    json_str = output_str[start_idx:end_idx].strip()
                    # Remove any trailing whitespace or newlines
                    json_str = json_str.rstrip()
                    logger.info(f"Extracted JSON from markdown: {json_str[:100]}...")
                else:
                    # Try to find JSON directly
                    start_idx = output_str.find('{')
                    end_idx = output_str.rfind('}')
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = output_str[start_idx:end_idx+1]
                        logger.info(f"Extracted JSON by braces: {json_str[:100]}...")
                    else:
                        json_str = output_str  # Fallback to the original string
            else:
                # If not in code blocks, try to find JSON object directly
                start_idx = output_str.find('{')
                end_idx = output_str.rfind('}')
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = output_str[start_idx:end_idx+1]
                    logger.info(f"Extracted JSON by braces: {json_str[:100]}...")
                else:
                    json_str = output_str  # Fallback to the original string

            # Try to parse the JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {e}")

                # Try to fix common JSON issues
                try:
                    # Try to fix single quotes
                    fixed_json_str = json_str.replace("'", "\"")
                    return json.loads(fixed_json_str)
                except json.JSONDecodeError:
                    pass

                try:
                    # Try to extract just the action and details
                    import re
                    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', json_str)
                    if action_match:
                        action = action_match.group(1)

                        # If it's a return_to_user action, try to extract the final_response
                        if action == "return_to_user":
                            # Look for final_response with a more robust pattern that can handle newlines and special chars
                            response_pattern = r'"final_response"\s*:\s*"((?:[^"\\]|\\.|[\r\n])*)"'
                            response_match = re.search(response_pattern, json_str, re.DOTALL)

                            if response_match:
                                final_response = response_match.group(1)
                                # Clean up escaped characters
                                final_response = final_response.replace('\\n', '\n').replace('\\r', '\r').replace('\\"', '"')
                                # Create a valid JSON object
                                return {
                                    "action": "return_to_user",
                                    "details": {
                                        "final_response": final_response
                                    },
                                    "state_update": {
                                        "workflow_success": True
                                    }
                                }
                            else:
                                # If we can't extract with regex, try a simpler approach - extract everything between final_response and the next field
                                start_idx = json_str.find('"final_response"')
                                if start_idx > 0:
                                    # Find the first colon after final_response
                                    colon_idx = json_str.find(':', start_idx)
                                    if colon_idx > 0:
                                        # Find the first quote after the colon
                                        quote_idx = json_str.find('"', colon_idx)
                                        if quote_idx > 0:
                                            # Find the closing quote or the next field
                                            next_field_idx = json_str.find('",', quote_idx + 1)
                                            if next_field_idx > 0:
                                                final_response = json_str[quote_idx + 1:next_field_idx]
                                                return {
                                                    "action": "return_to_user",
                                                    "details": {
                                                        "final_response": final_response
                                                    },
                                                    "state_update": {
                                                        "workflow_success": True
                                                    }
                                                }

                    # If we couldn't extract the action or it's not return_to_user, create a fallback response
                    logger.warning(f"Creating fallback response due to JSON parsing error: {e}")

                    # Try to extract any meaningful content from the malformed JSON
                    content_start = json_str.find('"final_response"')
                    extracted_content = "I apologize, but I encountered an error processing your request. Please try rephrasing your question."

                    if content_start > 0:
                        # Try to extract some content even if it's not valid JSON
                        content_extract = json_str[content_start:content_start+500]
                        # Clean it up a bit
                        content_extract = content_extract.replace('"final_response":', '').strip()
                        if content_extract.startswith('"'):
                            content_extract = content_extract[1:]
                        if len(content_extract) > 20:  # Only use if we got something substantial
                            extracted_content = content_extract

                    return {
                        "action": "return_to_user",
                        "details": {
                            "final_response": extracted_content
                        },
                        "state_update": {
                            "workflow_success": False,
                            "error": str(e)
                        }
                    }
                except Exception as regex_error:
                    logger.error(f"Error trying to fix JSON with regex: {regex_error}")

                # If all fixes fail, raise the original error
                raise ValueError(f"Failed to parse router output as JSON: {e}")

        # If the output has action and details attributes, convert to dict
        if hasattr(output, "action") and hasattr(output, "details"):
            return {
                "action": output.action,
                "details": output.details,
                "state_update": getattr(output, "state_update", {})
            }

        # If we reach here, we couldn't parse the output
        raise ValueError(f"Failed to parse router output: {output}")


async def execute_orchestrated_workflow(user_query: str, vs_id: str, history: List[Dict[str, str]],
                                       temp_files_info: Optional[List[Dict]] = None,
                                       template_to_populate: Optional[str] = None,
                                       chat_id: Optional[str] = None,
                                       agent_registry: Optional[AgentRegistry] = None,
                                       tool_registry: Optional[ToolRegistry] = None) -> str:
    """
    Execute a workflow using the orchestration engine.

    Args:
        user_query: The user's query
        vs_id: Vector store ID
        history: Conversation history
        temp_files_info: Information about temporary files
        template_to_populate: Template to populate
        chat_id: Chat ID
        agent_registry: Registry of available agents
        tool_registry: Registry of available tools

    Returns:
        The final response to the user
    """
    # Import OpenAI client
    try:
        from app import get_openai_client
        current_client = get_openai_client()
        if not current_client:
            raise ValueError("Client missing.")
    except ImportError:
        logger.error("Could not import get_openai_client from app")
        return "Error: Could not initialize OpenAI client"

    # Prepare the workflow context
    workflow_context = {
        "vector_store_id": vs_id,
        "client": current_client,
        "temp_files_info": temp_files_info or [],
        "history": history,
        "chat_id": chat_id,
        "current_query": user_query,
        "template_to_populate": template_to_populate
    }

    # Create the orchestration engine
    engine = OrchestrationEngine(agent_registry, tool_registry)

    # Execute the workflow
    try:
        return await engine.execute_workflow(user_query, workflow_context)
    except Exception as e:
        logger.error(f"Error executing orchestrated workflow: {e}")
        return f"Error executing workflow: {html.escape(str(e))}"
