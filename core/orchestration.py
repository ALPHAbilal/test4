"""
Orchestration Module

This module provides the main execution loop for step-by-step orchestration.
"""

import json
import logging
import html
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
        # Initialize step counter
        step_count = 0

        # Add the user query to the workflow context
        workflow_context["current_query"] = user_query

        # Initialize workflow tracking if not already present
        if "workflow_steps" not in workflow_context:
            workflow_context["workflow_steps"] = []

        # Get the WorkflowRouterAgent from the registry
        router_agent = self.agent_registry.get_agent("WorkflowRouterAgent")
        if not router_agent:
            logger.error("WorkflowRouterAgent not found in registry")
            return "Error: WorkflowRouterAgent not found in registry"

        # Main execution loop
        while step_count < self.max_steps:
            step_count += 1
            logger.info(f"Executing workflow step {step_count}")

            try:
                # Prepare input for the router agent
                router_input = {
                    "user_query": user_query,
                    "step_count": step_count,
                    "workflow_context": workflow_context
                }

                # Call the router agent
                router_result = await Runner.run(router_agent, input=json.dumps(router_input), context=workflow_context)

                # Parse the router's output
                try:
                    router_output = self._parse_router_output(router_result.final_output)
                except Exception as e:
                    logger.error(f"Error parsing router output: {e}")
                    return f"Error: Failed to parse router output: {html.escape(str(e))}"

                # Get the action and details
                action = router_output.get("action")
                details = router_output.get("details", {})
                state_update = router_output.get("state_update", {})

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
                            import time
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
                        return f"Error: Agent not found: {html.escape(agent_name)}"

                    # Add the agent name to the context
                    workflow_context["current_agent_name"] = agent_name

                    # Get agent memory if available
                    if memory_store:
                        agent_memory = memory_store.get_memory(agent_name, workflow_context.get("chat_id"))
                        if agent_memory:
                            # Add agent memory to the context
                            workflow_context["agent_memory"] = agent_memory

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
                            agent_memory = memory_store.get_memory(agent_name, workflow_context.get("chat_id"))
                            if agent_memory:
                                # Add agent memory to the context
                                workflow_context["agent_memory"] = agent_memory

                    # Create a context wrapper
                    ctx = RunContextWrapper(workflow_context)

                    # Call the tool
                    tool_result = await tool.on_invoke_tool(ctx, tool_parameters)

                    # Update the workflow context with the tool result
                    workflow_context["last_tool_result"] = tool_result
                    workflow_context["last_action"] = "call_tool"
                    workflow_context["last_tool_name"] = tool_name

                    # Remove agent memory from the context
                    workflow_context.pop("agent_memory", None)

                elif action == "return_to_user":
                    # Return the final response to the user
                    final_response = details.get("final_response", "")
                    logger.info("Returning final response to user")
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
