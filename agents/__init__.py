"""
Agents Package

This package contains agent-related functionality.
"""

# Import common classes that should be available at the package level
from .agent_output import AgentOutputSchema
from .run_context_wrapper import RunContextWrapper
from .function_tool import function_tool

# Define classes that might be imported from the OpenAI Agents SDK
# These are placeholder implementations that can be replaced when the actual SDK is available

class Agent:
    """Placeholder for the Agent class from the OpenAI Agents SDK."""
    def __init__(self, name=None, instructions=None, model=None, tools=None, tool_use_behavior=None, handoffs=None, output_type=None, model_settings=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.tool_use_behavior = tool_use_behavior
        self.handoffs = handoffs or []
        self.output_type = output_type
        self.model_settings = model_settings or {}

class Runner:
    """Implementation of the Runner class that calls the OpenAI API."""
    @staticmethod
    async def run(agent, input=None, context=None):
        """Run an agent with the given input and context asynchronously.

        Args:
            agent: The agent to run
            input: The input to the agent (string, dict, or list of messages)
            context: Optional context for the agent

        Returns:
            RunResult object with the agent's output
        """
        import logging
        import json
        from .result import RunResult

        logger = logging.getLogger(__name__)

        # Get the OpenAI client from the context
        client = None
        if context and "client" in context:
            client = context["client"]
        else:
            # Try to get the client from the app module
            try:
                from app import get_openai_client
                client = get_openai_client()
            except (ImportError, AttributeError):
                logger.error("Could not get OpenAI client")
                return RunResult(final_output={"error": "Could not get OpenAI client"})

        if not client:
            logger.error("OpenAI client not available")
            return RunResult(final_output={"error": "OpenAI client not available"})

        # Prepare the messages
        messages = []

        # Add system message with agent instructions
        if agent.instructions:
            messages.append({"role": "system", "content": agent.instructions})

        # Add user message with input
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, dict):
            # Convert dict to JSON string
            messages.append({"role": "user", "content": json.dumps(input)})
        elif isinstance(input, list):
            # Assume input is a list of messages
            messages.extend(input)

        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model=agent.model or "gpt-4o-mini",
                messages=messages,
                temperature=0.7,
            )

            # Get the response content
            content = response.choices[0].message.content

            # Try to parse as JSON if it looks like JSON
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    # If it's not valid JSON, keep it as a string
                    pass

            return RunResult(final_output=content)
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return RunResult(final_output={"error": f"Error calling OpenAI API: {str(e)}"})

    @staticmethod
    def run_sync(agent, input=None, context=None):
        """Run an agent with the given input and context synchronously.

        Args:
            agent: The agent to run
            input: The input to the agent (string, dict, or list of messages)
            context: Optional context for the agent

        Returns:
            RunResult object with the agent's output
        """
        import logging
        import json
        import asyncio
        from .result import RunResult

        logger = logging.getLogger(__name__)

        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async method in the event loop
        try:
            return loop.run_until_complete(Runner.run(agent, input, context))
        except Exception as e:
            logger.error(f"Error running agent synchronously: {e}")
            return RunResult(final_output={"error": f"Error running agent synchronously: {str(e)}"})

class Handoff:
    """Placeholder for the Handoff class from the OpenAI Agents SDK."""
    pass

# Import items module
from . import items

# Try to import ContentProcessorAgent
try:
    # Create ContentProcessorAgent instance
    content_processor_instructions = """
You are a specialized content processing agent responsible for analyzing, summarizing, and extracting information from knowledge base documents. Your primary role is to process document content and provide meaningful, well-structured responses based on the user's query.
"""
    
    ContentProcessorAgent = Agent(
        name="ContentProcessorAgent",
        instructions=content_processor_instructions,
        model="gpt-4o-mini"
    )
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to create ContentProcessorAgent: {e}")
    ContentProcessorAgent = None
