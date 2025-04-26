"""
Workflow Registry Module

This module provides a registry for dynamically loading and managing workflows.
"""

import logging
import importlib
from typing import Dict, Callable, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class WorkflowRegistry:
    """
    A registry for dynamically loading and managing workflows.

    This class provides functionality to:
    1. Register workflow functions by name
    2. Retrieve workflow functions by name
    3. Execute workflows with parameters
    """

    def __init__(self):
        """Initialize the workflow registry."""
        self.workflows = {}  # Dictionary mapping workflow names to workflow functions

    def register_workflow(self, name: str, workflow_func: Callable) -> None:
        """
        Register a workflow function by name.

        Args:
            name: Name of the workflow
            workflow_func: The workflow function to register
        """
        self.workflows[name] = workflow_func
        logger.info(f"Registered workflow: {name}")

    def get_workflow(self, name: str) -> Optional[Callable]:
        """
        Get a workflow function by name.

        Args:
            name: Name of the workflow to retrieve

        Returns:
            The workflow function, or None if not found
        """
        return self.workflows.get(name)

    def execute_workflow(self, name: str, **kwargs) -> Any:
        """
        Execute a workflow by name with the provided parameters.

        Args:
            name: Name of the workflow to execute
            **kwargs: Parameters to pass to the workflow function

        Returns:
            The result of the workflow execution

        Raises:
            ValueError: If the workflow is not found
        """
        workflow_func = self.get_workflow(name)
        if workflow_func is None:
            raise ValueError(f"Workflow not found: {name}")

        return workflow_func(**kwargs)

    def list_workflows(self) -> List[str]:
        """
        List all registered workflows.

        Returns:
            List of workflow names
        """
        return list(self.workflows.keys())


# Create a singleton instance of the workflow registry
workflow_registry = WorkflowRegistry()


def initialize_workflow_registry(app_module=None) -> WorkflowRegistry:
    """
    Initialize the workflow registry with the default workflows.

    Args:
        app_module: The app module containing the workflow functions

    Returns:
        The initialized workflow registry
    """
    # Register the default workflows
    try:
        # Define simple placeholder functions to avoid circular imports
        # These will be replaced with actual implementations when called

        async def placeholder_workflow(**kwargs):
            """Simple placeholder that returns a message"""
            return "This is a placeholder workflow. The actual workflow will be called when needed."

        # Map intent names to placeholder functions
        workflow_mapping = {
            "kb_query_workflow": placeholder_workflow,
            "temp_context_workflow": placeholder_workflow,
            "kb_temp_context_workflow": placeholder_workflow,
            "template_population_workflow": placeholder_workflow,
            "template_analysis_workflow": placeholder_workflow,
        }

        # Register all workflows
        for name, func in workflow_mapping.items():
            if func is not None:
                workflow_registry.register_workflow(name, func)

        logger.info(f"Initialized workflow registry with {len(workflow_registry.workflows)} workflows")
    except Exception as e:
        logger.error(f"Error initializing workflow registry: {e}")

    return workflow_registry
