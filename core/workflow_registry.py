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
    if app_module is None:
        try:
            import app as app_module
        except ImportError:
            logger.error("Could not import app module")
            return workflow_registry
    
    # Register the default workflows
    try:
        # Map intent names to workflow functions
        workflow_mapping = {
            "kb_query_workflow": app_module.run_kb_workflow,
            "temp_context_workflow": None,  # Will be set below
            "kb_temp_context_workflow": None,  # Will be set below
            "template_population_workflow": None,  # Will be set below
            "template_analysis_workflow": None,  # Will be set below
        }
        
        # Extract workflow functions from the run_complex_rag_workflow function
        if hasattr(app_module, "run_complex_rag_workflow"):
            complex_workflow = app_module.run_complex_rag_workflow
            
            # Define wrapper functions for each workflow type
            async def temp_context_workflow(**kwargs):
                """Wrapper for temporary context workflow"""
                # Set intent to temp_context_query for compatibility
                kwargs["intent"] = "temp_context_query"
                return await complex_workflow(**kwargs)
            
            async def kb_temp_context_workflow(**kwargs):
                """Wrapper for KB + temporary context workflow"""
                # Set intent to kb_query_with_temp_context for compatibility
                kwargs["intent"] = "kb_query_with_temp_context"
                return await complex_workflow(**kwargs)
            
            async def template_population_workflow(**kwargs):
                """Wrapper for template population workflow"""
                # Set intent to populate_template for compatibility
                kwargs["intent"] = "populate_template"
                return await complex_workflow(**kwargs)
            
            async def template_analysis_workflow(**kwargs):
                """Wrapper for template analysis workflow"""
                # Set intent to analyze_template for compatibility
                kwargs["intent"] = "analyze_template"
                return await complex_workflow(**kwargs)
            
            # Update the workflow mapping
            workflow_mapping["temp_context_workflow"] = temp_context_workflow
            workflow_mapping["kb_temp_context_workflow"] = kb_temp_context_workflow
            workflow_mapping["template_population_workflow"] = template_population_workflow
            workflow_mapping["template_analysis_workflow"] = template_analysis_workflow
        
        # Register all workflows
        for name, func in workflow_mapping.items():
            if func is not None:
                workflow_registry.register_workflow(name, func)
        
        logger.info(f"Initialized workflow registry with {len(workflow_registry.workflows)} workflows")
    except Exception as e:
        logger.error(f"Error initializing workflow registry: {e}")
    
    return workflow_registry
