"""
Core Package

This package contains the core functionality of the application.
"""

from .workflow_registry import WorkflowRegistry, workflow_registry, initialize_workflow_registry
from .orchestration import OrchestrationEngine, execute_orchestrated_workflow

__all__ = [
    'WorkflowRegistry', 'workflow_registry', 'initialize_workflow_registry',
    'OrchestrationEngine', 'execute_orchestrated_workflow'
]
