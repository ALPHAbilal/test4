"""
Agent Registry Package

This package provides functionality for dynamically loading and managing agents.
"""

from .agent_registry import AgentRegistry
from .registry_loader import initialize_agent_registry

__all__ = ['AgentRegistry', 'initialize_agent_registry']
