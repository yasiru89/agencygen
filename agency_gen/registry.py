"""
Agent registry for storing created agents.

Provides a simple in-memory registry for agents and RLMs created by
the meta-agent tools, allowing them to be retrieved and used later.
"""

from typing import Dict, Any, Optional, Union
from google.adk.agents import LlmAgent, SequentialAgent


# Global registry for storing created agents
_agent_registry: Dict[str, Any] = {}


def register_agent(name: str, agent: Any) -> None:
    """
    Register an agent in the global registry.
    
    Args:
        name: Unique name for the agent
        agent: The agent or agent dict to store
    """
    _agent_registry[name] = agent


def get_agent(name: str) -> Optional[Any]:
    """
    Retrieve an agent from the registry.
    
    Args:
        name: The name of the agent to retrieve
        
    Returns:
        The agent if found, None otherwise
    """
    return _agent_registry.get(name)


def get_all_agents() -> Dict[str, Any]:
    """
    Get all registered agents.
    
    Returns:
        Dictionary of all registered agents
    """
    return _agent_registry.copy()


def remove_agent(name: str) -> Optional[Any]:
    """
    Remove an agent from the registry.
    
    Args:
        name: The name of the agent to remove
        
    Returns:
        The removed agent if found, None otherwise
    """
    return _agent_registry.pop(name, None)


def clear_registry() -> None:
    """Clear all agents from the registry."""
    _agent_registry.clear()


def list_agents() -> list:
    """
    List all registered agent names.
    
    Returns:
        List of agent names
    """
    return list(_agent_registry.keys())

