"""
Backwards-compatible aggregation of AgencyGen public API.
All functionality is split into submodules for maintainability.
"""

from .config import DEFAULT_MODEL, AVAILABLE_MODELS, DEFAULT_COUNCIL_MODELS
from .analysis import _analyze_task_keywords, _analyze_task_llm
from .patterns import (
    create_single_agent,
    create_voting_agents,
    create_debate_agents,
    create_reflection_agent,
    create_sequential_agent,
)
from .runner import run_agent
from .solve import solve
from .a2a import create_a2a_app, connect_to_remote_agent
from .composite import create_agent_tool, create_composite_agent, create_composite_with_remote
from .meta_agent import AgencyGen

__all__ = [
    "solve",
    "AgencyGen",
    "create_single_agent",
    "create_voting_agents",
    "create_debate_agents",
    "create_reflection_agent",
    "create_sequential_agent",
    "run_agent",
    "create_a2a_app",
    "connect_to_remote_agent",
    "create_agent_tool",
    "create_composite_agent",
    "create_composite_with_remote",
    "_analyze_task_keywords",
    "_analyze_task_llm",
    "DEFAULT_MODEL",
    "AVAILABLE_MODELS",
    "DEFAULT_COUNCIL_MODELS",
]

