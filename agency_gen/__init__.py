"""
AgencyGen - A Multi-Agent System Generator using Google ADK
==========================================================

AgencyGen is a "meta-agent" that creates task-specific agents or 
multi-agent systems using Google's Agent Development Kit (ADK).

The core idea: AgencyGen is itself a basic ADK LlmAgent whose tools are
functions that create other agents. It analyzes a given task and
builds an optimal agent structure for that task.

Based on:
- Kaggle 5-Day Agents Intensive Course: https://www.kaggle.com/learn-guide/5-day-agents
- Google ADK: https://google.github.io/adk-docs/
- Common patterns in Multi-Agent Design research: https://arxiv.org/html/2502.02533v1
"""

from .agency_gen import (
    # The simplest way - just describe your task!
    solve,
    
    # The main meta-agent
    AgencyGen,
    
    # Tools that AgencyGen uses to build agents
    create_single_agent,
    create_voting_agents,
    create_debate_agents,
    create_reflection_agent,
    create_sequential_agent,
    
    # Helper to run agents
    run_agent,
    
    # A2A: Expose agents as network services
    create_a2a_app,
    connect_to_remote_agent,
    
    # Composite patterns: Compose multiple agents together
    create_agent_tool,
    create_composite_agent,
    create_composite_with_remote,
    
    # Model configuration
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_COUNCIL_MODELS,
)

__version__ = "0.1.0"
__all__ = [
    # The simplest way
    "solve",
    # Everything else
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
    "DEFAULT_MODEL",
    "AVAILABLE_MODELS",
    "DEFAULT_COUNCIL_MODELS",
]
