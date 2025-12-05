"""
Meta-agent definition and tool wrappers for AgencyGen.
"""

import re

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from .config import DEFAULT_MODEL
from .registry import register_agent

from .patterns import (
    create_single_agent,
    create_voting_agents,
    create_debate_agents,
    create_reflection_agent,
    create_sequential_agent,
)

from .rlm import (
    create_chunking_rlm,
    create_iterative_rlm,
    create_hierarchical_rlm,
    RLMConfig,
)


def _sanitize_name(name: str) -> str:
    """
    Sanitize a name to be a valid Python identifier.
    ADK requires agent names to be valid identifiers.
    """
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    if name and name[0].isdigit():
        name = "_" + name
    return name.lower() or "agent"


def _tool_create_single(
    name: str,
    instruction: str,
    description: str = "",
) -> str:
    safe_name = _sanitize_name(name)
    agent = create_single_agent(safe_name, instruction, description)
    register_agent(safe_name, agent)
    return f"Created single agent '{safe_name}' with instruction: {instruction[:100]}..."


def _tool_create_voting(
    name: str,
    instruction: str,
    num_voters: int = 3,
) -> str:
    safe_name = _sanitize_name(name)
    voting_system = create_voting_agents(safe_name, instruction, num_voters)
    register_agent(safe_name, voting_system)
    return f"Created voting system '{safe_name}' with {num_voters} voters"


def _tool_create_debate(
    name: str,
    topic: str,
    num_debaters: int = 2,
) -> str:
    safe_name = _sanitize_name(name)
    debate_system = create_debate_agents(safe_name, topic, num_debaters)
    register_agent(safe_name, debate_system)
    return f"Created debate '{safe_name}' with {num_debaters} debaters and a judge"


def _tool_create_reflection(
    name: str,
    task: str,
) -> str:
    safe_name = _sanitize_name(name)
    reflection_system = create_reflection_agent(safe_name, task)
    register_agent(safe_name, reflection_system)
    return f"Created reflection system '{safe_name}' with worker and critic"


def _tool_create_pipeline(
    name: str,
    steps_json: str,
) -> str:
    import json

    safe_name = _sanitize_name(name)
    steps = json.loads(steps_json)
    for step in steps:
        if "name" in step:
            step["name"] = _sanitize_name(step["name"])
    pipeline = create_sequential_agent(safe_name, steps)
    register_agent(safe_name, pipeline)
    return f"Created pipeline '{safe_name}' with {len(steps)} steps: {[s['name'] for s in steps]}"


def _tool_create_rlm_chunking(
    name: str,
    instruction: str,
    chunk_size: int = 4000,
) -> str:
    """Create a chunking RLM for processing long contexts."""
    safe_name = _sanitize_name(name)
    config = RLMConfig(chunk_size=chunk_size)
    rlm = create_chunking_rlm(safe_name, instruction, config)
    register_agent(safe_name, rlm)
    return f"Created chunking RLM '{safe_name}' with {chunk_size} char chunks"


def _tool_create_rlm_iterative(
    name: str,
    instruction: str,
    max_iterations: int = 5,
    convergence_threshold: float = 0.95,
) -> str:
    """Create an iterative RLM for self-refinement."""
    safe_name = _sanitize_name(name)
    config = RLMConfig(
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )
    rlm = create_iterative_rlm(safe_name, instruction, config)
    register_agent(safe_name, rlm)
    return f"Created iterative RLM '{safe_name}' with max {max_iterations} iterations"


def _tool_create_rlm_hierarchical(
    name: str,
    instruction: str,
    max_depth: int = 3,
) -> str:
    """Create a hierarchical RLM for recursive decomposition."""
    safe_name = _sanitize_name(name)
    config = RLMConfig(max_depth=max_depth)
    rlm = create_hierarchical_rlm(safe_name, instruction, config)
    register_agent(safe_name, rlm)
    return f"Created hierarchical RLM '{safe_name}' with max depth {max_depth}"


AgencyGen = LlmAgent(
    name="AgencyGen",
    model=DEFAULT_MODEL,
    description="A meta-agent that creates task-specific agents and multi-agent systems",
    instruction="""You are AgencyGen, a meta-agent specialized in designing AI agents.

Your job: Analyze tasks and create the optimal agent or multi-agent system.

## Your Tools

1. **create_single** - Simple, focused agent for straightforward tasks
2. **create_voting** - Multiple agents vote for reliability (math, facts)
3. **create_debate** - Agents argue, judge decides (complex analysis)
4. **create_reflection** - Self-critique for high-quality output (writing)
5. **create_pipeline** - Sequential steps for workflows
6. **create_rlm_chunking** - Process very long documents by chunking
7. **create_rlm_iterative** - Self-refine until convergence
8. **create_rlm_hierarchical** - Recursively decompose complex problems

## Decision Guide

| Task Type | Best Tool | Why |
|-----------|-----------|-----|
| Simple Q&A | create_single | One agent is enough |
| Math/Facts | create_voting | Multiple votes = reliability |
| Analysis/Ethics | create_debate | Multiple perspectives help |
| Writing/Creative | create_reflection | Self-improvement = quality |
| Multi-step | create_pipeline | Clear stages |
| Long documents | create_rlm_chunking | Handles context limits |
| Iterative polish | create_rlm_iterative | Refines until perfect |
| Complex problems | create_rlm_hierarchical | Divide and conquer |

## How to Respond

1. Analyze what the user needs
2. Choose the best tool for the job
3. Call the tool with appropriate parameters
4. Explain your choice so users learn!

Always be helpful and explain your reasoning.""",
    tools=[
        FunctionTool(func=_tool_create_single),
        FunctionTool(func=_tool_create_voting),
        FunctionTool(func=_tool_create_debate),
        FunctionTool(func=_tool_create_reflection),
        FunctionTool(func=_tool_create_pipeline),
        FunctionTool(func=_tool_create_rlm_chunking),
        FunctionTool(func=_tool_create_rlm_iterative),
        FunctionTool(func=_tool_create_rlm_hierarchical),
    ],
)

