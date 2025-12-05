"""
Helpers for composing agents and combining local/remote tools.
"""

from typing import List, Optional

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from .config import DEFAULT_MODEL


def create_agent_tool(agent: LlmAgent, description: Optional[str] = None) -> AgentTool:
    """
    Wrap an agent as a tool that another agent can use.
    """
    return AgentTool(agent=agent, description=description)


def create_composite_agent(
    name: str,
    instruction: str,
    sub_agents: List[LlmAgent],
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a composite agent that orchestrates multiple sub-agents.
    """
    tools = [AgentTool(agent=sub_agent) for sub_agent in sub_agents]

    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        tools=tools,
        description=f"Composite agent with {len(sub_agents)} sub-agents",
    )


def create_composite_with_remote(
    name: str,
    instruction: str,
    local_agents: Optional[List[LlmAgent]] = None,
    remote_urls: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a composite agent that uses both local and remote (A2A) agents.
    """
    tools = []

    if local_agents:
        for agent in local_agents:
            tools.append(AgentTool(agent=agent))

    if remote_urls:
        try:
            from google.adk.a2a import RemoteA2aAgent
            for url in remote_urls:
                remote = RemoteA2aAgent(url=url)
                tools.append(AgentTool(agent=remote))
        except ImportError:
            raise ImportError(
                "A2A support requires google-adk with A2A dependencies. "
                "Install with: pip install google-adk[a2a]"
            )

    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        tools=tools,
        description=f"Composite agent with {len(tools)} sub-agents (local + remote)",
    )

