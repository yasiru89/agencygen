"""
A2A (Agent-to-Agent) helpers for exposing and connecting agents.
"""

from google.adk.agents import LlmAgent


def create_a2a_app(
    agent: LlmAgent,
    app_name: str = "agency_gen_a2a",
):
    """
    Create an A2A (Agent-to-Agent) application to expose an agent over the network.
    """
    try:
        from google.adk.a2a import A2aServer
    except ImportError:
        raise ImportError(
            "A2A support requires google-adk with A2A dependencies. "
            "Install with: pip install google-adk[a2a]"
        )

    a2a_server = A2aServer(
        agent=agent,
        app_name=app_name,
    )
    return a2a_server.app


def connect_to_remote_agent(url: str):
    """
    Connect to a remote agent exposed via A2A protocol.
    """
    try:
        from google.adk.a2a import RemoteA2aAgent
    except ImportError:
        raise ImportError(
            "A2A support requires google-adk with A2A dependencies. "
            "Install with: pip install google-adk[a2a]"
        )

    return RemoteA2aAgent(url=url)

