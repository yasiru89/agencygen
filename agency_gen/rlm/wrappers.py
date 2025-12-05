"""
Wrapper agents for RLM patterns in composite agents.

When RLM patterns are used as sub-agents in a composite, they need special
handling to preserve their recursive behavior. These wrappers encapsulate
the RLM execution logic into agents that can be used as tools.
"""

from typing import Dict, Any, Callable, Awaitable, Optional

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ..config import DEFAULT_MODEL
from .repl import RLMREPLConfig, run_rlm_with_mcp


def create_rlm_wrapper_agent(
    name: str,
    rlm: Dict[str, Any],
    runner: Callable[[Dict[str, Any], str], Awaitable[Dict[str, Any]]],
    description: str,
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a wrapper agent that executes an RLM pattern when invoked.
    
    This wraps the RLM execution (chunking, iterative, or hierarchical)
    into a single agent that can be used as a sub-agent in composites.
    The agent receives input, runs the full RLM orchestration internally,
    and returns the result.
    
    Args:
        name: Name for the wrapper agent
        rlm: RLM dict from create_*_rlm()
        runner: Async function to run the RLM (run_*_rlm)
        description: Description of what this RLM does
        model: Model for the wrapper agent
        
    Returns:
        LlmAgent that internally executes the RLM pattern
    """
    # Store RLM and runner in closure for the tool
    _rlm = rlm
    _runner = runner
    
    async def execute_rlm(input_text: str) -> str:
        """Execute the RLM on the given input and return the result."""
        result = await _runner(_rlm, input_text)
        return result.get("result", str(result))
    
    execute_tool = FunctionTool(func=execute_rlm)
    
    rlm_type = rlm.get("type", "unknown")
    
    return LlmAgent(
        name=name,
        model=model,
        instruction=f"""You are an RLM ({rlm_type}) wrapper agent.

Your task: {description}

You have access to an 'execute_rlm' tool that performs the actual recursive processing.
When you receive a task, use the execute_rlm tool with the input text to process it.
The tool handles all the recursive logic internally (chunking, iteration, or decomposition).

Return the result from the tool as your response.""",
        tools=[execute_tool],
        description=f"RLM wrapper ({rlm_type}): {description}",
    )


def create_rlm_repl_wrapper_agent(
    name: str,
    config: Optional[RLMREPLConfig] = None,
    description: str = "Programmatic analysis with REPL",
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a wrapper agent that executes the true RLM with REPL when invoked.
    
    The REPL-based RLM allows the model to:
    - Write Python code to analyze context
    - Call llm() recursively on context slices
    - Perform programmatic operations (counting, searching, regex, etc.)
    
    Note: Requires MCP and Docker/Podman for sandboxed execution.
    
    Args:
        name: Name for the wrapper agent
        config: Optional REPL configuration
        description: Description of what this RLM does
        model: Model for the wrapper agent
        
    Returns:
        LlmAgent that internally executes the REPL-based RLM
    """
    _config = config or RLMREPLConfig(model=model)
    
    async def execute_repl_rlm(query: str, context: str = "") -> str:
        """
        Execute the REPL-based RLM on the given query and context.
        
        Args:
            query: The question or task to perform
            context: The text context to analyze (if empty, uses query as context)
        """
        actual_context = context if context else query
        result = await run_rlm_with_mcp(
            query=query,
            context=actual_context,
            config=_config,
            name=f"{name}_repl",
        )
        return result.get("result", str(result))
    
    execute_tool = FunctionTool(func=execute_repl_rlm)
    
    return LlmAgent(
        name=name,
        model=model,
        instruction=f"""You are an RLM (REPL) wrapper agent with programmatic analysis capabilities.

Your task: {description}

You have access to an 'execute_repl_rlm' tool that can:
- Write and execute Python code to analyze text
- Count occurrences, search for patterns, use regex
- Recursively call an LLM on context slices
- Perform any programmatic text analysis

When you receive a task, use the execute_repl_rlm tool:
- query: The question or analysis to perform
- context: The text to analyze (optional, defaults to query)

The tool handles all the REPL execution and recursive LLM calls internally.

Return the result from the tool as your response.""",
        tools=[execute_tool],
        description=f"RLM REPL wrapper: {description}",
    )
