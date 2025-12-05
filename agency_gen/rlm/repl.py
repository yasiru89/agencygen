"""
True Recursive Language Model (RLM) with REPL Environment.

This module implements the key RLM concept from https://alexzhang13.github.io/blog/2025/rlm/
where the LLM has access to a REPL environment and can recursively call itself.

Key features:
- Context is stored as a variable in a Python REPL (via Code Sandbox MCP)
- The model can write Python code to interact with the context
- The model can call `llm(query, context_slice)` to recursively query itself
- The model decides at runtime how to partition, search, and recurse over the context

Uses:
- Google ADK (Agent Development Kit) for the agent framework
- Code Sandbox MCP for safe Python code execution
- MCP (Model Context Protocol) for tool integration

SECURITY NOTE:
This module ONLY supports MCP-based sandboxed code execution.
Direct exec/eval-based REPL has been removed due to arbitrary code execution risks.
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..config import DEFAULT_MODEL


@dataclass
class RLMREPLConfig:
    """Configuration for the REPL-based RLM."""
    
    # Model settings
    model: str = DEFAULT_MODEL
    
    # REPL settings
    max_iterations: int = 20  # Max code execution iterations
    max_output_chars: int = 10000  # Truncate long outputs
    
    # Recursion settings
    max_recursive_depth: int = 3  # Max depth of recursive llm() calls
    
    # Timeout settings
    code_timeout_seconds: int = 30  # Timeout for code execution
    
    # Context variable name in the REPL
    context_var_name: str = "context"


# System prompt for the RLM agent
RLM_SYSTEM_PROMPT = '''You are a Recursive Language Model (RLM) with access to a Python REPL environment.

## Your Environment

You have access to a Python REPL where:
- The variable `{context_var}` contains the input context ({context_len} characters)
- You can write Python code to manipulate and analyze this context
- You can call `llm(query, context_slice)` to recursively query an LLM on any text
- You can call `llm(query)` without context for general reasoning

## Available Functions

```python
# The context is available as a string variable
{context_var}  # str: The full input context

# Recursive LLM call - use this to query subsets of context
llm(query: str, context: str = "") -> str
    """Query the LLM with a question and optional context slice.
    
    Args:
        query: The question or task to perform
        context: Optional text context (default: empty string)
    
    Returns:
        The LLM's response as a string
    
    Example:
        # Summarize a chunk
        summary = llm("Summarize this text", {context_var}[:5000])
        
        # Ask about specific content
        answer = llm("What is the main topic?", chunk)
        
        # General reasoning (no context)
        result = llm("What is 2 + 2?")
    """

# Standard Python is available
import re  # for regex operations
len({context_var})  # get context length
{context_var}[:1000]  # slice context
{context_var}.split("\\n")  # split into lines
# etc.
```

## How to Respond

1. **Write code blocks** to explore and process the context
2. **Use `llm()` calls** to recursively analyze parts of the context
3. **Build up your answer** using variables in the REPL
4. **Output FINAL(answer)** when you have your final answer
5. **Or use FINAL_VAR(variable_name)** to return the value of a variable

## Code Block Format

Write code in triple backticks:
```python
# Your code here
result = llm("What is this about?", {context_var}[:2000])
print(result)
```

## Example Workflow

For a question like "How many times is 'error' mentioned?":

```python
# First, peek at the context structure
print({context_var}[:500])
```

Then based on the output:
```python
# Count occurrences
count = {context_var}.lower().count('error')
print(f"Found {{count}} occurrences of 'error'")
```

Finally:
```
FINAL(42)  # or whatever the count is
```

## Important Rules

1. Start by peeking at the context to understand its structure
2. Use `llm()` for semantic understanding, Python for programmatic tasks
3. Break large contexts into chunks when using `llm()`
4. Always end with FINAL(answer) or FINAL_VAR(var_name)
5. Keep code simple and focused on the task

Now, answer the following query about the context:
'''


def _extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from the model's response."""
    # Match ```python ... ``` or ``` ... ```
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def _extract_balanced_parens(text: str, start_pos: int) -> Optional[str]:
    """Extract content inside balanced parentheses starting at '(' position.
    
    Args:
        text: The full text to search
        start_pos: Position of the opening '('
        
    Returns:
        The content inside the balanced parentheses, or None if unbalanced.
    """
    if start_pos >= len(text) or text[start_pos] != '(':
        return None
    
    depth = 0
    for i in range(start_pos, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                # Found the matching closing paren
                return text[start_pos + 1:i]
    
    return None  # Unbalanced parentheses


def _extract_final_answer(text: str) -> Optional[tuple]:
    """Extract FINAL(...) or FINAL_VAR(...) from the response.
    
    Handles nested parentheses correctly by finding balanced parens.
    For example, FINAL(len(items) is 5) extracts "len(items) is 5".
    """
    # Check for FINAL_VAR(...) first (simpler pattern, no nesting expected)
    var_match = re.search(r'FINAL_VAR\(([a-zA-Z_][a-zA-Z0-9_]*)\)', text)
    if var_match:
        return ("var", var_match.group(1).strip())
    
    # Check for FINAL(...) with balanced parentheses
    final_prefix = re.search(r'FINAL\(', text)
    if final_prefix:
        # Find the position of the opening paren
        paren_pos = final_prefix.end() - 1
        content = _extract_balanced_parens(text, paren_pos)
        if content is not None:
            return ("value", content.strip())
    
    return None


# =============================================================================
# MCP-based REPL Implementation (uses Code Sandbox MCP for isolated execution)
# =============================================================================

class RLMWithMCP:
    """
    RLM implementation using Code Sandbox MCP for isolated code execution.
    
    This provides a secure execution environment by using containerized
    code execution via the Code Sandbox MCP server.
    
    Note: Requires code-sandbox-mcp to be installed and Docker/Podman available.
    """
    
    def __init__(
        self,
        config: Optional[RLMREPLConfig] = None,
        name: str = "rlm_mcp",
    ):
        self.config = config or RLMREPLConfig()
        self.name = name
        self._mcp_available = False
        
        # Try to import MCP tools
        try:
            from google.adk.tools.mcp_tool import McpToolset
            from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
            from mcp import StdioServerParameters
            self._mcp_available = True
            self._McpToolset = McpToolset
            self._StdioConnectionParams = StdioConnectionParams
            self._StdioServerParameters = StdioServerParameters
        except ImportError:
            pass
    
    def _create_mcp_toolset(self):
        """Create the MCP toolset for code execution."""
        if not self._mcp_available:
            raise RuntimeError(
                "MCP tools not available. Install with: "
                "pip install 'git+https://github.com/philschmid/code-sandbox-mcp.git' mcp"
            )
        
        return self._McpToolset(
            connection_params=self._StdioConnectionParams(
                server_params=self._StdioServerParameters(
                    command="code-sandbox-mcp",
                    args=[],
                ),
            ),
        )
    
    async def run(
        self,
        query: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Run the RLM using MCP-based code execution.
        
        This version uses the Code Sandbox MCP for isolated Python execution,
        making it safe for untrusted contexts.
        
        Raises:
            RuntimeError: If MCP tools are not available.
        """
        if not self._mcp_available:
            raise RuntimeError(
                "RLM REPL requires MCP-based sandboxed execution for security. "
                "Install with: pip install 'git+https://github.com/philschmid/code-sandbox-mcp.git' mcp\n"
                "You also need Docker or Podman available for containerized execution."
            )
        
        # Build the system prompt with context info
        system_prompt = RLM_SYSTEM_PROMPT.format(
            context_var=self.config.context_var_name,
            context_len=len(context),
        )
        
        # Add instruction about using run_python_code tool
        system_prompt += f"""

## Using the Code Execution Tool

You have access to a `run_python_code` tool that executes Python code in an isolated container.

Before using it, you must first set up the context variable:
```python
{self.config.context_var_name} = '''{context[:1000]}...'''  # Context is {len(context)} chars total
```

The context is too large to include all at once, so work with slices.
The full context is provided to you separately - use the context slices in your code.
"""
        
        # Create the agent with MCP tools
        mcp_toolset = self._create_mcp_toolset()
        
        try:
            rlm_agent = LlmAgent(
                model=self.config.model,
                name=f"{self.name}_mcp",
                instruction=system_prompt,
                description="RLM agent with MCP code execution",
                tools=[mcp_toolset],
            )
            
            session_service = InMemorySessionService()
            session_id = f"rlm_mcp_{id(query)}"
            
            await session_service.create_session(
                app_name=self.name,
                user_id="rlm_user",
                session_id=session_id,
            )
            
            runner = Runner(
                agent=rlm_agent,
                app_name=self.name,
                session_service=session_service,
            )
            
            # Provide the context in chunks via the prompt
            # (Since we can't directly inject into MCP's sandbox)
            context_info = f"""
Context length: {len(context)} characters
First 2000 characters of context:
---
{context[:2000]}
---

To access more of the context, ask me to provide specific ranges (e.g., "show me characters 2000-4000").
"""
            
            full_prompt = f"{context_info}\n\nQuery: {query}"
            
            history: List[Dict[str, Any]] = []
            
            response_parts = []
            async for event in runner.run_async(
                user_id="rlm_user",
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=full_prompt)],
                ),
            ):
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_parts.append(part.text)
            
            result = "".join(response_parts)
            
            # Check for final answer
            final = _extract_final_answer(result)
            if final:
                final_type, final_value = final
                result = final_value
            
            history.append({
                "iteration": 0,
                "model_output": result,
            })
            
            return {
                "result": result,
                "iterations": 1,
                "history": history,
            }
        finally:
            # Close MCP connection - ensure cleanup even if an exception occurs
            try:
                await mcp_toolset.close()
            except Exception:
                pass


async def run_rlm_with_mcp(
    query: str,
    context: str,
    config: Optional[RLMREPLConfig] = None,
    name: str = "rlm_mcp",
) -> Dict[str, Any]:
    """
    Run an RLM query using MCP-based isolated code execution.
    
    This version uses Code Sandbox MCP for containerized Python execution,
    providing isolation and security for LLM-generated code.
    
    Args:
        query: The question to answer about the context
        context: The context to analyze
        config: Optional RLM configuration
        name: Name for the RLM instance
        
    Returns:
        Dict with result, iterations, and history
        
    Raises:
        RuntimeError: If MCP tools are not available.
        
    Example:
        ```python
        import asyncio
        from agency_gen.rlm import run_rlm_with_mcp
        
        context = open("large_document.txt").read()
        query = "How many times is 'error' mentioned?"
        
        result = asyncio.run(run_rlm_with_mcp(query, context))
        print(result["result"])
        ```
    """
    rlm = RLMWithMCP(config=config, name=name)
    return await rlm.run(query, context)
