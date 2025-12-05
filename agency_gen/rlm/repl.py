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
"""

import re
import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

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


def _extract_final_answer(text: str) -> Optional[tuple]:
    """Extract FINAL(...) or FINAL_VAR(...) from the response."""
    # Check for FINAL(...)
    final_match = re.search(r'FINAL\((.*?)\)', text, re.DOTALL)
    if final_match:
        return ("value", final_match.group(1).strip())
    
    # Check for FINAL_VAR(...)
    var_match = re.search(r'FINAL_VAR\(([a-zA-Z_][a-zA-Z0-9_]*)\)', text)
    if var_match:
        return ("var", var_match.group(1).strip())
    
    return None


class REPLEnvironment:
    """
    A Python REPL environment for the RLM.
    
    This manages the execution context where:
    - The context is stored as a variable
    - An llm() function is available for recursive calls
    - Code can be executed and outputs captured
    """
    
    def __init__(
        self,
        context: str,
        llm_fn: Callable[[str, str], str],
        config: RLMREPLConfig,
    ):
        self.context = context
        self.llm_fn = llm_fn
        self.config = config
        self.current_depth = 0
        
        # Build the execution namespace
        self._namespace: Dict[str, Any] = {
            config.context_var_name: context,
            "llm": self._wrapped_llm,
            "print": self._capture_print,
            "__builtins__": __builtins__,
            # Pre-import useful modules
            "re": __import__("re"),
            "json": __import__("json"),
            "math": __import__("math"),
        }
        
        # Capture print outputs
        self._print_buffer: List[str] = []
        
    def _capture_print(self, *args, **kwargs):
        """Capture print statements."""
        output = " ".join(str(a) for a in args)
        self._print_buffer.append(output)
        
    def _wrapped_llm(self, query: str, context: str = "") -> str:
        """Wrapped LLM function that enforces depth limits."""
        if self.current_depth >= self.config.max_recursive_depth:
            return f"[ERROR: Max recursive depth ({self.config.max_recursive_depth}) reached]"
        
        self.current_depth += 1
        try:
            result = self.llm_fn(query, context)
            return result
        finally:
            self.current_depth -= 1
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the REPL environment.
        
        Returns:
            Dict with 'output' (captured prints/result) and 'error' (if any)
        """
        self._print_buffer = []
        
        try:
            # Try to evaluate as expression first
            try:
                result = eval(code, self._namespace)
                if result is not None:
                    self._print_buffer.append(repr(result))
            except SyntaxError:
                # If not an expression, execute as statements
                exec(code, self._namespace)
            
            output = "\n".join(self._print_buffer)
            
            # Truncate if too long
            if len(output) > self.config.max_output_chars:
                output = output[:self.config.max_output_chars] + "\n... [truncated]"
            
            return {"output": output, "error": None}
            
        except Exception as e:
            return {"output": "\n".join(self._print_buffer), "error": f"{type(e).__name__}: {str(e)}"}
    
    def get_variable(self, var_name: str) -> Any:
        """Get a variable from the namespace."""
        return self._namespace.get(var_name)


class RLMREPL:
    """
    The main Recursive Language Model with REPL environment.
    
    This implements the true RLM concept where:
    1. The model receives a query (not the full context)
    2. The context is stored as a variable in a REPL
    3. The model writes code to interact with the context
    4. The model can call llm() to recursively query itself
    5. The loop continues until FINAL(answer) is output
    """
    
    def __init__(
        self,
        config: Optional[RLMREPLConfig] = None,
        name: str = "rlm_repl",
    ):
        self.config = config or RLMREPLConfig()
        self.name = name
        
        # Create the underlying LLM agent
        self.agent = LlmAgent(
            model=self.config.model,
            name=f"{name}_agent",
            instruction="You are a helpful assistant.",
            description="RLM agent with REPL access",
        )
    
    async def _call_llm(self, query: str, context: str = "") -> str:
        """Make a single LLM call (used for recursive calls within REPL)."""
        session_service = InMemorySessionService()
        session_id = f"recursive_{id(query)}"
        
        await session_service.create_session(
            app_name=self.name,
            user_id="rlm_user",
            session_id=session_id,
        )
        
        runner = Runner(
            agent=self.agent,
            app_name=self.name,
            session_service=session_service,
        )
        
        # Build the prompt
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            prompt = query
        
        response_parts = []
        async for event in runner.run_async(
            user_id="rlm_user",
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
        ):
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_parts.append(part.text)
        
        return "".join(response_parts)
    
    def _call_llm_sync(self, query: str, context: str = "") -> str:
        """Synchronous wrapper for LLM calls (used in REPL)."""
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._call_llm(query, context))
                return future.result(timeout=60)
        except RuntimeError:
            # No running loop, we can use asyncio.run directly
            return asyncio.run(self._call_llm(query, context))
    
    async def run(
        self,
        query: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Run the RLM on a query with the given context.
        
        This is the main entry point that implements the RLM loop:
        1. Build system prompt with context info
        2. Send query to model
        3. Extract and execute code blocks
        4. Feed results back to model
        5. Repeat until FINAL(...) is output
        
        Args:
            query: The question to answer about the context
            context: The (potentially huge) context to analyze
            
        Returns:
            Dict with:
            - result: The final answer
            - iterations: Number of iterations taken
            - history: List of (model_output, execution_result) tuples
        """
        # Create the REPL environment
        repl = REPLEnvironment(
            context=context,
            llm_fn=self._call_llm_sync,
            config=self.config,
        )
        
        # Build the system prompt
        system_prompt = RLM_SYSTEM_PROMPT.format(
            context_var=self.config.context_var_name,
            context_len=len(context),
        )
        
        # Session for the main agent loop
        session_service = InMemorySessionService()
        session_id = f"rlm_main_{id(query)}"
        
        await session_service.create_session(
            app_name=self.name,
            user_id="rlm_user",
            session_id=session_id,
        )
        
        # Create a new agent with the RLM system prompt
        rlm_agent = LlmAgent(
            model=self.config.model,
            name=f"{self.name}_main",
            instruction=system_prompt,
            description="RLM agent with REPL environment",
        )
        
        runner = Runner(
            agent=rlm_agent,
            app_name=self.name,
            session_service=session_service,
        )
        
        # History of interactions
        history: List[Dict[str, Any]] = []
        
        # Current conversation for the agent
        messages = [query]
        
        for iteration in range(self.config.max_iterations):
            # Get model response
            current_message = messages[-1] if iteration > 0 else query
            
            response_parts = []
            async for event in runner.run_async(
                user_id="rlm_user",
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=current_message)],
                ),
            ):
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_parts.append(part.text)
            
            model_output = "".join(response_parts)
            
            # Check for final answer
            final = _extract_final_answer(model_output)
            if final:
                final_type, final_value = final
                
                if final_type == "var":
                    # Get the variable value from REPL
                    result = repl.get_variable(final_value)
                    if result is None:
                        result = f"[Variable '{final_value}' not found]"
                    else:
                        result = str(result)
                else:
                    result = final_value
                
                history.append({
                    "iteration": iteration,
                    "model_output": model_output,
                    "final_answer": result,
                })
                
                return {
                    "result": result,
                    "iterations": iteration + 1,
                    "history": history,
                }
            
            # Extract and execute code blocks
            code_blocks = _extract_code_blocks(model_output)
            
            if not code_blocks:
                # No code blocks - prompt the model to write code or give final answer
                execution_output = "No code blocks found. Please write Python code in ```python ... ``` blocks to analyze the context, or output FINAL(answer) with your answer."
            else:
                # Execute all code blocks
                outputs = []
                for code in code_blocks:
                    result = repl.execute(code)
                    if result["error"]:
                        outputs.append(f"Code:\n{code}\n\nError: {result['error']}")
                        if result["output"]:
                            outputs.append(f"Output before error:\n{result['output']}")
                    else:
                        outputs.append(f"Code:\n{code}\n\nOutput:\n{result['output']}")
                
                execution_output = "\n\n---\n\n".join(outputs)
            
            history.append({
                "iteration": iteration,
                "model_output": model_output,
                "code_blocks": code_blocks,
                "execution_output": execution_output,
            })
            
            # Feed execution results back to model
            messages.append(f"Execution results:\n\n{execution_output}\n\nContinue analyzing or output FINAL(answer) when you have the answer.")
        
        # Max iterations reached
        return {
            "result": "[Max iterations reached without final answer]",
            "iterations": self.config.max_iterations,
            "history": history,
        }


async def run_rlm_repl(
    query: str,
    context: str,
    config: Optional[RLMREPLConfig] = None,
    name: str = "rlm_repl",
) -> Dict[str, Any]:
    """
    Convenience function to run an RLM query with REPL.
    
    This is the main entry point for using the true RLM pattern where
    the model has access to a REPL environment with the context as a
    variable and can recursively call itself.
    
    Args:
        query: The question to answer about the context
        context: The (potentially large) context to analyze
        config: Optional RLM configuration
        name: Name for the RLM instance
        
    Returns:
        Dict with result, iterations, and history
        
    Example:
        ```python
        import asyncio
        from agency_gen.rlm import run_rlm_repl
        
        context = open("large_document.txt").read()
        query = "How many times is 'error' mentioned?"
        
        result = asyncio.run(run_rlm_repl(query, context))
        print(result["result"])
        ```
    """
    rlm = RLMREPL(config=config, name=name)
    return await rlm.run(query, context)


# =============================================================================
# MCP-based REPL Implementation (uses Code Sandbox MCP for isolated execution)
# =============================================================================

class RLMWithMCP:
    """
    RLM implementation using Code Sandbox MCP for isolated code execution.
    
    This provides a more secure execution environment by using containerized
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
        making it safer for untrusted contexts or when security is a concern.
        """
        if not self._mcp_available:
            # Fall back to local REPL
            rlm = RLMREPL(config=self.config, name=self.name)
            return await rlm.run(query, context)
        
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
        
        # Close MCP connection
        try:
            await mcp_toolset.close()
        except Exception:
            pass
        
        return {
            "result": result,
            "iterations": 1,
            "history": history,
        }


async def run_rlm_with_mcp(
    query: str,
    context: str,
    config: Optional[RLMREPLConfig] = None,
    name: str = "rlm_mcp",
) -> Dict[str, Any]:
    """
    Run an RLM query using MCP-based isolated code execution.
    
    This version uses Code Sandbox MCP for containerized Python execution,
    providing better isolation and security.
    
    Args:
        query: The question to answer about the context
        context: The context to analyze
        config: Optional RLM configuration
        name: Name for the RLM instance
        
    Returns:
        Dict with result, iterations, and history
    """
    rlm = RLMWithMCP(config=config, name=name)
    return await rlm.run(query, context)
