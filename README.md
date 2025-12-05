# 🏭 AgencyGen - An Agent System Generator

> A "meta-agent" that creates task-specific agents using [Google ADK](https://google.github.io/adk-docs/).

## 🎯 What is AgencyGen?

AgencyGen is an **agent that builds other agents**. Just describe your task and it figures out the best approach automatically.

```python
from agency_gen import solve
import asyncio

# Just describe what you need - AgencyGen handles the rest!
result = asyncio.run(solve("What is 25% of 180?"))
print(result['result'])   # "45"
print(result['pattern'])  # "majority_voting" (chose voting for math accuracy)
```

That's it! Behind the scenes, AgencyGen:
1. Analyzed your task
2. Chose the best pattern (voting for math reliability)
3. Created the agents
4. Ran them and aggregated the result

## 🧠 The Core Idea

This project was built for Google's 5-Day Agents Intensive Course from Kaggle.
It was largely vibe-coded with a spec and test-driven approach using Claude Opus 4.5 and Gemini 3 Pro with Cursor.

The project is inspired by research on **Multi-Agent System Design**:

- 📄 ["Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies"](https://arxiv.org/html/2502.02533v1)
- 💬 [LLM Council](https://github.com/karpathy/llm-council) by Andrej Karpathy
- 📚 [Kaggle 5-Day Agents Course](https://www.kaggle.com/learn-guide/5-day-agents)
- 🔧 [Google ADK Documentation](https://google.github.io/adk-docs/)
- 🔄 [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/)

**Key Insight**: Different tasks need different agent structures:

| Task Type | Pattern | Why? |
|-----------|---------|------|
| Simple Q&A | Single Agent | One focused agent is enough |
| Math/Facts | Majority Voting | Multiple votes = reliability |
| Complex Analysis | Debate | Multiple perspectives help |
| Writing/Creative | Reflection | Self-critique = quality |
| Multi-step | Sequential | Pipeline of specialists |
| Long Context | RLM | Recursive decomposition |

## 📦 Installation

```bash
# Install Google ADK
pip install google-adk

# Set your API key
export GOOGLE_API_KEY=your_key_here  # Linux/Mac
set GOOGLE_API_KEY=your_key_here     # Windows

# Run the examples
python examples/basic.py
```

## 🚀 Quick Start

### The Easiest Way: `solve()`

```python
from agency_gen import solve
import asyncio

async def main():
    # Math → uses voting for reliability
    result = await solve("Calculate 15% of 240")
    print(result['result'])  # "36"
    
    # Writing → uses reflection for quality  
    result = await solve("Write a thank you email")
    print(result['result'])  # Polished email
    
    # Analysis → uses debate for perspectives
    result = await solve("Should I learn Python or JavaScript first?")
    print(result['result'])  # Balanced analysis
    
    # Use LLM to analyze task (smarter, extra API call)
    result = await solve("Help me plan my weekend", use_llm_analyzer=True)
    print(result['pattern'])  # LLM chooses the best pattern

asyncio.run(main())
```

**Two analysis modes:**
- `use_llm_analyzer=False` (default): Fast keyword matching, no extra API call
- `use_llm_analyzer=True`: LLM analyzes task and chooses pattern (smarter)

### More Control: Create Agents Directly

```python
from agency_gen import create_single_agent
from google.adk.runners import Runner
from google.genai import types

# Create an agent
translator = create_single_agent(
    name="translator",
    instruction="Translate text to Spanish accurately."
)

# Run it
runner = Runner(agent=translator, app_name="demo")
async for event in runner.run_async(
    user_id="user",
    session_id="session",
    new_message=types.Content(
        role="user",
        parts=[types.Part(text="Hello, how are you?")]
    )
):
    print(event)
```

### Create Voting Agents (for reliability)

```python
from agency_gen import create_voting_agents

# Create 3 voters
voting = create_voting_agents(
    name="math_solver",
    instruction="Solve the math problem. Give only the final number.",
    num_voters=3
)

# Run each voter, then aggregate
responses = [await run(voter, "What is 15 * 23?") for voter in voting['voters']]
final_answer = voting['aggregate'](responses)
```

### Create Debate Agents (for complex questions)

```python
from agency_gen import create_debate_agents

# Create debaters + judge
debate = create_debate_agents(
    name="ethics_debate",
    topic_instruction="Should AI make hiring decisions?",
    num_debaters=2
)

# Run debaters, then judge synthesizes
# debate['debaters'] - list of debater agents
# debate['judge'] - the judge agent
```

### Create Sequential Pipeline (for workflows)

```python
from agency_gen import create_sequential_agent

# Create a pipeline
pipeline = create_sequential_agent(
    name="research",
    steps=[
        {"name": "researcher", "instruction": "Find key facts"},
        {"name": "analyzer", "instruction": "Analyze the facts"},
        {"name": "writer", "instruction": "Write a summary"},
    ]
)

# ADK's SequentialAgent passes output from step to step
```

## 🔄 Recursive Language Models (RLM)

AgencyGen includes implementations of [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) for handling long contexts and complex reasoning.

### Two RLM Approaches

#### 1. Pattern-Based RLM (Rudimentary)

Pre-defined strategies for recursive processing:

```python
from agency_gen import create_chunking_rlm, run_chunking_rlm

# Process long documents by chunking
rlm = create_chunking_rlm(
    name="document_processor",
    instruction="Extract key insights from this document"
)

result = await run_chunking_rlm(rlm, very_long_document)
print(result['result'])
```

**Available patterns:**
- **Chunking RLM**: Split long input → process chunks → compress → aggregate
- **Iterative RLM**: Generate → critique → improve → repeat until convergence
- **Hierarchical RLM**: Decompose problem → solve sub-problems → aggregate

#### 2. True RLM with REPL (Recommended)

The model has access to a Python REPL and can recursively call itself:

```python
from agency_gen import run_rlm_repl, RLMREPLConfig
import asyncio

# Long context (could be 10M+ tokens)
context = open("server_logs.txt").read()

# Query about the context
result = asyncio.run(run_rlm_repl(
    query="How many ERROR entries are there? What are the main categories?",
    context=context,
    config=RLMREPLConfig(max_iterations=15)
))

print(result['result'])
print(f"Completed in {result['iterations']} iterations")
```

**How True RLM Works:**

1. **Context as Variable**: The huge context is stored as `context` variable in a REPL
2. **Model Writes Code**: The LLM writes Python to interact with the context
3. **Recursive `llm()` Function**: Model can call `llm(query, context_slice)` to query itself
4. **Runtime Decisions**: Model decides how to partition, search, and recurse

**Example interaction:**

```
Query: "Count errors and categorize them"

Model writes:
```python
# Peek at the context
print(context[:500])
```

Execution output:
[10:15:32] ERROR: Database timeout...
[10:16:45] ERROR: SSL validation failed...

Model writes:
```python
# Count and categorize
errors = [line for line in context.split('\n') if 'ERROR' in line]
print(f"Total errors: {len(errors)}")

# Use llm() to categorize (recursive call)
categories = llm("Categorize these errors", '\n'.join(errors[:10]))
print(categories)
```

FINAL(5 errors: 2 database, 2 security, 1 memory)
```

### RLM with MCP (Isolated Execution)

For security-sensitive contexts, use containerized code execution:

```python
from agency_gen import run_rlm_with_mcp

# Code runs in isolated Docker/Podman container
result = await run_rlm_with_mcp(
    query="Analyze this untrusted data",
    context=untrusted_content
)
```

Requires: `pip install git+https://github.com/philschmid/code-sandbox-mcp.git`

## 🔌 Advanced: Compose & Connect Agents

### Compose Multiple Patterns

```python
from agency_gen import create_composite_agent, create_single_agent

# Create specialists
math_expert = create_single_agent("math", "Solve math problems")
writer = create_single_agent("writer", "Write content")

# Compose them into one agent
tutor = create_composite_agent(
    name="tutor",
    instruction="Help students with math and writing",
    sub_agents=[math_expert, writer]
)
```

### Connect Remote Agents (A2A)

```python
from agency_gen import create_composite_with_remote

# Combine local + remote agents
coordinator = create_composite_with_remote(
    name="coordinator",
    instruction="Orchestrate local and remote services",
    local_agents=[analyzer],
    remote_urls=["http://localhost:8001"]  # A2A endpoints
)
```

## 📁 Project Structure

```
agencygen/
├── agency_gen/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Shared constants (models, defaults)
│   ├── patterns.py          # Agent creation patterns
│   ├── analysis.py          # Task analysis helpers
│   ├── runner.py            # run_agent helper
│   ├── solve.py             # solve() orchestration
│   ├── composite.py         # Composition helpers
│   ├── a2a.py               # A2A helpers
│   ├── meta_agent.py        # AgencyGen meta-agent definition
│   ├── agency_gen.py        # Thin aggregator for backwards compatibility
│   └── rlm/                  # Recursive Language Model module
│       ├── __init__.py      # RLM exports
│       ├── patterns.py      # RLM pattern definitions (chunking, iterative, hierarchical)
│       ├── runner.py        # RLM execution runners
│       ├── termination.py   # Termination strategies
│       └── repl.py          # True RLM with REPL environment
├── examples/                 # Example scripts
│   ├── basic.py             # Basic patterns demo
│   ├── a2a.py               # A2A example
│   └── rlm_repl.py          # RLM with REPL demo
├── tests/                    # Test suite
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 How It Works

AgencyGen is built entirely on [Google ADK](https://google.github.io/adk-docs/):

- **LlmAgent**: The core agent type - wraps an LLM with instructions and tools
- **SequentialAgent**: Runs sub-agents in sequence (pipeline)
- **LoopAgent**: Runs agents in a loop (reflection)
- **AgentTool**: Wraps an agent as a tool for another agent (composition!)
- **FunctionTool**: Wraps Python functions as agent tools
- **Runner**: Executes agents and handles the conversation
- **McpToolset**: Integrates MCP servers for tool access (used in RLM)

### The `solve()` Function

The simplest entry point. It:
1. Analyzes your task description
2. Picks the best pattern (voting, debate, reflection, or single)
3. Creates and runs the agent(s)
4. Returns the result + the agent for reuse

### The `AgencyGen` Meta-Agent

For more control, `AgencyGen` is an `LlmAgent` whose tools create other agents:

```python
AgencyGen = LlmAgent(
    name="AgencyGen",
    instruction="You design agents...",
    tools=[
        FunctionTool(func=create_single),
        FunctionTool(func=create_voting),
        FunctionTool(func=create_debate),
        # etc.
    ]
)
```

### The RLM Module

The `agency_gen.rlm` module provides:

**Pattern-based (rudimentary):**
- `create_chunking_rlm()` / `run_chunking_rlm()` - Process long texts via chunking
- `create_iterative_rlm()` / `run_iterative_rlm()` - Self-refine until convergence
- `create_hierarchical_rlm()` / `run_hierarchical_rlm()` - Recursive decomposition

**True RLM with REPL:**
- `run_rlm_repl()` - Main entry point for REPL-based RLM
- `run_rlm_with_mcp()` - Isolated execution via Code Sandbox MCP
- `RLMREPL` class - Full control over the RLM loop

**Termination strategies:**
- `DepthTermination` - Stop at max recursion depth
- `ConvergenceTermination` - Stop when output stabilizes
- `QualityTermination` - Stop when critic approves
- `CompositeTermination` - Combine multiple strategies

## 🎓 For Beginners

### What's Google ADK?

[Agent Development Kit](https://google.github.io/adk-docs/) is Google's framework for building AI agents. Key concepts:

- **Agent**: An autonomous entity that can reason and act
- **LlmAgent**: An agent powered by an LLM (like Gemini)
- **Tool**: A function that an agent can call
- **Runner**: Executes agents and manages conversation

### What's a "Meta-Agent"?

- A regular agent answers questions
- AgencyGen is an agent that **builds** agents
- It is "meta" because it operates one level up.

### What are the Patterns?

Patterns are recipes for how agents work together:

1. **Single**: One agent, one task
2. **Voting**: Multiple agents vote → reliability
3. **Debate**: Agents argue, judge decides → nuance
4. **Reflection**: Agent critiques itself → quality
5. **Sequential**: Pipeline of agents → workflows

### What's an RLM?

A **Recursive Language Model** is an inference strategy where:
- The model can recursively call itself or other LLMs
- It processes unbounded context by decomposition
- The model has access to a REPL to execute code
- It decides at runtime how to analyze the context

This enables processing of essentially unlimited context without "context rot".

## 📚 Learn More

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [ADK GitHub - adk-python](https://github.com/google/adk-python)
- [Kaggle 5-Day Agents Course](https://www.kaggle.com/learn-guide/5-day-agents)
- [Multi-Agent Design Paper](https://arxiv.org/html/2502.02533v1)
- [Recursive Language Models Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Code Sandbox MCP](https://www.philschmid.de/code-sandbox-mcp)
