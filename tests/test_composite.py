"""
Test Composite Patterns - Composing Multiple Agent Patterns
============================================================

This demonstrates how to compose multiple agent patterns together
using ADK's AgentTool. A composite agent can orchestrate:
- Multiple local pattern-based agents
- Remote A2A agents (black boxes)
"""

import asyncio
import os

# Load API key
with open(".env", "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            os.environ[key] = value

print(f"✅ API key loaded: {os.environ.get('GOOGLE_API_KEY', 'NOT FOUND')[:20]}...")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agency_gen import (
    create_single_agent,
    create_voting_agents,
    create_reflection_agent,
    create_agent_tool,
    create_composite_agent,
)

APP_NAME = "composite_test"
USER_ID = "test_user"


async def run_agent(agent, query: str, session_id: str, session_service) -> str:
    """Helper to run an agent."""
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    
    response_parts = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)])
    ):
        if hasattr(event, 'content') and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    response_parts.append(part.text)
    
    return "".join(response_parts)


async def test_composite_agent():
    """
    Test a composite agent that orchestrates multiple specialists.
    
    Architecture:
        Tutor (composite)
        ├── math_expert (single agent)
        └── writing_expert (single agent)
    """
    print("\n" + "="*60)
    print("  TEST 1: COMPOSITE AGENT (Multiple Specialists)")
    print("="*60)
    
    # Create specialist agents
    print("\n1. Creating specialist agents...")
    
    math_expert = create_single_agent(
        name="math_expert",
        instruction="""You are a math expert. Solve math problems step by step.
Always show your work and explain your reasoning."""
    )
    print(f"   ✅ Created: {math_expert.name}")
    
    writing_expert = create_single_agent(
        name="writing_expert", 
        instruction="""You are a writing expert. Help with essays, grammar, and style.
Give constructive feedback and suggestions."""
    )
    print(f"   ✅ Created: {writing_expert.name}")
    
    # Create composite agent that uses both
    print("\n2. Creating composite tutor agent...")
    tutor = create_composite_agent(
        name="tutor",
        instruction="""You are a helpful tutor who assists students.

You have access to two specialist tools:
- math_expert: For math problems and calculations
- writing_expert: For writing help and essay feedback

When a student asks a question:
1. Determine which specialist(s) to consult
2. Use the appropriate tool(s)
3. Synthesize a helpful response

Be encouraging and educational!""",
        sub_agents=[math_expert, writing_expert]
    )
    print(f"   ✅ Created composite: {tutor.name}")
    print(f"   Sub-agents: {[t.agent.name for t in tutor.tools]}")
    
    # Test the composite agent
    print("\n3. Testing composite agent...")
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="composite_test"
    )
    
    # Math question - should use math_expert
    query = "What is 15% of 240?"
    print(f"\n   📝 Query: {query}")
    response = await run_agent(tutor, query, "composite_test", session_service)
    print(f"   🎓 Tutor: {response[:300]}...")


async def test_pattern_composition():
    """
    Test composing different PATTERNS together.
    
    Architecture:
        Research Assistant (composite)
        ├── fact_checker (voting pattern - for reliability)
        └── writer (reflection pattern - for quality)
    """
    print("\n" + "="*60)
    print("  TEST 2: PATTERN COMPOSITION (Voting + Reflection)")
    print("="*60)
    
    # Create a voting-based fact checker (for reliable facts)
    print("\n1. Creating voting-based fact checker...")
    voting = create_voting_agents(
        name="fact_checker",
        instruction="Answer factual questions accurately and concisely.",
        num_voters=3
    )
    # Use the first voter as representative (in practice, you'd run all and aggregate)
    fact_checker = voting['voters'][0]
    print(f"   ✅ Created fact_checker (voting pattern, {voting['num_voters']} voters)")
    
    # Create a reflection-based writer (for quality writing)
    print("\n2. Creating reflection-based writer...")
    reflection = create_reflection_agent(
        name="writer",
        task_instruction="Write clear, engaging content"
    )
    writer = reflection['worker']
    print(f"   ✅ Created writer (reflection pattern)")
    
    # Compose them into a research assistant
    print("\n3. Creating composite research assistant...")
    research_assistant = create_composite_agent(
        name="research_assistant",
        instruction="""You are a research assistant that produces high-quality reports.

You have two specialist tools:
- fact_checker: Verifies facts with high reliability (uses voting)
- writer: Produces polished, well-written content (uses reflection)

For research tasks:
1. Use fact_checker to verify key facts
2. Use writer to produce well-written content
3. Combine into a coherent response""",
        sub_agents=[fact_checker, writer]
    )
    print(f"   ✅ Created composite: {research_assistant.name}")
    
    # Test it
    print("\n4. Testing pattern composition...")
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="pattern_test"
    )
    
    query = "What is the capital of France and write a brief description of it."
    print(f"\n   📝 Query: {query}")
    response = await run_agent(research_assistant, query, "pattern_test", session_service)
    print(f"   📚 Research Assistant: {response[:400]}...")


async def test_agent_as_tool():
    """
    Test using create_agent_tool directly to wrap agents.
    """
    print("\n" + "="*60)
    print("  TEST 3: AGENT AS TOOL (Direct AgentTool usage)")
    print("="*60)
    
    # Create a calculator agent
    calculator = create_single_agent(
        name="calculator",
        instruction="You are a calculator. Compute the result and return ONLY the number."
    )
    
    # Wrap it as a tool
    from agency_gen import create_agent_tool
    calc_tool = create_agent_tool(calculator)
    print(f"\n1. Created calculator agent and wrapped as tool")
    print(f"   Tool wraps agent: {calc_tool.agent.name}")
    
    # Create an agent that uses this tool
    assistant = create_single_agent(
        name="assistant",
        instruction="""You help with calculations.
Use the calculator tool for any math operations."""
    )
    assistant.tools = [calc_tool]
    print(f"\n2. Created assistant with calculator tool")
    
    # Test
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="tool_test"
    )
    
    query = "What is 123 + 456?"
    print(f"\n3. Query: {query}")
    response = await run_agent(assistant, query, "tool_test", session_service)
    print(f"   Response: {response}")


async def main():
    print("\n🔧 " + "="*50 + " 🔧")
    print("     COMPOSITE PATTERNS TEST")
    print("     Composing Multiple Agent Patterns")
    print("🔧 " + "="*50 + " 🔧")
    
    await test_composite_agent()
    await test_pattern_composition()
    await test_agent_as_tool()
    
    print("\n" + "="*60)
    print("✅ All composite pattern tests completed!")
    print("="*60)
    print("""
Key Takeaways:
1. create_agent_tool() wraps any agent as a tool
2. create_composite_agent() creates an orchestrator with sub-agents
3. Different patterns (voting, reflection) can be composed
4. create_composite_with_remote() adds A2A agents too!
""")


if __name__ == "__main__":
    asyncio.run(main())

