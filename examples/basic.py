"""
AgencyGen Examples - Using Google ADK
=====================================

This file shows how to use AgencyGen with Google's Agent Development Kit.

Prerequisites:
    pip install google-adk
    Create a .env file with: GOOGLE_API_KEY=your_key_here

ADK Documentation: https://google.github.io/adk-docs/
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key from .env file
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value
    print(f"✅ API key loaded from .env")
else:
    print("⚠️  No .env file found. Make sure GOOGLE_API_KEY is set.")

# Import ADK components
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import our AgencyGen components
from agency_gen import (
    solve,  # The easiest way!
    AgencyGen,
    create_single_agent,
    create_voting_agents,
    create_debate_agents,
    create_sequential_agent,
    run_agent,
)


# =============================================================================
# HELPER: Create runner and run agent
# =============================================================================

async def run_with_runner(agent, query: str, app_name: str = "demo"):
    """Helper to run an agent with proper session setup."""
    session_service = InMemorySessionService()
    session_id = f"session_{id(agent)}"
    
    await session_service.create_session(
        app_name=app_name,
        user_id="user",
        session_id=session_id
    )
    
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
    ):
        if hasattr(event, 'content') and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    response += part.text
    
    return response


# =============================================================================
# EXAMPLE 0: The Easiest Way - solve()
# =============================================================================

async def example_solve():
    """Just describe your task and get it done!"""
    print("\n" + "="*60)
    print("  EXAMPLE 0: solve() - The Easiest Way!")
    print("="*60)
    
    print("\n📝 Task: What is 15% of 240?")
    result = await solve("What is 15% of 240?")
    print(f"   Pattern: {result['pattern']} ({result['reasoning']})")
    print(f"   Result: {result['result'][:100]}")
    
    print("\n📝 Task: Write a one-sentence thank you note")
    result = await solve("Write a one-sentence thank you note for a birthday gift")
    print(f"   Pattern: {result['pattern']} ({result['reasoning']})")
    print(f"   Result: {result['result'][:200]}")


# =============================================================================
# EXAMPLE 1: Single Agent
# =============================================================================

async def example_single_agent():
    """Create and use a simple single agent."""
    print("\n" + "="*60)
    print("  EXAMPLE 1: Single Agent (Translator)")
    print("="*60)
    
    # Create a translator agent
    agent = create_single_agent(
        name="translator",
        instruction="You are a Spanish translator. Translate the text to Spanish. Only output the translation."
    )
    
    query = "Good morning! How can I help you today?"
    print(f"\n📝 Input: {query}")
    
    response = await run_with_runner(agent, query)
    print(f"🤖 Output: {response}")


# =============================================================================
# EXAMPLE 2: Majority Voting
# =============================================================================

async def example_voting():
    """Use majority voting for reliable answers."""
    print("\n" + "="*60)
    print("  EXAMPLE 2: Majority Voting (Math)")
    print("="*60)
    
    # Create voting agents
    voting = create_voting_agents(
        name="math_solver",
        instruction="Solve the math problem. Give ONLY the final number, nothing else.",
        num_voters=3
    )
    
    query = "What is 17 × 23?"
    print(f"\n📝 Question: {query}")
    print(f"🗳️  Running {voting['num_voters']} voters...\n")
    
    # Run each voter
    responses = []
    for i, voter in enumerate(voting['voters']):
        response = await run_with_runner(voter, query, f"voter_{i}")
        responses.append(response.strip())
        print(f"   Voter {i+1}: {response.strip()}")
    
    # Aggregate votes
    final = voting['aggregate'](responses)
    print(f"\n✅ Final Answer: {final}")


# =============================================================================
# EXAMPLE 3: Sequential Pipeline
# =============================================================================

async def example_sequential():
    """Use a sequential pipeline for multi-step tasks."""
    print("\n" + "="*60)
    print("  EXAMPLE 3: Sequential Pipeline")
    print("="*60)
    
    # Create a research pipeline
    pipeline = create_sequential_agent(
        name="research_pipeline",
        steps=[
            {
                "name": "researcher",
                "instruction": "List 3 key facts about the topic. Be concise."
            },
            {
                "name": "summarizer", 
                "instruction": "Summarize the facts into one paragraph."
            },
        ]
    )
    
    query = "Benefits of regular exercise"
    print(f"\n📝 Topic: {query}")
    
    response = await run_with_runner(pipeline, query)
    print(f"\n🤖 Output:\n{response}")


# =============================================================================
# EXAMPLE 4: AgencyGen Meta-Agent
# =============================================================================

async def example_agency_gen():
    """Use AgencyGen to recommend the best agent type."""
    print("\n" + "="*60)
    print("  EXAMPLE 4: AgencyGen Meta-Agent")
    print("="*60)
    
    query = "I need to reliably solve math problems. What kind of agent should I create?"
    print(f"\n📝 Question: {query}")
    
    response = await run_with_runner(AgencyGen, query, "agency_gen")
    print(f"\n🤖 AgencyGen says:\n{response}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all examples."""
    print("\n" + "🏭 "*15)
    print("     AGENCYGEN DEMO")
    print("     Using Google ADK")
    print("🏭 "*15)
    
    # Start with the easiest way
    await example_solve()
    
    await example_single_agent()
    await example_voting()
    await example_sequential()
    await example_agency_gen()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
