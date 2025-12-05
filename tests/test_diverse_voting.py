"""
Test Diverse Voting - LLM Council Style
=======================================

This tests the majority voting pattern with DIVERSE models,
inspired by Karpathy's LLM Council (github.com/karpathy/llm-council).

The idea: Different models have different strengths and biases.
Using a mix of models can produce more robust results!
"""

import asyncio
import os
import pytest

# Load API key from env file
with open(".env", "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            os.environ[key] = value

print(f"✅ API key loaded: {os.environ.get('GOOGLE_API_KEY', 'NOT FOUND')[:20]}...")

# Now import ADK
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our AgencyGen
from agency_gen import (
    create_voting_agents,
    DEFAULT_COUNCIL_MODELS,
    AVAILABLE_MODELS,
)

# Constants
APP_NAME = "diverse_voting_test"
USER_ID = "test_user"

pytestmark = pytest.mark.asyncio


async def run_agent(agent, query: str, session_id: str, session_service) -> str:
    """Helper to run an agent and get the response."""
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    response_parts = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
    ):
        if hasattr(event, 'content') and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    response_parts.append(part.text)
    
    return "".join(response_parts)


async def test_diverse_voting():
    """Test voting with diverse models (LLM Council style)."""
    print("\n" + "="*60)
    print("  TEST: DIVERSE MODEL VOTING (LLM Council Style)")
    print("="*60)
    
    print(f"\n📋 Available models: {AVAILABLE_MODELS}")
    print(f"📋 Default council models: {DEFAULT_COUNCIL_MODELS}")
    
    # Create voting agents - defaults to diverse council models
    print("\n1. Creating diverse voter council...")
    voting = create_voting_agents(
        name="diverse_council",
        instruction="""You are an expert. Answer the question concisely.
At the end, state your final answer as: ANSWER: [your answer]""",
        num_voters=3,
        # models defaults to DEFAULT_COUNCIL_MODELS for diversity!
    )
    
    print(f"   ✅ Created {voting['num_voters']} voters")
    print(f"   📊 Models used: {voting['models_used']}")
    
    # Create session service
    session_service = InMemorySessionService()
    
    # The question - something where models might give slightly different answers
    query = "What is the capital of Australia?"
    print(f"\n2. Question: {query}")
    
    # Run each voter
    print("\n3. Running diverse voters...")
    responses = []
    for i, voter in enumerate(voting['voters']):
        model_name = voting['models_used'][i]
        
        # Create a unique session for each voter
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=f"voter_{i}"
        )
        
        response = await run_agent(voter, query, f"voter_{i}", session_service)
        responses.append(response)
        
        # Extract the answer
        if "ANSWER:" in response.upper():
            answer = response.upper().split("ANSWER:")[-1].strip()[:30]
        else:
            answer = response[-50:]
        print(f"   🗳️ Voter {i+1} ({model_name}): {answer}")
    
    # Aggregate votes
    print("\n4. Aggregating diverse votes...")
    
    def extract_answer(text):
        """Extract the ANSWER from response."""
        text_upper = text.upper()
        if "ANSWER:" in text_upper:
            return text_upper.split("ANSWER:")[-1].strip().split()[0]
        return text.strip().split()[-1]
    
    answers = [extract_answer(r) for r in responses]
    print(f"   Extracted answers: {answers}")
    
    winner = voting['aggregate'](answers)
    print(f"\n✅ COUNCIL DECISION: {winner}")


async def test_custom_models():
    """Test voting with custom model list."""
    print("\n" + "="*60)
    print("  TEST: CUSTOM MODEL LIST")
    print("="*60)
    
    # Create voting agents with custom models
    print("\n1. Creating voters with custom model list...")
    custom_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    
    voting = create_voting_agents(
        name="custom_council",
        instruction="""Solve the math problem. Give only the number.
ANSWER: [number]""",
        num_voters=4,  # 4 voters with 2 models = each model used twice
        models=custom_models
    )
    
    print(f"   ✅ Created {voting['num_voters']} voters")
    print(f"   📊 Models used: {voting['models_used']}")
    print(f"   (Notice: models cycle through the list)")
    
    # Create session service
    session_service = InMemorySessionService()
    
    # Math question
    query = "What is 25 × 4?"
    print(f"\n2. Question: {query}")
    
    # Run each voter
    print("\n3. Running voters...")
    responses = []
    for i, voter in enumerate(voting['voters']):
        model_name = voting['models_used'][i]
        
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=f"custom_voter_{i}"
        )
        
        response = await run_agent(voter, query, f"custom_voter_{i}", session_service)
        responses.append(response)
        
        # Extract answer
        if "ANSWER:" in response.upper():
            answer = response.upper().split("ANSWER:")[-1].strip()[:20]
        else:
            answer = response[-30:]
        print(f"   🗳️ Voter {i+1} ({model_name}): {answer}")
    
    # Aggregate
    def extract_answer(text):
        text_upper = text.upper()
        if "ANSWER:" in text_upper:
            return text_upper.split("ANSWER:")[-1].strip().split()[0]
        words = text.split()
        for word in reversed(words):
            if word.strip(".,").isdigit():
                return word.strip(".,")
        return text[-10:]
    
    answers = [extract_answer(r) for r in responses]
    winner = voting['aggregate'](answers)
    
    print(f"\n✅ ANSWER: {winner} (correct: 100)")


async def main():
    print("\n🏛️ " + "="*50 + " 🏛️")
    print("     DIVERSE VOTING TESTS (LLM Council Style)")
    print("🏛️ " + "="*50 + " 🏛️")
    print("""
Inspired by Karpathy's LLM Council:
"Different models have different strengths and biases.
Using a diverse set of models can produce more robust answers."
""")
    
    await test_diverse_voting()
    await test_custom_models()
    
    print("\n" + "="*60)
    print("✅ All diverse voting tests completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

