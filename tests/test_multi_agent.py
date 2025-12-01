"""
Multi-Agent Test - Majority Voting Example
==========================================

This tests the majority voting pattern where multiple agents
answer the same question and we pick the most common answer.

Great for reliability on math/factual questions!
"""

import asyncio
import os

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
from agency_gen import create_voting_agents, create_debate_agents

# Constants
APP_NAME = "multi_agent_test"
USER_ID = "test_user"


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


async def test_majority_voting():
    """Test the majority voting pattern."""
    print("\n" + "="*60)
    print("  TEST 1: MAJORITY VOTING")
    print("="*60)
    
    # Create voting agents
    print("\n1. Creating 3 voter agents...")
    voting = create_voting_agents(
        name="math_solver",
        instruction="""You are a math expert. Solve the problem step by step.
At the very end, state your final answer as: FINAL ANSWER: [number]""",
        num_voters=3
    )
    print(f"   ✅ Created {voting['num_voters']} voters")
    
    # Create session service
    session_service = InMemorySessionService()
    
    # The question
    query = "What is 17 × 23?"
    print(f"\n2. Question: {query}")
    
    # Run each voter
    print("\n3. Running voters...")
    responses = []
    for i, voter in enumerate(voting['voters']):
        # Create a unique session for each voter
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=f"voter_{i}"
        )
        
        response = await run_agent(voter, query, f"voter_{i}", session_service)
        responses.append(response)
        
        # Extract just the final answer for display
        if "FINAL ANSWER:" in response.upper():
            answer = response.upper().split("FINAL ANSWER:")[-1].strip()[:20]
        else:
            answer = response[-50:]
        print(f"   Voter {i+1}: ...{answer}")
    
    # Aggregate votes
    print("\n4. Aggregating votes...")
    
    def extract_answer(text):
        """Extract the FINAL ANSWER from response."""
        text_upper = text.upper()
        if "FINAL ANSWER:" in text_upper:
            return text_upper.split("FINAL ANSWER:")[-1].strip().split()[0]
        # Try to find a number at the end
        words = text.split()
        for word in reversed(words):
            cleaned = word.strip(".,!?")
            if cleaned.isdigit():
                return cleaned
        return text[-20:]
    
    answers = [extract_answer(r) for r in responses]
    print(f"   Extracted answers: {answers}")
    
    winner = voting['aggregate'](answers)
    print(f"\n✅ MAJORITY ANSWER: {winner}")
    print(f"   (Correct answer is 391)")


async def test_debate():
    """Test the debate pattern."""
    print("\n" + "="*60)
    print("  TEST 2: DEBATE PATTERN")
    print("="*60)
    
    # Create debate agents
    print("\n1. Creating debate agents...")
    debate = create_debate_agents(
        name="tech_debate",
        topic_instruction="Is Python or JavaScript better for beginners learning to code?",
        num_debaters=2
    )
    print(f"   ✅ Created {debate['num_debaters']} debaters + 1 judge")
    
    # Create session service
    session_service = InMemorySessionService()
    
    # The question
    query = "Which language should a complete beginner learn first: Python or JavaScript? Give your argument in 2-3 sentences."
    print(f"\n2. Topic: {query}")
    
    # Run debaters
    print("\n3. Debaters arguing...")
    arguments = []
    for i, debater in enumerate(debate['debaters']):
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=f"debater_{i}"
        )
        
        response = await run_agent(debater, query, f"debater_{i}", session_service)
        arguments.append(f"Debater {i+1}: {response}")
        print(f"\n   💬 Debater {i+1}:")
        print(f"   {response[:300]}...")
    
    # Run judge
    print("\n4. Judge synthesizing...")
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="judge"
    )
    
    judge_query = f"""Here are the arguments from the debate:

{chr(10).join(arguments)}

Please provide your final verdict in 2-3 sentences. Which argument is more compelling and why?"""
    
    verdict = await run_agent(debate['judge'], judge_query, "judge", session_service)
    print(f"\n   ⚖️ Judge's Verdict:")
    print(f"   {verdict}")


async def main():
    print("\n🏭 AgencyGen Multi-Agent Tests")
    print("="*60)
    
    await test_majority_voting()
    await test_debate()
    
    print("\n" + "="*60)
    print("✅ All multi-agent tests completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

