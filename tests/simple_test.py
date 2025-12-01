"""
Simple test of AgencyGen with Google ADK
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

# Now import ADK (after setting the key)
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our AgencyGen
from agency_gen import create_single_agent

# Constants
APP_NAME = "agency_gen_test"
USER_ID = "test_user"
SESSION_ID = "test_session"


async def main():
    print("\n🏭 AgencyGen Simple Test")
    print("="*50)
    
    # Create a simple agent
    print("\n1. Creating a simple translator agent...")
    agent = create_single_agent(
        name="translator",
        instruction="You are a Spanish translator. Translate the user's text to Spanish. Only output the translation."
    )
    print(f"   ✅ Created: {agent.name}")
    
    # Create session service and session
    print("\n2. Setting up session...")
    session_service = InMemorySessionService()
    
    # Create a session first
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"   ✅ Session created: {session.id}")
    
    # Create runner
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # Run the agent
    print("\n3. Running the agent...")
    query = "Hello, how are you today?"
    print(f"   Input: {query}")
    print("   Output: ", end="")
    
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
    ):
        if hasattr(event, 'content') and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    print(part.text, end="", flush=True)
    
    print("\n\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
