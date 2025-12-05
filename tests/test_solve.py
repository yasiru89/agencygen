"""
Test solve() - The Simplest Way to Use AgencyGen
=================================================

Just describe your task and get it done!
AgencyGen figures out the best approach automatically.

Two analysis modes:
- Keyword-based (default): Fast, no API call
- LLM-based: Smarter, uses extra API call
"""

import asyncio
import os
import pytest

# Load API key
with open(".env", "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            os.environ[key] = value

print(f"✅ API key loaded")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agency_gen import solve

pytestmark = pytest.mark.asyncio


async def test_keyword_analysis():
    """Test the default keyword-based analysis (fast)."""
    print("\n" + "="*60)
    print("  KEYWORD-BASED ANALYSIS (default, fast)")
    print("="*60)
    
    # Test 1: Math (should use voting for reliability)
    print("\n📝 Task 1: Math calculation")
    result = await solve("What is 25% of 180?")
    print(f"   Pattern: {result['pattern']}")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   Result: {result['result'][:150]}")
    
    # Test 2: Writing (should use reflection for quality)
    print("\n📝 Task 2: Writing task")
    result = await solve("Write a short thank you note")
    print(f"   Pattern: {result['pattern']}")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   Result: {result['result'][:200]}...")
    
    # Test 3: Analysis (should use debate for perspectives)
    print("\n📝 Task 3: Analysis task")
    result = await solve("Should I learn Python or JavaScript first?")
    print(f"   Pattern: {result['pattern']}")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   Result: {result['result'][:200]}...")
    
    # Test 4: Simple question (should use single agent)
    print("\n📝 Task 4: Simple question")
    result = await solve("What is the capital of Japan?")
    print(f"   Pattern: {result['pattern']}")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   Result: {result['result'][:150]}")


async def test_llm_analysis():
    """Test the LLM-based analysis (smarter)."""
    print("\n" + "="*60)
    print("  LLM-BASED ANALYSIS (use_llm_analyzer=True)")
    print("="*60)
    
    # Test a task that might be ambiguous to keywords
    tasks = [
        "Help me plan my weekend",  # Ambiguous - LLM might choose differently
        "What are the pros and cons of remote work?",  # Should be debate
        "Summarize the key points of machine learning",  # Could be single or reflection
    ]
    
    for task in tasks:
        print(f"\n📝 Task: {task}")
        result = await solve(task, use_llm_analyzer=True)
        print(f"   Pattern: {result['pattern']}")
        print(f"   Reasoning: {result['reasoning']}")
        print(f"   Result: {result['result'][:200]}...")


async def main():
    print("\n🔧 " + "="*50 + " 🔧")
    print("     SOLVE() TEST - Two Analysis Modes")
    print("🔧 " + "="*50 + " 🔧")
    
    await test_keyword_analysis()
    await test_llm_analysis()
    
    print("\n" + "="*60)
    print("✅ All solve() tests completed!")
    print("="*60)
    print("""
Summary:
- use_llm_analyzer=False (default): Fast keyword matching
- use_llm_analyzer=True: LLM chooses pattern (smarter but extra API call)
""")


if __name__ == "__main__":
    asyncio.run(main())

