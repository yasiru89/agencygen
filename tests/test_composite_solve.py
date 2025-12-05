"""
Test solve() with composite patterns.
"""
import asyncio
import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load API key from .env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    print(f"✅ API key loaded")

from agency_gen import solve
from agency_gen.agency_gen import _analyze_task_keywords


def test_keyword_detection():
    """Test that composite patterns are detected correctly."""
    print("\n📋 Testing keyword-based composite detection...")
    
    # Test 1: Math + Writing
    result = _analyze_task_keywords("Calculate the ROI and then write a professional report")
    print(f"  Task: 'Calculate ROI then write report'")
    print(f"  Pattern: {result['pattern']}")
    print(f"  Sub-patterns: {result.get('sub_patterns')}")
    assert result['pattern'] == 'composite', f"Expected composite, got {result['pattern']}"
    assert 'majority_voting' in result.get('sub_patterns', [])
    assert 'reflection' in result.get('sub_patterns', [])
    print("  ✅ Passed!")
    
    # Test 2: Analysis + Writing
    result = _analyze_task_keywords("First analyze the pros and cons, then write an essay")
    print(f"  Task: 'Analyze pros/cons then write essay'")
    print(f"  Pattern: {result['pattern']}")
    print(f"  Sub-patterns: {result.get('sub_patterns')}")
    assert result['pattern'] == 'composite'
    print("  ✅ Passed!")
    
    # Test 3: Simple task (should NOT be composite)
    result = _analyze_task_keywords("What is the capital of France?")
    print(f"  Task: 'Capital of France' (simple)")
    print(f"  Pattern: {result['pattern']}")
    assert result['pattern'] == 'single_agent'
    print("  ✅ Passed!")
    
    print("\n✅ All keyword detection tests passed!")


@pytest.mark.asyncio
async def test_composite_solve():
    """Test solve() with a composite task."""
    print("\n🚀 Testing solve() with composite pattern...")
    
    task = "Calculate 15 + 25 and then write a short professional summary of the result"
    print(f"  Task: '{task}'")
    
    result = await solve(task)
    
    print(f"  Pattern: {result['pattern']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Sub-patterns: {result.get('sub_patterns')}")
    print(f"  Result preview: {result['result'][:150]}...")
    
    assert result['pattern'] == 'composite', f"Expected composite, got {result['pattern']}"
    assert result.get('sub_patterns') is not None, "Expected sub_patterns for composite"
    
    print("\n✅ Composite solve test passed!")


@pytest.mark.asyncio
async def test_solve_without_composite():
    """Ensure solve() can disable composite selection."""
    print("\n🚫 Testing solve() with composite disabled...")
    
    task = "Calculate the ROI and then write a professional report"
    result = await solve(task, use_composite_patterns=False)
    
    print(f"  Pattern: {result['pattern']}")
    print(f"  Reasoning: {result['reasoning']}")
    assert result['pattern'] != 'composite', "Composite should be disabled"
    print("  ✅ Composite disabled and single pattern selected")


if __name__ == "__main__":
    # Run keyword detection tests (no API needed)
    test_keyword_detection()
    
    # Run full solve test (requires API)
    asyncio.run(test_composite_solve())
    asyncio.run(test_solve_without_composite())

