"""
Example: True Recursive Language Model (RLM) with REPL Environment

This demonstrates the RLM pattern from https://alexzhang13.github.io/blog/2025/rlm/
where the LLM has access to a Python REPL and can recursively call itself.

Key features:
- The context is stored as a variable in a Python REPL
- The model writes Python code to interact with the context
- The model can call llm(query, context_slice) to recursively query itself
- The model decides at runtime how to partition, search, and recurse

Setup:
    export GOOGLE_API_KEY=your_key_here
    pip install -r requirements.txt
    python rlm_repl.py
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is set
if not os.environ.get("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY environment variable not set")
    print("Please run: export GOOGLE_API_KEY=your_key_here")
    exit(1)

from agency_gen import run_rlm_repl, RLMREPLConfig


# =============================================================================
# Example 1: Simple text analysis
# =============================================================================

SAMPLE_TEXT = """
Error Log Analysis Report
=========================

Date: 2024-01-15
System: Production Server Alpha

[10:15:32] INFO: Server started successfully
[10:15:33] INFO: Database connection established
[10:16:45] WARNING: High memory usage detected (85%)
[10:17:02] ERROR: Failed to process request ID 12345 - timeout
[10:17:15] INFO: Retry successful for request ID 12345
[10:18:30] ERROR: Database query timeout on table 'users'
[10:18:45] WARNING: Falling back to cache
[10:19:00] INFO: Cache hit, serving stale data
[10:20:15] ERROR: SSL certificate validation failed for external API
[10:20:30] WARNING: Using insecure connection as fallback
[10:21:00] INFO: External API call successful
[10:22:45] ERROR: Disk space low on /var/log (95% full)
[10:23:00] WARNING: Log rotation triggered
[10:24:30] INFO: Log rotation complete, freed 2GB
[10:25:00] ERROR: Memory allocation failed for large request
[10:25:15] INFO: Request rejected, client notified
[10:30:00] INFO: Scheduled maintenance check started
[10:30:15] WARNING: 3 pending system updates available
[10:31:00] INFO: Maintenance check complete

Summary:
- Total entries: 18
- INFO: 10
- WARNING: 5  
- ERROR: 5

Critical issues requiring attention:
1. SSL certificate validation (security risk)
2. Disk space management
3. Memory allocation failures
"""


async def example_simple_analysis():
    """Example: Count and categorize log entries."""
    print("=" * 60)
    print("Example 1: Simple Log Analysis with RLM REPL")
    print("=" * 60)
    
    query = "How many ERROR entries are there, and what are the main error categories?"
    
    print(f"\nQuery: {query}")
    print(f"Context length: {len(SAMPLE_TEXT)} characters")
    print("\nRunning RLM with REPL...")
    print("-" * 40)
    
    config = RLMREPLConfig(
        max_iterations=10,
        max_recursive_depth=2,
    )
    
    result = await run_rlm_repl(query, SAMPLE_TEXT, config=config)
    
    print(f"\nFinal Answer: {result['result']}")
    print(f"Iterations: {result['iterations']}")
    print("\nExecution History:")
    for i, entry in enumerate(result['history']):
        print(f"\n--- Iteration {entry['iteration']} ---")
        if 'code_blocks' in entry and entry['code_blocks']:
            print(f"Code executed: {len(entry['code_blocks'])} block(s)")
            for j, code in enumerate(entry['code_blocks']):
                print(f"  Block {j+1}:\n    {code[:100]}..." if len(code) > 100 else f"  Block {j+1}:\n    {code}")
        if 'final_answer' in entry:
            print(f"Final answer provided: {entry['final_answer'][:100]}...")


# =============================================================================
# Example 2: Multi-hop reasoning over structured data
# =============================================================================

STRUCTURED_DATA = """
Employee Database Export
========================

Department: Engineering
------------------------
- ID: E001, Name: Alice Chen, Role: Senior Engineer, Manager: E010, Salary: 150000
- ID: E002, Name: Bob Smith, Role: Junior Engineer, Manager: E001, Salary: 85000
- ID: E003, Name: Carol White, Role: Mid Engineer, Manager: E001, Salary: 110000
- ID: E010, Name: David Park, Role: Engineering Director, Manager: E050, Salary: 200000

Department: Product
-------------------
- ID: P001, Name: Eve Johnson, Role: Product Manager, Manager: P010, Salary: 130000
- ID: P002, Name: Frank Lee, Role: Associate PM, Manager: P001, Salary: 95000
- ID: P010, Name: Grace Kim, Role: VP Product, Manager: E050, Salary: 220000

Department: Executive
---------------------
- ID: E050, Name: Henry Wilson, Role: CEO, Manager: None, Salary: 500000

Project Assignments
===================

Project Alpha (High Priority):
- Lead: E001 (Alice Chen)
- Members: E002, E003, P001
- Budget: $500,000
- Status: In Progress

Project Beta (Medium Priority):
- Lead: P001 (Eve Johnson)
- Members: E002, P002
- Budget: $200,000
- Status: Planning

Project Gamma (Low Priority):
- Lead: E003 (Carol White)
- Members: E002
- Budget: $100,000
- Status: On Hold
"""


async def example_multi_hop():
    """Example: Multi-hop reasoning requiring relationship traversal."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-hop Reasoning with RLM REPL")
    print("=" * 60)
    
    query = """
    Who is the ultimate manager (CEO) of Bob Smith? 
    List the complete management chain from Bob to the CEO.
    Also, what is the total budget of all projects Bob is working on?
    """
    
    print(f"\nQuery: {query}")
    print(f"Context length: {len(STRUCTURED_DATA)} characters")
    print("\nRunning RLM with REPL...")
    print("-" * 40)
    
    config = RLMREPLConfig(
        max_iterations=15,
        max_recursive_depth=3,
    )
    
    result = await run_rlm_repl(query, STRUCTURED_DATA, config=config)
    
    print(f"\nFinal Answer:\n{result['result']}")
    print(f"\nIterations: {result['iterations']}")


# =============================================================================
# Example 3: Large context with chunking (simulated)
# =============================================================================

def generate_large_context(num_entries: int = 500) -> str:
    """Generate a large log-like context for testing chunking behavior."""
    import random
    
    lines = ["Large Scale Log Analysis", "=" * 40, ""]
    
    levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
    components = ["auth", "database", "api", "cache", "scheduler", "network"]
    
    error_count = 0
    for i in range(num_entries):
        hour = 10 + (i // 60)
        minute = i % 60
        second = random.randint(0, 59)
        level = random.choices(levels, weights=[50, 25, 15, 10])[0]
        component = random.choice(components)
        
        if level == "ERROR":
            error_count += 1
            messages = [
                f"Failed to connect to {component} service",
                f"Timeout waiting for {component} response",
                f"Invalid data received from {component}",
                f"Authentication failed in {component}",
            ]
        elif level == "WARNING":
            messages = [
                f"High latency detected in {component}",
                f"Retry attempt for {component} operation",
                f"Resource usage high in {component}",
            ]
        else:
            messages = [
                f"{component.capitalize()} operation completed",
                f"Request processed by {component}",
                f"Health check passed for {component}",
            ]
        
        message = random.choice(messages)
        lines.append(f"[{hour:02d}:{minute:02d}:{second:02d}] {level}: [{component}] {message}")
    
    lines.append("")
    lines.append(f"Total entries: {num_entries}")
    lines.append(f"Actual ERROR count (for verification): {error_count}")
    
    return "\n".join(lines)


async def example_large_context():
    """Example: Processing a large context using RLM's ability to chunk and recurse."""
    print("\n" + "=" * 60)
    print("Example 3: Large Context Processing with RLM REPL")
    print("=" * 60)
    
    large_context = generate_large_context(500)
    
    query = "Count the total number of ERROR entries and identify which component has the most errors."
    
    print(f"\nQuery: {query}")
    print(f"Context length: {len(large_context)} characters ({len(large_context.split(chr(10)))} lines)")
    print("\nRunning RLM with REPL...")
    print("-" * 40)
    
    config = RLMREPLConfig(
        max_iterations=10,
        max_recursive_depth=2,
    )
    
    result = await run_rlm_repl(query, large_context, config=config)
    
    print(f"\nFinal Answer:\n{result['result']}")
    print(f"\nIterations: {result['iterations']}")


# =============================================================================
# Main entry point
# =============================================================================

async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("True RLM (Recursive Language Model) with REPL Examples")
    print("Based on: https://alexzhang13.github.io/blog/2025/rlm/")
    print("=" * 60)
    
    # Run examples
    await example_simple_analysis()
    await example_multi_hop()
    await example_large_context()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
