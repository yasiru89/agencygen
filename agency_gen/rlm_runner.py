"""
Runtime execution for Recursive Language Model (RLM) patterns.

Provides async functions to execute the different RLM types:
- run_chunking_rlm: Process long inputs via chunking
- run_iterative_rlm: Self-refine until convergence
- run_hierarchical_rlm: Recursive problem decomposition
"""

import json
from typing import Dict, Any, List, Optional

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService

from .runner import run_agent
from .termination import RLMState, TerminationStrategy
from .rlm import RLMConfig, create_chunking_rlm, create_iterative_rlm, create_hierarchical_rlm


async def run_chunking_rlm(
    rlm: Dict[str, Any],
    input_text: str,
    app_name: str = "rlm_chunking",
    user_id: str = "user",
) -> Dict[str, Any]:
    """
    Execute a chunking RLM on the given input.
    
    Args:
        rlm: RLM dict from create_chunking_rlm()
        input_text: The full text to process
        app_name: App name for sessions
        user_id: User ID for sessions
        
    Returns:
        Dict with:
        - result: Final aggregated result
        - chunks: Number of chunks processed
        - chunk_results: Results from each chunk
        - compressed_context: Compressed intermediate results
    """
    chunk_fn = rlm["chunk_fn"]
    worker = rlm["worker"]
    compressor = rlm["compressor"]
    aggregator = rlm["aggregator"]
    config: RLMConfig = rlm["config"]
    
    # Split input into chunks
    chunks = chunk_fn(input_text)
    
    # Initialize state
    state = RLMState(
        total_chunks=len(chunks),
        max_depth=config.max_depth,
    )
    
    session_service = InMemorySessionService()
    chunk_results: List[str] = []
    compressed_results: List[str] = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        state.chunks_processed = i
        
        # Create session for this chunk
        session_id = f"chunk_{i}"
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        
        # Process chunk
        prompt = f"Process this chunk ({i+1}/{len(chunks)}):\n\n{chunk}"
        result = await run_agent(worker, prompt, app_name, user_id, session_id)
        chunk_results.append(result)
        state.add_output(result)
        
        # Compress if not the last chunk (to manage context size)
        if i < len(chunks) - 1:
            compress_session = f"compress_{i}"
            await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=compress_session,
            )
            compressed = await run_agent(
                compressor,
                f"Compress this while keeping key info:\n\n{result}",
                app_name,
                user_id,
                compress_session,
            )
            compressed_results.append(compressed)
    
    state.chunks_processed = len(chunks)
    
    # Aggregate all results
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id="aggregate",
    )
    
    # Build aggregation prompt with compressed context
    if compressed_results:
        context = "\n\n---\n\n".join(compressed_results)
        aggregation_prompt = f"""Previous chunks (compressed):
{context}

Final chunk result:
{chunk_results[-1]}

Aggregate all chunk results into a final, coherent response."""
    else:
        # Only one chunk
        aggregation_prompt = f"Finalize this result:\n\n{chunk_results[0]}"
    
    final_result = await run_agent(
        aggregator,
        aggregation_prompt,
        app_name,
        user_id,
        "aggregate",
    )
    
    return {
        "result": final_result,
        "chunks": len(chunks),
        "chunk_results": chunk_results,
        "compressed_context": "\n".join(compressed_results),
        "state": state,
    }


async def run_iterative_rlm(
    rlm: Dict[str, Any],
    task: str,
    app_name: str = "rlm_iterative",
    user_id: str = "user",
) -> Dict[str, Any]:
    """
    Execute an iterative RLM on the given task.
    
    Args:
        rlm: RLM dict from create_iterative_rlm()
        task: The task to complete
        app_name: App name for sessions
        user_id: User ID for sessions
        
    Returns:
        Dict with:
        - result: Final refined result
        - iterations: Number of iterations performed
        - history: List of (output, feedback) tuples
        - termination_reason: Why iteration stopped
    """
    worker = rlm["worker"]
    critic = rlm["critic"]
    config: RLMConfig = rlm["config"]
    termination: TerminationStrategy = rlm["termination"]
    
    state = RLMState(
        max_depth=config.max_iterations,
    )
    
    session_service = InMemorySessionService()
    history: List[Dict[str, str]] = []
    
    # Initial generation
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id="iter_0_work",
    )
    
    current_output = await run_agent(
        worker,
        task,
        app_name,
        user_id,
        "iter_0_work",
    )
    state.add_output(current_output)
    state.current_depth = 1
    
    # Iterative refinement loop
    for i in range(1, config.max_iterations):
        # Check termination
        if termination.should_terminate(state):
            break
        
        # Get critique
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=f"iter_{i}_critique",
        )
        
        feedback = await run_agent(
            critic,
            f"Evaluate this output:\n\n{current_output}",
            app_name,
            user_id,
            f"iter_{i}_critique",
        )
        
        history.append({
            "iteration": i,
            "output": current_output,
            "feedback": feedback,
        })
        
        # Check for approval in feedback
        if "APPROVED" in feedback.upper():
            break
        
        # Improve based on feedback
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=f"iter_{i}_improve",
        )
        
        current_output = await run_agent(
            worker,
            f"Improve based on this feedback:\n\n{feedback}\n\nYour previous output:\n\n{current_output}",
            app_name,
            user_id,
            f"iter_{i}_improve",
        )
        
        state.add_output(current_output)
        state.current_depth = i + 1
    
    return {
        "result": current_output,
        "iterations": state.current_depth,
        "history": history,
        "termination_reason": termination.reason(state),
        "state": state,
    }


async def run_hierarchical_rlm(
    rlm: Dict[str, Any],
    problem: str,
    app_name: str = "rlm_hierarchical",
    user_id: str = "user",
    current_depth: int = 0,
) -> Dict[str, Any]:
    """
    Execute a hierarchical RLM on the given problem.
    
    Args:
        rlm: RLM dict from create_hierarchical_rlm()
        problem: The problem to solve
        app_name: App name for sessions
        user_id: User ID for sessions
        current_depth: Current recursion depth (internal use)
        
    Returns:
        Dict with:
        - result: Final aggregated solution
        - depth_reached: Maximum depth reached
        - decomposition_tree: Structure of how problem was decomposed
    """
    decomposer = rlm["decomposer"]
    solver = rlm["solver"]
    aggregator = rlm["aggregator"]
    config: RLMConfig = rlm["config"]
    termination: TerminationStrategy = rlm["termination"]
    
    state = RLMState(
        current_depth=current_depth,
        max_depth=config.max_depth,
    )
    
    session_service = InMemorySessionService()
    
    # Check if we've hit max depth - solve directly
    if termination.should_terminate(state):
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=f"solve_depth_{current_depth}",
        )
        
        result = await run_agent(
            solver,
            f"Solve this problem directly (max depth reached):\n\n{problem}",
            app_name,
            user_id,
            f"solve_depth_{current_depth}",
        )
        
        return {
            "result": result,
            "depth_reached": current_depth,
            "decomposition_tree": {
                "problem": problem[:100] + "..." if len(problem) > 100 else problem,
                "type": "leaf (depth limit)",
                "solution": result[:200] + "..." if len(result) > 200 else result,
            },
        }
    
    # Try to decompose
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"decompose_{current_depth}",
    )
    
    decomposition_response = await run_agent(
        decomposer,
        f"Analyze this problem:\n\n{problem}",
        app_name,
        user_id,
        f"decompose_{current_depth}",
    )
    
    # Parse decomposition decision
    try:
        # Try to extract JSON from response
        decomp_text = decomposition_response
        if "```json" in decomp_text:
            decomp_text = decomp_text.split("```json")[1].split("```")[0]
        elif "```" in decomp_text:
            decomp_text = decomp_text.split("```")[1].split("```")[0]
        
        # Find JSON object
        start = decomp_text.find("{")
        end = decomp_text.rfind("}") + 1
        if start >= 0 and end > start:
            decomp_text = decomp_text[start:end]
        
        decomposition = json.loads(decomp_text)
        should_decompose = decomposition.get("decompose", False)
        sub_problems = decomposition.get("sub_problems", [])
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, treat as leaf problem
        should_decompose = False
        sub_problems = []
    
    if not should_decompose or not sub_problems:
        # Leaf problem - solve directly
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=f"solve_leaf_{current_depth}",
        )
        
        result = await run_agent(
            solver,
            f"Solve this problem:\n\n{problem}",
            app_name,
            user_id,
            f"solve_leaf_{current_depth}",
        )
        
        return {
            "result": result,
            "depth_reached": current_depth,
            "decomposition_tree": {
                "problem": problem[:100] + "..." if len(problem) > 100 else problem,
                "type": "leaf",
                "solution": result[:200] + "..." if len(result) > 200 else result,
            },
        }
    
    # Recursively solve sub-problems
    sub_results: List[Dict[str, Any]] = []
    max_child_depth = current_depth
    
    for i, sub_problem in enumerate(sub_problems):
        sub_result = await run_hierarchical_rlm(
            rlm,
            sub_problem,
            app_name,
            user_id,
            current_depth + 1,
        )
        sub_results.append(sub_result)
        max_child_depth = max(max_child_depth, sub_result["depth_reached"])
    
    # Aggregate sub-solutions
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=f"aggregate_{current_depth}",
    )
    
    solutions_text = "\n\n---\n\n".join([
        f"Sub-problem {i+1}: {sp}\nSolution: {sr['result']}"
        for i, (sp, sr) in enumerate(zip(sub_problems, sub_results))
    ])
    
    aggregated = await run_agent(
        aggregator,
        f"Original problem:\n{problem}\n\nSub-solutions:\n{solutions_text}\n\nAggregate into final solution:",
        app_name,
        user_id,
        f"aggregate_{current_depth}",
    )
    
    return {
        "result": aggregated,
        "depth_reached": max_child_depth,
        "decomposition_tree": {
            "problem": problem[:100] + "..." if len(problem) > 100 else problem,
            "type": "composite",
            "sub_problems": [sr["decomposition_tree"] for sr in sub_results],
        },
    }


async def run_recursive_agent(
    rlm: Dict[str, Any],
    input_text: str,
    app_name: str = "rlm",
    user_id: str = "user",
) -> Dict[str, Any]:
    """
    Execute an RLM based on its type.
    
    This is a convenience function that dispatches to the appropriate
    runner based on the RLM type.
    
    Args:
        rlm: RLM dict from create_*_rlm()
        input_text: The input to process
        app_name: App name for sessions
        user_id: User ID for sessions
        
    Returns:
        Dict with results (varies by type)
    """
    rlm_type = rlm.get("type", "iterative")
    
    runners = {
        "chunking": run_chunking_rlm,
        "iterative": run_iterative_rlm,
        "hierarchical": run_hierarchical_rlm,
    }
    
    if rlm_type not in runners:
        raise ValueError(f"Unknown RLM type: {rlm_type}")
    
    return await runners[rlm_type](rlm, input_text, app_name, user_id)

