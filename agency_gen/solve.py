"""
High-level solve orchestration for AgencyGen.
"""

from typing import Dict, Any, List

from google.adk.sessions import InMemorySessionService

from .analysis import _analyze_task_keywords, _analyze_task_llm
from .config import DEFAULT_MODEL, DEFAULT_COUNCIL_MODELS
from .composite import create_composite_agent
from .patterns import (
    create_single_agent,
    create_voting_agents,
    create_reflection_agent,
    create_debate_agents,
)
from .runner import run_agent
from .rlm import (
    create_chunking_rlm,
    create_iterative_rlm,
    create_hierarchical_rlm,
    RLMConfig,
    run_chunking_rlm,
    run_iterative_rlm,
    run_hierarchical_rlm,
    RLMREPLConfig,
    run_rlm_with_mcp,
    create_rlm_wrapper_agent,
    create_rlm_repl_wrapper_agent,
)


async def solve(
    task: str,
    model: str = DEFAULT_MODEL,
    use_llm_analyzer: bool = False,
    use_composite_patterns: bool = True,
) -> Dict[str, Any]:
    """
    Simplest entry point: analyze task, pick pattern, build/run agent(s), return result.
    """
    if use_llm_analyzer:
        analysis = await _analyze_task_llm(task, model, allow_composite=use_composite_patterns)
    else:
        analysis = _analyze_task_keywords(task, allow_composite=use_composite_patterns)

    pattern = analysis["pattern"]
    reasoning = analysis["reasoning"]

    if not use_composite_patterns and pattern == "composite":
        fallback = (analysis.get("sub_patterns") or ["single_agent"])[0]
        pattern = fallback
        reasoning = f"Composite disabled; using {fallback} instead. Original: {analysis.get('reasoning')}"

    if pattern == "majority_voting":
        voting = create_voting_agents(
            name="solver",
            instruction=f"Complete this task:\n{task}\n\nProvide a clear, accurate answer.",
            num_voters=3,
            models=DEFAULT_COUNCIL_MODELS,
        )
        session_service = InMemorySessionService()
        responses = []
        for i, voter in enumerate(voting["voters"]):
            await session_service.create_session(
                app_name="solve",
                user_id="user",
                session_id=f"voter_{i}",
            )
            response = await run_agent(voter, task, "solve", "user", f"voter_{i}")
            responses.append(response)
        result = voting["aggregate"](responses)
        agent = voting["voters"][0]

    elif pattern == "reflection":
        reflection = create_reflection_agent(
            name="solver",
            task_instruction=task,
            model=model,
        )
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="solve", user_id="user", session_id="work"
        )
        initial = await run_agent(reflection["worker"], task, "solve", "user", "work")

        await session_service.create_session(
            app_name="solve", user_id="user", session_id="critique"
        )
        critique = await run_agent(
            reflection["critic"],
            f"Review this:\n{initial}",
            "solve",
            "user",
            "critique",
        )

        await session_service.create_session(
            app_name="solve", user_id="user", session_id="improve"
        )
        result = await run_agent(
            reflection["worker"],
            f"Improve based on feedback:\n{critique}\n\nOriginal:\n{initial}",
            "solve",
            "user",
            "improve",
        )
        agent = reflection["worker"]

    elif pattern == "debate":
        debate = create_debate_agents(
            name="solver",
            topic_instruction=task,
            num_debaters=2,
            model=model,
        )
        session_service = InMemorySessionService()
        arguments: List[str] = []
        for i, debater in enumerate(debate["debaters"]):
            await session_service.create_session(
                app_name="solve", user_id="user", session_id=f"debate_{i}"
            )
            arg = await run_agent(debater, task, "solve", "user", f"debate_{i}")
            arguments.append(f"Debater {i+1}: {arg}")

        await session_service.create_session(
            app_name="solve", user_id="user", session_id="judge"
        )
        result = await run_agent(
            debate["judge"],
            f"Judge this debate:\n\n" + "\n\n".join(arguments),
            "solve",
            "user",
            "judge",
        )
        agent = debate["judge"]

    elif pattern == "rlm_chunking":
        rlm_config = RLMConfig(model=model)
        rlm = create_chunking_rlm(
            name="solver",
            instruction=f"Process and analyze this content:\n{task}",
            config=rlm_config,
        )
        rlm_result = await run_chunking_rlm(rlm, task)
        result = rlm_result["result"]
        agent = rlm["worker"]

    elif pattern == "rlm_iterative":
        rlm_config = RLMConfig(model=model, max_iterations=5)
        rlm = create_iterative_rlm(
            name="solver",
            instruction=task,
            config=rlm_config,
        )
        rlm_result = await run_iterative_rlm(rlm, task)
        result = rlm_result["result"]
        agent = rlm["worker"]

    elif pattern == "rlm_hierarchical":
        rlm_config = RLMConfig(model=model, max_depth=3)
        rlm = create_hierarchical_rlm(
            name="solver",
            instruction=task,
            config=rlm_config,
        )
        rlm_result = await run_hierarchical_rlm(rlm, task)
        result = rlm_result["result"]
        agent = rlm["decomposer"]

    elif pattern == "rlm_repl":
        # True RLM with REPL environment - model can write code and call llm() recursively
        # Uses MCP-based sandboxed execution for security
        # Separate the query from the context if provided
        # Convention: task can be "query\n---\ncontext" or just the query with context inline
        if "\n---\n" in task:
            query, context = task.split("\n---\n", 1)
        else:
            # Treat entire task as both query and context
            query = task
            context = task
        
        repl_config = RLMREPLConfig(model=model)
        rlm_result = await run_rlm_with_mcp(
            query=query,
            context=context,
            config=repl_config,
            name="solver_repl",
        )
        result = rlm_result["result"]
        # REPL-based RLM doesn't have a persistent agent, use a placeholder
        agent = create_single_agent(
            name="solver_repl_info",
            instruction="This was solved using RLM with MCP-sandboxed REPL",
            model=model,
        )

    elif pattern == "composite":
        sub_patterns = analysis.get("sub_patterns", ["single_agent", "single_agent"])
        sub_agents = []

        for i, sub_pattern in enumerate(sub_patterns):
            if sub_pattern == "majority_voting":
                voting = create_voting_agents(
                    name=f"sub_voter_{i}",
                    instruction="Answer accurately",
                    num_voters=3,
                )
                sub_agents.append(voting["voters"][0])
            elif sub_pattern == "reflection":
                reflection = create_reflection_agent(
                    name=f"sub_writer_{i}",
                    task_instruction="Write high-quality content",
                    model=model,
                )
                sub_agents.append(reflection["worker"])
            elif sub_pattern == "debate":
                debate = create_debate_agents(
                    name=f"sub_analyst_{i}",
                    topic_instruction="Analyze from multiple perspectives",
                    num_debaters=2,
                    model=model,
                )
                sub_agents.append(debate["judge"])
            elif sub_pattern == "rlm_chunking":
                rlm_config = RLMConfig(model=model)
                rlm = create_chunking_rlm(
                    name=f"sub_chunker_{i}",
                    instruction="Process and analyze content in chunks",
                    config=rlm_config,
                )
                wrapper = create_rlm_wrapper_agent(
                    name=f"sub_chunker_{i}_wrapper",
                    rlm=rlm,
                    runner=run_chunking_rlm,
                    description="Process long content by chunking, compressing, and aggregating",
                    model=model,
                )
                sub_agents.append(wrapper)
            elif sub_pattern == "rlm_iterative":
                rlm_config = RLMConfig(model=model, max_iterations=5)
                rlm = create_iterative_rlm(
                    name=f"sub_refiner_{i}",
                    instruction="Iteratively refine and improve the output",
                    config=rlm_config,
                )
                wrapper = create_rlm_wrapper_agent(
                    name=f"sub_refiner_{i}_wrapper",
                    rlm=rlm,
                    runner=run_iterative_rlm,
                    description="Iteratively refine output through critique and improvement cycles",
                    model=model,
                )
                sub_agents.append(wrapper)
            elif sub_pattern == "rlm_hierarchical":
                rlm_config = RLMConfig(model=model, max_depth=3)
                rlm = create_hierarchical_rlm(
                    name=f"sub_decomposer_{i}",
                    instruction="Decompose and solve complex problems hierarchically",
                    config=rlm_config,
                )
                wrapper = create_rlm_wrapper_agent(
                    name=f"sub_decomposer_{i}_wrapper",
                    rlm=rlm,
                    runner=run_hierarchical_rlm,
                    description="Recursively decompose complex problems and aggregate solutions",
                    model=model,
                )
                sub_agents.append(wrapper)
            elif sub_pattern == "rlm_repl":
                repl_config = RLMREPLConfig(model=model)
                wrapper = create_rlm_repl_wrapper_agent(
                    name=f"sub_repl_{i}_wrapper",
                    config=repl_config,
                    description="Analyze content programmatically with Python code and recursive LLM calls",
                    model=model,
                )
                sub_agents.append(wrapper)
            else:
                sub_agents.append(
                    create_single_agent(
                        name=f"sub_helper_{i}",
                        instruction="Help with the task",
                        model=model,
                    )
                )

        agent = create_composite_agent(
            name="solver",
            instruction=f"""You are a composite agent that orchestrates multiple specialized sub-agents.
            
Your task: {task}

You have access to these specialized tools:
- Each sub-agent handles a specific aspect of the task
- Use them in the appropriate order to complete the full task
- Combine their outputs into a coherent final answer""",
            sub_agents=sub_agents,
            model=model,
        )
        result = await run_agent(agent, task)

    else:
        agent = create_single_agent(
            name="solver",
            instruction=f"Complete this task accurately and helpfully:\n{task}",
            model=model,
        )
        result = await run_agent(agent, task)

    return {
        "result": result,
        "agent": agent,
        "pattern": pattern,
        "reasoning": reasoning,
        "sub_patterns": analysis.get("sub_patterns"),
    }

