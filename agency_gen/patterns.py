"""
Agent creation patterns used by AgencyGen.
"""

from collections import Counter
from typing import Optional, List, Dict, Any, Union

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent

from .config import DEFAULT_MODEL, DEFAULT_COUNCIL_MODELS


def create_single_agent(
    name: str,
    instruction: str,
    description: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a single LLM agent with the given instruction.
    """
    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        description=description or f"Agent: {name}",
    )


def create_voting_agents(
    name: str,
    instruction: str,
    num_voters: int = 3,
    models: Union[str, List[str]] = DEFAULT_COUNCIL_MODELS,
) -> Dict[str, Any]:
    """
    Create multiple agents for majority voting.
    """
    model_list = [models] if isinstance(models, str) else models
    voters: List[LlmAgent] = []
    models_used: List[str] = []

    for i in range(num_voters):
        voter_model = model_list[i % len(model_list)]
        models_used.append(voter_model)

        voter = LlmAgent(
            name=f"{name}_voter_{i+1}",
            model=voter_model,
            instruction=instruction,
            description=f"Voter {i+1} of {num_voters} using {voter_model}",
        )
        voters.append(voter)

    def aggregate_votes(responses: List[str]) -> str:
        normalized = [r.strip().lower() for r in responses]
        vote_counts = Counter(normalized)
        winner, _ = vote_counts.most_common(1)[0]
        for original, norm in zip(responses, normalized):
            if norm == winner:
                return original
        return winner

    return {
        "voters": voters,
        "aggregate": aggregate_votes,
        "num_voters": num_voters,
        "models_used": models_used,
    }


def create_debate_agents(
    name: str,
    topic_instruction: str,
    num_debaters: int = 2,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Create agents for a debate with a judge.
    """
    debaters: List[LlmAgent] = []
    perspectives = [
        "Consider practical implications and real-world constraints.",
        "Focus on theoretical principles and ideal outcomes.",
        "Play devil's advocate and challenge assumptions.",
        "Synthesize different viewpoints and find common ground.",
    ]

    for i in range(num_debaters):
        perspective = perspectives[i % len(perspectives)]
        debater = LlmAgent(
            name=f"{name}_debater_{i+1}",
            model=model,
            instruction=f"""You are Debater {i+1} in a structured debate.

{topic_instruction}

Your perspective: {perspective}

Present clear, well-reasoned arguments. Be willing to acknowledge 
good points from others while defending your position.""",
            description=f"Debater {i+1} with perspective: {perspective}",
        )
        debaters.append(debater)

    judge = LlmAgent(
        name=f"{name}_judge",
        model=model,
        instruction=f"""You are the judge in a debate about:

{topic_instruction}

Your role:
1. Listen to all arguments fairly
2. Identify the strongest points from each side
3. Synthesize a balanced, well-reasoned conclusion
4. Explain your reasoning clearly

Be objective and thorough in your analysis.""",
        description="Judge who synthesizes the debate",
    )

    return {
        "debaters": debaters,
        "judge": judge,
        "num_debaters": num_debaters,
    }


def create_reflection_agent(
    name: str,
    task_instruction: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Create agents for self-reflection and improvement.
    """
    worker = LlmAgent(
        name=f"{name}_worker",
        model=model,
        instruction=f"""You are a skilled worker. Your task:

{task_instruction}

Do your best work. If you receive feedback, incorporate it
to improve your output. Always aim for high quality.""",
        description="Worker who completes the task",
    )

    critic = LlmAgent(
        name=f"{name}_critic",
        model=model,
        instruction=f"""You are a constructive critic. The task is:

{task_instruction}

Review the work and provide specific, actionable feedback:
1. What's good about this work
2. What could be improved  
3. Specific suggestions for improvement

If the work is excellent and needs no changes, say "APPROVED".
Be helpful but maintain high standards.""",
        description="Critic who reviews and suggests improvements",
    )

    loop = LoopAgent(
        name=f"{name}_reflection_loop",
        sub_agents=[worker, critic],
        max_iterations=3,  # Prevent infinite loops
        description="Reflection loop: work -> critique -> improve",
    )

    return {
        "worker": worker,
        "critic": critic,
        "loop": loop,
    }


def create_sequential_agent(
    name: str,
    steps: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
) -> SequentialAgent:
    """
    Create a sequential pipeline of agents.
    """
    sub_agents: List[LlmAgent] = []
    for i, step in enumerate(steps):
        agent = LlmAgent(
            name=step.get("name", f"step_{i+1}"),
            model=model,
            instruction=step.get("instruction", f"Complete step {i+1}"),
            description=f"Step {i+1}: {step.get('name', 'unnamed')}",
        )
        sub_agents.append(agent)

    return SequentialAgent(
        name=name,
        sub_agents=sub_agents,
        description=f"Sequential pipeline with {len(steps)} steps",
    )

