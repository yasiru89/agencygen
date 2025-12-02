"""
AgencyGen - Meta-Agent using Google ADK
=======================================

This module implements AgencyGen using Google's Agent Development Kit (ADK).
ADK documentation: https://google.github.io/adk-docs/

Key ADK concepts used:
- LlmAgent: An LLM-powered agent that can use tools
- SequentialAgent: Runs sub-agents in sequence  
- LoopAgent: Runs agents in a loop with exit conditions
- FunctionTool: Wraps Python functions as agent tools

AgencyGen is an LlmAgent whose tools create other agents.
This makes it a "meta-agent" - an agent that builds agents!
"""

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools import FunctionTool, AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from collections import Counter
from typing import Optional, List, Dict, Any, Union
import asyncio


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model to use for agents
# This can be changed to any Gemini model
DEFAULT_MODEL = "gemini-2.0-flash"

# Available Gemini models for diversity in multi-agent systems
# Check available models: https://ai.google.dev/gemini-api/docs/models/gemini
AVAILABLE_MODELS = [
    "gemini-2.0-flash",           # Fast, good for most tasks (default)
    "gemini-2.0-flash-lite",      # Faster, lighter version
    "gemini-2.5-flash-preview-05-20",  # Preview of next version
]

# Default diverse model set for voting/council patterns
DEFAULT_COUNCIL_MODELS = [
    "gemini-2.0-flash",           # Main model
    "gemini-2.0-flash-lite",      # Lighter variant - different trade-offs
    "gemini-2.0-flash",           # Third voter uses main model again
    # Note: For true diversity like LLM Council, you'd want different
    # model families (GPT, Claude, Gemini) but here the focus is Gemini
]


# =============================================================================
# TOOL 1: CREATE A SINGLE AGENT
# =============================================================================

def create_single_agent(
    name: str,
    instruction: str,
    description: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> LlmAgent:
    """
    Create a single LLM agent with the given instruction.
    
    This is the simplest agent type - one LLM with a specific role.
    
    Args:
        name: Unique name for the agent (e.g., "researcher")
        instruction: What the agent should do (its system prompt)
        description: Optional description for documentation
        model: Which Gemini model to use
    
    Returns:
        An LlmAgent ready to be used
    
    Example:
        >>> agent = create_single_agent(
        ...     name="translator",
        ...     instruction="You translate text to Spanish accurately."
        ... )
        >>> # Then run it with: runner = Runner(agent=agent, app_name="app")
    """
    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        description=description or f"Agent: {name}",
    )


# =============================================================================
# TOOL 2: CREATE VOTING AGENTS (Majority Voting Pattern)
# =============================================================================

def create_voting_agents(
    name: str,
    instruction: str,
    num_voters: int = 3,
    models: Union[str, List[str]] = DEFAULT_COUNCIL_MODELS,
) -> Dict[str, Any]:
    """
    Create multiple agents for majority voting.
    
    Majority voting improves reliability by having multiple agents
    answer the same question, then picking the most common answer.
    
    Inspired by Andrej Karpathy's LLM Council (github.com/karpathy/llm-council).
    Using different models adds diversity and can improve results.
    Here Gemini models are used, but multiple families of models will be better still.
    
    Args:
        name: Base name for the voting system
        instruction: What each voter should do
        num_voters: How many agents vote (odd numbers avoid ties!)
        models: Either a single model name (str) or list of models.
                Defaults to DEFAULT_COUNCIL_MODELS for diversity.
                If fewer models than voters, cycles through the list.
    
    Returns:
        Dict with 'voters', 'aggregate', 'num_voters', 'models_used'
    
    Examples:
        # Default: uses diverse council models
        >>> voting = create_voting_agents("solver", "Solve it.", num_voters=3)
        
        # Single model for all voters
        >>> voting = create_voting_agents("solver", "Solve it.", 
        ...     models="gemini-2.0-flash")
        
        # Custom model list (cycles if fewer than num_voters)
        >>> voting = create_voting_agents("solver", "Solve it.", num_voters=5,
        ...     models=["gemini-2.0-flash", "gemini-2.0-flash-lite"])
    """
    # Normalize models to a list
    if isinstance(models, str):
        model_list = [models]
    else:
        model_list = models
    
    # Create voter agents, cycling through models
    voters = []
    models_used = []
    
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
        """
        Aggregate votes and return the majority answer.
        
        Args:
            responses: List of response strings from voters
        
        Returns:
            The most common response
        """
        # Normalize responses (lowercase, strip whitespace)
        normalized = [r.strip().lower() for r in responses]
        
        # Count votes
        vote_counts = Counter(normalized)
        
        # Get winner
        winner, _ = vote_counts.most_common(1)[0]
        
        # Return original (non-normalized) version
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


# =============================================================================
# TOOL 3: CREATE DEBATE AGENTS (Debate Pattern)
# =============================================================================

def create_debate_agents(
    name: str,
    topic_instruction: str,
    num_debaters: int = 2,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Create agents for a debate with a judge.
    
    Debate is useful for complex questions that benefit from
    multiple perspectives being explored and synthesized.
    Inspired by github.com/karpathy/llm-council and arxiv:2502.02533
    
    TODO: Perspectives are currently hardcoded.
    This can be made dynamic by taking an optional list of perspectives.

    Args:
        name: Base name for the debate system
        topic_instruction: Context about what to debate
        num_debaters: How many agents participate
        model: Which Gemini model to use
    
    Returns:
        Dict with 'debaters', 'judge', and 'orchestrate' function
    
    Example:
        >>> debate = create_debate_agents(
        ...     name="ethics_debate",
        ...     topic_instruction="Analyze the ethics of AI in healthcare",
        ...     num_debaters=3
        ... )
    """
    # Create debater agents with different perspectives
    debaters = []
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
    
    # Create judge agent
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


# =============================================================================
# TOOL 4: CREATE REFLECTION AGENT (Reflection Pattern)
# =============================================================================

def create_reflection_agent(
    name: str,
    task_instruction: str,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Create agents for self-reflection and improvement.
    
    Reflection helps produce higher quality outputs by having
    an agent critique and improve its own work.
    
    Args:
        name: Base name for the reflection system
        task_instruction: What task to complete
        model: Which Gemini model to use
    
    Returns:
        Dict with 'worker', 'critic', and a LoopAgent 'loop'
    
    Example:
        >>> reflection = create_reflection_agent(
        ...     name="writer",
        ...     task_instruction="Write a professional email"
        ... )
    """
    # Worker agent - does the actual task
    worker = LlmAgent(
        name=f"{name}_worker",
        model=model,
        instruction=f"""You are a skilled worker. Your task:

{task_instruction}

Do your best work. If you receive feedback, incorporate it
to improve your output. Always aim for high quality.""",
        description="Worker who completes the task",
    )
    
    # Critic agent - reviews and suggests improvements
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
    
    # Create a LoopAgent that iterates until approved
    # The loop continues until critic says "APPROVED"
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


# =============================================================================
# TOOL 5: CREATE SEQUENTIAL AGENT (Pipeline Pattern)
# =============================================================================

def create_sequential_agent(
    name: str,
    steps: List[Dict[str, str]],
    model: str = DEFAULT_MODEL
) -> SequentialAgent:
    """
    Create a sequential pipeline of agents.
    
    Each agent in the pipeline processes the output of the previous one.
    Great for multi-step workflows!
    
    Args:
        name: Name for the pipeline
        steps: List of dicts with 'name' and 'instruction' for each step
        model: Which Gemini model to use
    
    Returns:
        A SequentialAgent that runs steps in order
    
    Example:
        >>> pipeline = create_sequential_agent(
        ...     name="research_pipeline",
        ...     steps=[
        ...         {"name": "researcher", "instruction": "Find key facts"},
        ...         {"name": "analyzer", "instruction": "Analyze the facts"},
        ...         {"name": "writer", "instruction": "Write a summary"},
        ...     ]
        ... )
    """
    # Create an agent for each step
    sub_agents = []
    for i, step in enumerate(steps):
        agent = LlmAgent(
            name=step.get("name", f"step_{i+1}"),
            model=model,
            instruction=step.get("instruction", f"Complete step {i+1}"),
            description=f"Step {i+1}: {step.get('name', 'unnamed')}",
        )
        sub_agents.append(agent)
    
    # Wrap in SequentialAgent
    return SequentialAgent(
        name=name,
        sub_agents=sub_agents,
        description=f"Sequential pipeline with {len(steps)} steps",
    )


# =============================================================================
# HELPER: RUN AN AGENT
# =============================================================================

async def run_agent(
    agent: LlmAgent,
    query: str,
    app_name: str = "agency_gen",
    user_id: str = "user",
    session_id: str = "session"
) -> str:
    """
    Run an agent and get the response.
    
    This is a helper function that wraps ADK's Runner.
    
    Args:
        agent: The agent to run
        query: The input query/task
        app_name: Name for the application
        user_id: User ID for the session
        session_id: Session ID
    
    Returns:
        The agent's response as a string
    
    Example:
        >>> agent = create_single_agent("helper", "Be helpful")
        >>> response = await run_agent(agent, "What is Python?")
        >>> print(response)
    """
    # Create session service and session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    # Create a runner for this agent
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )
    
    # Run the agent
    # ADK uses an async generator pattern
    response_parts = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
    ):
        # Collect response parts
        if hasattr(event, 'content') and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    response_parts.append(part.text)
    
    return "".join(response_parts)


# =============================================================================
# SOLVE: One-liner to get things done
# =============================================================================

async def solve(
    task: str, 
    model: str = DEFAULT_MODEL,
    use_llm_analyzer: bool = False,
) -> Dict[str, Any]:
    """
    This is the simplest way to use AgencyGen - just describe your task
    and let it figure out the best approach!
    
    AgencyGen automatically:
    1. Analyzes your task
    2. Chooses the best agent pattern (single, voting, debate, reflection, or composite)
    3. Creates the agent (or multi-agent system for composite)
    4. Runs it on your task
    5. Returns the result
    
    Args:
        task: What you want to accomplish (natural language)
        model: Model to use (default: gemini-2.0-flash)
        use_llm_analyzer: If True, use an LLM to analyze the task (smarter
            but uses an extra API call). If False (default), use fast
            keyword-based analysis.
    
    Returns:
        Dict with:
        - 'result': The agent's response
        - 'agent': The agent that was created (for reuse)
        - 'pattern': Which pattern was chosen (single_agent, majority_voting,
          reflection, debate, or composite)
        - 'reasoning': Why that pattern was chosen
        - 'sub_patterns': (Only for composite) List of patterns combined
    
    Example:
        >>> # Fast keyword-based analysis (default)
        >>> result = await solve("What is 15% of 240?")
        >>> print(result['result'])  # "36"
        >>> print(result['pattern']) # "majority_voting"
        
        >>> # Smarter LLM-based analysis
        >>> result = await solve("Help me decide on a career", use_llm_analyzer=True)
        >>> print(result['pattern']) # LLM chooses best pattern
        
        >>> # Complex task triggers composite pattern
        >>> result = await solve("Calculate the budget then write a professional report")
        >>> print(result['pattern'])      # "composite"
        >>> print(result['sub_patterns']) # ["majority_voting", "reflection"]
    """
    # Analyze the task to determine the best pattern
    if use_llm_analyzer:
        analysis = await _analyze_task_llm(task, model)
    else:
        analysis = _analyze_task_keywords(task)
    pattern = analysis['pattern']
    reasoning = analysis['reasoning']
    
    # Create the appropriate agent based on the pattern
    if pattern == "majority_voting":
        voting = create_voting_agents(
            name="solver",
            instruction=f"Complete this task:\n{task}\n\nProvide a clear, accurate answer.",
            num_voters=3,
            models=DEFAULT_COUNCIL_MODELS,
        )
        # Run all voters and aggregate
        session_service = InMemorySessionService()
        responses = []
        for i, voter in enumerate(voting['voters']):
            await session_service.create_session(
                app_name="solve", user_id="user", session_id=f"voter_{i}"
            )
            response = await run_agent(voter, task, "solve", "user", f"voter_{i}")
            responses.append(response)
        result = voting['aggregate'](responses)
        agent = voting['voters'][0]  # Return first voter as representative
        
    elif pattern == "reflection":
        reflection = create_reflection_agent(
            name="solver",
            task_instruction=task,
            model=model,
        )
        # Run worker, then critic, then improved
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="solve", user_id="user", session_id="work"
        )
        initial = await run_agent(reflection['worker'], task, "solve", "user", "work")
        
        await session_service.create_session(
            app_name="solve", user_id="user", session_id="critique"
        )
        critique = await run_agent(
            reflection['critic'], 
            f"Review this:\n{initial}", 
            "solve", "user", "critique"
        )
        
        await session_service.create_session(
            app_name="solve", user_id="user", session_id="improve"
        )
        result = await run_agent(
            reflection['worker'],
            f"Improve based on feedback:\n{critique}\n\nOriginal:\n{initial}",
            "solve", "user", "improve"
        )
        agent = reflection['worker']
        
    elif pattern == "debate":
        debate = create_debate_agents(
            name="solver",
            topic_instruction=task,
            num_debaters=2,
            model=model,
        )
        # Run debaters then judge
        session_service = InMemorySessionService()
        arguments = []
        for i, debater in enumerate(debate['debaters']):
            await session_service.create_session(
                app_name="solve", user_id="user", session_id=f"debate_{i}"
            )
            arg = await run_agent(debater, task, "solve", "user", f"debate_{i}")
            arguments.append(f"Debater {i+1}: {arg}")
        
        await session_service.create_session(
            app_name="solve", user_id="user", session_id="judge"
        )
        result = await run_agent(
            debate['judge'],
            f"Judge this debate:\n\n" + "\n\n".join(arguments),
            "solve", "user", "judge"
        )
        agent = debate['judge']
        
    elif pattern == "composite":
        # Create sub-agents based on sub_patterns
        sub_patterns = analysis.get('sub_patterns', ['single_agent', 'single_agent'])
        sub_agents = []
        
        for i, sub_pattern in enumerate(sub_patterns):
            if sub_pattern == "majority_voting":
                # Create a voting agent (use first voter as representative)
                voting = create_voting_agents(
                    name=f"sub_voter_{i}",
                    instruction="Answer accurately",
                    num_voters=3,
                )
                sub_agents.append(voting['voters'][0])
            elif sub_pattern == "reflection":
                reflection = create_reflection_agent(
                    name=f"sub_writer_{i}",
                    task_instruction="Write high-quality content",
                    model=model,
                )
                sub_agents.append(reflection['worker'])
            elif sub_pattern == "debate":
                debate = create_debate_agents(
                    name=f"sub_analyst_{i}",
                    topic_instruction="Analyze from multiple perspectives",
                    num_debaters=2,
                    model=model,
                )
                sub_agents.append(debate['judge'])
            else:  # single_agent
                sub_agents.append(create_single_agent(
                    name=f"sub_helper_{i}",
                    instruction="Help with the task",
                    model=model,
                ))
        
        # Create composite agent orchestrating all sub-agents
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
        
    else:  # single_agent (default)
        agent = create_single_agent(
            name="solver",
            instruction=f"Complete this task accurately and helpfully:\n{task}",
            model=model,
        )
        result = await run_agent(agent, task)
    
    return {
        'result': result,
        'agent': agent,
        'pattern': pattern,
        'reasoning': reasoning,
        'sub_patterns': analysis.get('sub_patterns'),  # Include if composite
    }


def _analyze_task_keywords(task: str) -> Dict[str, Any]:
    """
    Analyze a task using keyword matching (fast, no API call).
    
    Returns pattern and reasoning based on keyword detection.
    For composite patterns, also returns sub_patterns list.
    """
    task_lower = task.lower()
    
    # Track which patterns are relevant
    detected_patterns = []
    
    # Check for reliability/accuracy needs -> voting
    reliability_keywords = ['calculate', 'math', 'compute', 'exact', 'precise',
                           'accurate', 'correct', 'factual', 'verify']
    if any(kw in task_lower for kw in reliability_keywords):
        detected_patterns.append(('majority_voting', 'accuracy/calculation'))
    
    # Check for quality/polish needs -> reflection
    quality_keywords = ['write', 'compose', 'draft', 'create', 'essay', 'email',
                       'letter', 'story', 'article', 'polish', 'professional']
    if any(kw in task_lower for kw in quality_keywords):
        detected_patterns.append(('reflection', 'quality writing'))
    
    # Check for analysis/perspective needs -> debate
    debate_keywords = ['analyze', 'compare', 'pros and cons', 'debate', 'argue',
                      'perspective', 'opinion', 'ethics', 'should']
    if any(kw in task_lower for kw in debate_keywords):
        detected_patterns.append(('debate', 'multiple perspectives'))
    
    # Check for multi-step/complex needs -> composite indicators
    composite_keywords = ['then', 'and then', 'after that', 'first', 'second',
                         'finally', 'steps', 'multi-step', 'pipeline']
    has_composite_indicators = any(kw in task_lower for kw in composite_keywords)
    
    # If multiple patterns detected or composite indicators present with 2+ patterns
    if len(detected_patterns) >= 2 or (has_composite_indicators and len(detected_patterns) >= 1):
        # If we have composite indicators but only one pattern, add single_agent for the other steps
        if len(detected_patterns) == 1:
            detected_patterns.append(('single_agent', 'general task handling'))
        
        sub_patterns = [p[0] for p in detected_patterns]
        reasons = [p[1] for p in detected_patterns]
        return {
            'pattern': 'composite',
            'reasoning': f'Complex task requiring: {", ".join(reasons)} - using composite pattern',
            'sub_patterns': sub_patterns
        }
    
    # Single pattern detected
    if len(detected_patterns) == 1:
        return {
            'pattern': detected_patterns[0][0],
            'reasoning': f'Task requires {detected_patterns[0][1]} - using {detected_patterns[0][0]}'
        }
    
    # Default to single agent
    return {
        'pattern': 'single_agent',
        'reasoning': 'Straightforward task - using single focused agent'
    }


async def _analyze_task_llm(task: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Analyze a task using an LLM (smarter, uses API call).
    
    An LlmAgent analyzes the task and chooses the best pattern.
    More nuanced than keyword matching but requires an API call.
    """
    # Create an analyzer agent
    analyzer = LlmAgent(
        name="task_analyzer",
        model=model,
        instruction="""You analyze tasks and choose the best agent pattern.

Available patterns:
- single_agent: For simple, straightforward tasks
- majority_voting: For tasks requiring accuracy/reliability (math, facts)
- reflection: For tasks requiring quality/polish (writing, creative)
- debate: For tasks benefiting from multiple perspectives (analysis, ethics)
- composite: For complex multi-step tasks requiring MULTIPLE patterns combined

For composite, also specify which sub_patterns to combine (2-3 patterns from above).

Respond with ONLY a JSON object (no markdown):
{"pattern": "pattern_name", "reasoning": "one sentence why"}

For composite patterns:
{"pattern": "composite", "reasoning": "why", "sub_patterns": ["pattern1", "pattern2"]}

Examples:
Task: "What is 15% of 240?"
{"pattern": "majority_voting", "reasoning": "Math calculation benefits from voting for accuracy"}

Task: "Write a cover letter"
{"pattern": "reflection", "reasoning": "Writing task benefits from self-critique for quality"}

Task: "Should I buy or rent a house?"
{"pattern": "debate", "reasoning": "Decision benefits from exploring multiple perspectives"}

Task: "What is the capital of France?"
{"pattern": "single_agent", "reasoning": "Simple factual question needs one focused answer"}

Task: "Calculate the ROI and then write a professional report analyzing the investment"
{"pattern": "composite", "reasoning": "Requires accurate calculation AND quality writing", "sub_patterns": ["majority_voting", "reflection"]}

Task: "First analyze the pros and cons of remote work, then write a persuasive essay"
{"pattern": "composite", "reasoning": "Needs debate for analysis AND reflection for quality writing", "sub_patterns": ["debate", "reflection"]}
"""
    )
    
    # Run the analyzer
    response = await run_agent(analyzer, f"Task: {task}")
    
    # Parse the JSON response
    import json
    try:
        # Try to extract JSON from the response
        response = response.strip()
        # Handle markdown code blocks if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        result = json.loads(response)
        
        # Validate the pattern
        valid_patterns = ['single_agent', 'majority_voting', 'reflection', 'debate', 'composite']
        if result.get('pattern') not in valid_patterns:
            result['pattern'] = 'single_agent'
            result['reasoning'] = 'LLM returned invalid pattern - defaulting to single agent'
        
        # Validate sub_patterns for composite
        if result.get('pattern') == 'composite':
            sub_patterns = result.get('sub_patterns', [])
            valid_sub = ['single_agent', 'majority_voting', 'reflection', 'debate']
            sub_patterns = [p for p in sub_patterns if p in valid_sub]
            if len(sub_patterns) < 2:
                # Not enough valid sub-patterns, fall back to single agent
                result['pattern'] = 'single_agent'
                result['reasoning'] = 'Composite needs 2+ sub-patterns - defaulting to single agent'
            else:
                result['sub_patterns'] = sub_patterns
        
        return result
    except (json.JSONDecodeError, KeyError):
        # Fallback to keyword analysis if LLM response is malformed
        return _analyze_task_keywords(task)


# =============================================================================
# A2A: EXPOSE AGENTS AS NETWORK SERVICES
# =============================================================================

def create_a2a_app(
    agent: LlmAgent,
    app_name: str = "agency_gen_a2a",
):
    """
    Create an A2A (Agent-to-Agent) application to expose an agent over the network.
    
    This allows other agents (even from different systems) to communicate
    with this agent using the A2A protocol.
    
    Based on Day 5a of the Kaggle 5-Day Agents Course:
    https://www.kaggle.com/learn-guide/5-day-agents
    
    Args:
        agent: The agent to expose via A2A
        app_name: Name for the A2A application
    
    Returns:
        An A2A application that can be run with uvicorn
    
    Example:
        >>> agent = create_single_agent("helper", "Answer questions")
        >>> app = create_a2a_app(agent)
        >>> # Run with: uvicorn module:app --port 8000
        >>> # Or programmatically: uvicorn.run(app, port=8000)
    
    To consume this agent from another system:
        >>> from google.adk.a2a import RemoteA2aAgent
        >>> remote = RemoteA2aAgent(url="http://localhost:8000")
    """
    try:
        from google.adk.a2a import A2aServer
    except ImportError:
        raise ImportError(
            "A2A support requires google-adk with A2A dependencies. "
            "Install with: pip install google-adk[a2a]"
        )
    
    # Create the A2A server wrapping our agent
    a2a_server = A2aServer(
        agent=agent,
        app_name=app_name,
    )
    
    # Return the FastAPI/Starlette app
    return a2a_server.app


def connect_to_remote_agent(url: str):
    """
    Connect to a remote agent exposed via A2A protocol.
    
    This allows your agent to communicate with agents running
    on other systems/servers.
    
    Args:
        url: The URL where the remote agent is exposed
             (e.g., "http://localhost:8000")
    
    Returns:
        A RemoteA2aAgent that can be used like a local agent
    
    Example:
        >>> # Connect to a remote product catalog agent
        >>> catalog = connect_to_remote_agent("http://localhost:8000")
        >>> 
        >>> # Use it in your agent's tools
        >>> response = await catalog.run("What products do you have?")
    """
    try:
        from google.adk.a2a import RemoteA2aAgent
    except ImportError:
        raise ImportError(
            "A2A support requires google-adk with A2A dependencies. "
            "Install with: pip install google-adk[a2a]"
        )
    
    return RemoteA2aAgent(url=url)


# =============================================================================
# COMPOSITE PATTERNS: Compose multiple agents/patterns together
# =============================================================================

def create_agent_tool(agent: LlmAgent, description: Optional[str] = None) -> AgentTool:
    """
    Wrap an agent as a tool that another agent can use.
    
    This enables composition - one agent can delegate to another!
    The sub-agent becomes a "tool" the parent agent can call.
    
    Args:
        agent: The agent to wrap as a tool
        description: Optional description (defaults to agent's description)
    
    Returns:
        An AgentTool that can be added to another agent's tools
    
    Example:
        >>> # Create specialist agents
        >>> researcher = create_single_agent("researcher", "Research topics")
        >>> writer = create_single_agent("writer", "Write content")
        >>> 
        >>> # Wrap them as tools
        >>> research_tool = create_agent_tool(researcher)
        >>> write_tool = create_agent_tool(writer)
        >>> 
        >>> # Create a coordinator that uses both
        >>> coordinator = create_single_agent(
        ...     "coordinator",
        ...     "Coordinate research and writing tasks",
        ... )
        >>> coordinator.tools = [research_tool, write_tool]
    """
    return AgentTool(agent=agent)


def create_composite_agent(
    name: str,
    instruction: str,
    sub_agents: List[LlmAgent],
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a composite agent that orchestrates multiple sub-agents.
    
    The sub-agents become tools that the composite agent can call.
    This allows composing multiple patterns into a single system.
    
    Args:
        name: Name for the composite agent
        instruction: What the composite agent should do
        sub_agents: List of agents to use as tools
        model: Model for the composite agent
    
    Returns:
        An LlmAgent with sub-agents as tools
    
    Example:
        >>> # Create pattern-based agents
        >>> voting = create_voting_agents("math", "Solve math", num_voters=3)
        >>> math_solver = voting['voters'][0]  # Use one as representative
        >>> 
        >>> reflection = create_reflection_agent("writer", "Write essays")
        >>> essay_writer = reflection['worker']
        >>> 
        >>> # Compose them
        >>> tutor = create_composite_agent(
        ...     name="tutor",
        ...     instruction="Help students with math and writing",
        ...     sub_agents=[math_solver, essay_writer]
        ... )
    """
    # Wrap each sub-agent as a tool
    tools = [AgentTool(agent=sub_agent) for sub_agent in sub_agents]
    
    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        tools=tools,
        description=f"Composite agent with {len(sub_agents)} sub-agents",
    )


def create_composite_with_remote(
    name: str,
    instruction: str,
    local_agents: Optional[List[LlmAgent]] = None,
    remote_urls: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create a composite agent that uses both local and remote (A2A) agents.
    
    This is powerful for distributed systems where some agents run
    on different servers or are "black boxes" we connect to via A2A.
    
    Args:
        name: Name for the composite agent
        instruction: What the composite agent should do
        local_agents: List of local agents to use as tools
        remote_urls: List of A2A URLs for remote agents
        model: Model for the composite agent
    
    Returns:
        An LlmAgent that can orchestrate local and remote agents
    
    Example:
        >>> # Local specialist
        >>> analyzer = create_single_agent("analyzer", "Analyze data")
        >>> 
        >>> # Remote specialists (running elsewhere via A2A)
        >>> # These could be completely different implementations!
        >>> remote_urls = [
        ...     "http://localhost:8001",  # Product catalog service
        ...     "http://localhost:8002",  # Inventory service
        ... ]
        >>> 
        >>> # Compose local + remote
        >>> coordinator = create_composite_with_remote(
        ...     name="coordinator",
        ...     instruction="Coordinate analysis with product and inventory data",
        ...     local_agents=[analyzer],
        ...     remote_urls=remote_urls
        ... )
    """
    tools = []
    
    # Add local agents as tools
    if local_agents:
        for agent in local_agents:
            tools.append(AgentTool(agent=agent))
    
    # Add remote agents as tools
    if remote_urls:
        try:
            from google.adk.a2a import RemoteA2aAgent
            for url in remote_urls:
                remote = RemoteA2aAgent(url=url)
                tools.append(AgentTool(agent=remote))
        except ImportError:
            raise ImportError(
                "A2A support requires google-adk with A2A dependencies. "
                "Install with: pip install google-adk[a2a]"
            )
    
    return LlmAgent(
        name=name,
        model=model,
        instruction=instruction,
        tools=tools,
        description=f"Composite agent with {len(tools)} sub-agents (local + remote)",
    )


# =============================================================================
# AGENCYGEN - THE META-AGENT
# =============================================================================

# Define the tools that AgencyGen can use
# These are the agent-building functions wrapped as FunctionTools

import re

def _sanitize_name(name: str) -> str:
    """
    Sanitize a name to be a valid Python identifier.
    ADK requires agent names to be valid identifiers.
    """
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    # Remove any characters that aren't alphanumeric or underscore
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    # Ensure it starts with a letter or underscore
    if name and name[0].isdigit():
        name = '_' + name
    return name.lower() or 'agent'


def _tool_create_single(
    name: str,
    instruction: str,
    description: str = ""
) -> str:
    """
    Tool: Create a single agent.
    
    Use this when the task is straightforward and can be handled
    by one focused agent.
    
    Args:
        name: Name for the agent
        instruction: What the agent should do
        description: Optional description
    
    Returns:
        Confirmation message with agent details
    """
    safe_name = _sanitize_name(name)
    agent = create_single_agent(safe_name, instruction, description)
    return f"Created single agent '{safe_name}' with instruction: {instruction[:100]}..."


def _tool_create_voting(
    name: str,
    instruction: str,
    num_voters: int = 3
) -> str:
    """
    Tool: Create a majority voting system.
    
    Use this when you need high reliability, especially for tasks
    with clear correct answers like math or factual questions.
    
    Args:
        name: Name for the voting system
        instruction: What each voter should do
        num_voters: How many agents vote (odd numbers avoid ties)
    
    Returns:
        Confirmation message with voting system details
    """
    safe_name = _sanitize_name(name)
    voting = create_voting_agents(safe_name, instruction, num_voters)
    return f"Created voting system '{safe_name}' with {num_voters} voters"


def _tool_create_debate(
    name: str,
    topic: str,
    num_debaters: int = 2
) -> str:
    """
    Tool: Create a debate system with judge.
    
    Use this for complex questions that benefit from exploring
    multiple perspectives before reaching a conclusion.
    
    Args:
        name: Name for the debate
        topic: What to debate
        num_debaters: How many debaters participate
    
    Returns:
        Confirmation message with debate system details
    """
    safe_name = _sanitize_name(name)
    debate = create_debate_agents(safe_name, topic, num_debaters)
    return f"Created debate '{safe_name}' with {num_debaters} debaters and a judge"


def _tool_create_reflection(
    name: str,
    task: str
) -> str:
    """
    Tool: Create a self-improving reflection system.
    
    Use this when the output needs to be high quality and polished,
    like writing, code, or creative work.
    
    Args:
        name: Name for the system
        task: What task to complete
    
    Returns:
        Confirmation message with reflection system details
    """
    safe_name = _sanitize_name(name)
    reflection = create_reflection_agent(safe_name, task)
    return f"Created reflection system '{safe_name}' with worker and critic"


def _tool_create_pipeline(
    name: str,
    steps_json: str
) -> str:
    """
    Tool: Create a sequential pipeline.
    
    Use this for multi-step workflows where each step builds
    on the previous one.
    
    Args:
        name: Name for the pipeline
        steps_json: JSON string of steps, each with 'name' and 'instruction'
    
    Returns:
        Confirmation message with pipeline details
    """
    import json
    safe_name = _sanitize_name(name)
    steps = json.loads(steps_json)
    # Sanitize step names too
    for step in steps:
        if 'name' in step:
            step['name'] = _sanitize_name(step['name'])
    pipeline = create_sequential_agent(safe_name, steps)
    return f"Created pipeline '{safe_name}' with {len(steps)} steps: {[s['name'] for s in steps]}"


# Create AgencyGen - the meta-agent
AgencyGen = LlmAgent(
    name="AgencyGen",
    model=DEFAULT_MODEL,
    description="A meta-agent that creates task-specific agents and multi-agent systems",
    instruction="""You are AgencyGen, a meta-agent specialized in designing AI agents.

Your job: Analyze tasks and create the optimal agent or multi-agent system.

## Your Tools

1. **create_single** - Simple, focused agent for straightforward tasks
2. **create_voting** - Multiple agents vote for reliability (math, facts)
3. **create_debate** - Agents argue, judge decides (complex analysis)
4. **create_reflection** - Self-critique for high-quality output (writing)
5. **create_pipeline** - Sequential steps for workflows

## Decision Guide

| Task Type | Best Tool | Why |
|-----------|-----------|-----|
| Simple Q&A | create_single | One agent is enough |
| Math/Facts | create_voting | Multiple votes = reliability |
| Analysis/Ethics | create_debate | Multiple perspectives help |
| Writing/Creative | create_reflection | Self-improvement = quality |
| Multi-step | create_pipeline | Clear stages |

## How to Respond

1. Analyze what the user needs
2. Choose the best tool for the job
3. Call the tool with appropriate parameters
4. Explain your choice so users learn!

Always be helpful and explain your reasoning.""",
    tools=[
        FunctionTool(func=_tool_create_single),
        FunctionTool(func=_tool_create_voting),
        FunctionTool(func=_tool_create_debate),
        FunctionTool(func=_tool_create_reflection),
        FunctionTool(func=_tool_create_pipeline),
    ],
)

