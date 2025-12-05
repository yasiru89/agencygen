"""
Recursive Language Model (RLM) pattern definitions for AgencyGen.

Implements pre-defined RLM patterns as described in research on recursive self-improvement:
- Chunking RLM: Process long contexts by chunking and compressing
- Iterative RLM: Self-refine until convergence
- Hierarchical RLM: Decompose complex problems recursively

Note: These are "rudimentary" RLM patterns that use pre-defined strategies.
For the true RLM with REPL environment, see the repl.py module.

Based on: https://alexzhang13.github.io/blog/2025/rlm/
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict

from google.adk.agents import LlmAgent, LoopAgent

from ..config import DEFAULT_MODEL
from .termination import (
    RLMState,
    TerminationStrategy,
    DepthTermination,
    ConvergenceTermination,
    QualityTermination,
    CompositeTermination,
)


@dataclass
class RLMConfig:
    """
    Configuration for Recursive Language Model execution.
    """
    # Termination settings
    max_depth: int = 5
    max_iterations: int = 10
    convergence_threshold: float = 0.95
    
    # Chunking settings
    chunk_size: int = 4000  # characters per chunk
    chunk_overlap: int = 200  # overlap between chunks
    
    # Quality settings
    use_quality_check: bool = False
    quality_approval_keyword: str = "APPROVED"
    
    # Model settings
    model: str = DEFAULT_MODEL
    compression_model: Optional[str] = None  # defaults to model
    
    # Custom termination
    termination_strategy: Optional[TerminationStrategy] = None

    def get_termination_strategy(self) -> TerminationStrategy:
        """Get the configured termination strategy."""
        if self.termination_strategy:
            return self.termination_strategy
        
        strategies: List[TerminationStrategy] = [
            DepthTermination(max_depth=self.max_depth),
        ]
        
        if self.convergence_threshold < 1.0:
            strategies.append(
                ConvergenceTermination(similarity_threshold=self.convergence_threshold)
            )
        
        if self.use_quality_check:
            strategies.append(
                QualityTermination(approval_keyword=self.quality_approval_keyword)
            )
        
        return CompositeTermination(strategies=strategies, mode="or")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start, accounting for overlap
        start = end - overlap
        
        # Prevent infinite loop on very small overlap
        if start >= len(text) - overlap:
            break
    
    return chunks


def create_compression_agent(
    name: str,
    model: str = DEFAULT_MODEL,
) -> LlmAgent:
    """
    Create an agent that compresses/summarizes text while preserving key information.
    """
    return LlmAgent(
        name=f"{name}_compressor",
        model=model,
        instruction="""You are a context compression agent. Your task is to:

1. Summarize the given text while preserving ALL key information
2. Maintain factual accuracy - never add information not in the original
3. Keep important details, names, numbers, and relationships
4. Reduce verbosity and redundancy
5. Output a compressed version that captures the essence

Be concise but complete. The compressed output will be used for further processing.""",
        description="Compresses context while preserving key information",
    )


def create_chunking_rlm(
    name: str,
    instruction: str,
    config: Optional[RLMConfig] = None,
) -> Dict[str, Any]:
    """
    Create a Chunking RLM for processing long contexts.
    
    The chunking RLM:
    1. Splits long input into manageable chunks
    2. Processes each chunk with the worker agent
    3. Compresses intermediate results
    4. Aggregates all chunk results into final output
    
    Args:
        name: Base name for the agents
        instruction: Task instruction for processing
        config: RLM configuration (uses defaults if None)
        
    Returns:
        Dict containing:
        - worker: Agent that processes each chunk
        - compressor: Agent that compresses intermediate results
        - aggregator: Agent that combines all results
        - config: The configuration used
        - chunk_fn: Function to chunk text
        - type: "chunking"
    """
    config = config or RLMConfig()
    model = config.model
    compression_model = config.compression_model or model
    
    worker = LlmAgent(
        name=f"{name}_chunk_worker",
        model=model,
        instruction=f"""You are processing a CHUNK of a larger document.

Your task for this chunk:
{instruction}

Important:
- Focus only on the content in this chunk
- Extract relevant information completely
- Note if information seems incomplete (may continue in next chunk)
- Be thorough but concise""",
        description="Processes individual chunks of input",
    )
    
    compressor = create_compression_agent(f"{name}_chunk", compression_model)
    
    aggregator = LlmAgent(
        name=f"{name}_aggregator",
        model=model,
        instruction=f"""You are aggregating results from multiple chunks of a document.

Original task:
{instruction}

Your job:
1. Combine the processed chunk results into a coherent whole
2. Resolve any overlaps or redundancies
3. Ensure the final output addresses the original task completely
4. Synthesize a unified response from the chunk outputs""",
        description="Aggregates results from all chunks",
    )
    
    def chunk_fn(text: str) -> List[str]:
        return _chunk_text(text, config.chunk_size, config.chunk_overlap)
    
    return {
        "worker": worker,
        "compressor": compressor,
        "aggregator": aggregator,
        "config": config,
        "chunk_fn": chunk_fn,
        "type": "chunking",
    }


def create_iterative_rlm(
    name: str,
    instruction: str,
    config: Optional[RLMConfig] = None,
) -> Dict[str, Any]:
    """
    Create an Iterative RLM for self-refinement until convergence.
    
    The iterative RLM:
    1. Generates an initial response
    2. Critiques and improves the response
    3. Repeats until convergence or max iterations
    
    Args:
        name: Base name for the agents
        instruction: Task instruction
        config: RLM configuration
        
    Returns:
        Dict containing:
        - worker: Agent that generates/improves output
        - critic: Agent that provides feedback
        - loop: LoopAgent combining worker and critic
        - config: The configuration used
        - termination: Termination strategy
        - type: "iterative"
    """
    config = config or RLMConfig()
    model = config.model
    
    worker = LlmAgent(
        name=f"{name}_iterative_worker",
        model=model,
        instruction=f"""You are an iterative improvement agent.

Your task:
{instruction}

On first call: Generate your best initial response.

On subsequent calls: You will receive feedback on your previous output.
Carefully incorporate the feedback to improve your response.
Make meaningful improvements while preserving what was good.

Always output your complete, improved response.""",
        description="Generates and iteratively improves output",
    )
    
    critic = LlmAgent(
        name=f"{name}_iterative_critic",
        model=model,
        instruction=f"""You are a constructive critic for iterative improvement.

The task being worked on:
{instruction}

Your job:
1. Evaluate the current output against the task requirements
2. Identify specific areas for improvement
3. Provide actionable feedback
4. If the output is excellent and needs no changes, respond with "APPROVED"

Be specific and constructive. Focus on substance over style.
Only say "APPROVED" if the output truly meets high standards.""",
        description="Critiques output and suggests improvements",
    )
    
    loop = LoopAgent(
        name=f"{name}_iterative_loop",
        sub_agents=[worker, critic],
        max_iterations=config.max_iterations,
        description=f"Iterative refinement loop (max {config.max_iterations} iterations)",
    )
    
    termination = config.get_termination_strategy()
    
    return {
        "worker": worker,
        "critic": critic,
        "loop": loop,
        "config": config,
        "termination": termination,
        "type": "iterative",
    }


def create_hierarchical_rlm(
    name: str,
    instruction: str,
    config: Optional[RLMConfig] = None,
) -> Dict[str, Any]:
    """
    Create a Hierarchical RLM for recursive problem decomposition.
    
    The hierarchical RLM:
    1. Decomposes complex problems into sub-problems
    2. Recursively solves each sub-problem
    3. Aggregates solutions back up the hierarchy
    
    Args:
        name: Base name for the agents
        instruction: Task instruction
        config: RLM configuration
        
    Returns:
        Dict containing:
        - decomposer: Agent that breaks down problems
        - solver: Agent that solves leaf problems
        - aggregator: Agent that combines solutions
        - config: The configuration used
        - termination: Termination strategy
        - type: "hierarchical"
    """
    config = config or RLMConfig()
    model = config.model
    
    decomposer = LlmAgent(
        name=f"{name}_decomposer",
        model=model,
        instruction=f"""You are a problem decomposition agent.

Main task:
{instruction}

Your job:
1. Analyze if the problem can be broken into smaller sub-problems
2. If decomposable, output a JSON list of sub-problems:
   {{"decompose": true, "sub_problems": ["sub-problem 1", "sub-problem 2", ...]}}
3. If the problem is simple enough to solve directly:
   {{"decompose": false, "reason": "why it's a leaf problem"}}

Guidelines:
- Each sub-problem should be self-contained
- Aim for 2-4 sub-problems when decomposing
- Don't over-decompose simple problems
- Sub-problems should combine to solve the original""",
        description="Decomposes complex problems into sub-problems",
    )
    
    solver = LlmAgent(
        name=f"{name}_solver",
        model=model,
        instruction=f"""You are a problem solver for atomic (leaf) problems.

Context - This is part of a larger task:
{instruction}

Your job:
1. Solve the specific sub-problem given to you
2. Provide a clear, complete solution
3. Your solution will be combined with others

Be thorough and precise. Focus on solving THIS specific problem.""",
        description="Solves leaf-level problems",
    )
    
    aggregator = LlmAgent(
        name=f"{name}_hierarchical_aggregator",
        model=model,
        instruction=f"""You are a solution aggregation agent.

Original task:
{instruction}

Your job:
1. Receive solutions to sub-problems
2. Combine them into a coherent overall solution
3. Resolve any conflicts or overlaps
4. Ensure the combined solution addresses the original task

Synthesize a unified, complete response.""",
        description="Aggregates solutions from sub-problems",
    )
    
    termination = DepthTermination(max_depth=config.max_depth)
    
    return {
        "decomposer": decomposer,
        "solver": solver,
        "aggregator": aggregator,
        "config": config,
        "termination": termination,
        "type": "hierarchical",
    }


def create_recursive_agent(
    name: str,
    instruction: str,
    rlm_type: str = "iterative",
    config: Optional[RLMConfig] = None,
) -> Dict[str, Any]:
    """
    Create a recursive agent with the specified RLM type.
    
    This is a convenience function that dispatches to the appropriate
    RLM factory based on the type.
    
    Args:
        name: Base name for the agents
        instruction: Task instruction
        rlm_type: Type of RLM - "chunking", "iterative", or "hierarchical"
        config: RLM configuration
        
    Returns:
        Dict containing the RLM components (varies by type)
    """
    factories = {
        "chunking": create_chunking_rlm,
        "iterative": create_iterative_rlm,
        "hierarchical": create_hierarchical_rlm,
    }
    
    if rlm_type not in factories:
        raise ValueError(
            f"Unknown RLM type: {rlm_type}. "
            f"Valid types: {list(factories.keys())}"
        )
    
    return factories[rlm_type](name, instruction, config)
