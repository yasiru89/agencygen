"""
Recursive Language Model (RLM) module for AgencyGen.

This module provides implementations of Recursive Language Models as described in:
https://alexzhang13.github.io/blog/2025/rlm/

Two approaches are available:

1. **Pattern-based RLM** (rudimentary): Pre-defined patterns for recursive processing
   - Chunking RLM: Process long contexts by chunking and compressing
   - Iterative RLM: Self-refine until convergence
   - Hierarchical RLM: Decompose complex problems recursively

2. **REPL-based RLM** (true RLM): Model has access to a Python REPL where it can:
   - Access context as a variable
   - Write code to manipulate and analyze context
   - Call llm() recursively on context slices
   - Decide at runtime how to partition and recurse

Usage:
    # Pattern-based (rudimentary)
    from agency_gen.rlm import create_chunking_rlm, run_chunking_rlm
    
    rlm = create_chunking_rlm("analyzer", "Extract key points")
    result = await run_chunking_rlm(rlm, long_text)
    
    # REPL-based (true RLM)
    from agency_gen.rlm import run_rlm_repl
    
    result = await run_rlm_repl(
        query="How many errors are mentioned?",
        context=long_log_file
    )
"""

# Pattern-based RLM (rudimentary approach)
from .patterns import (
    RLMConfig,
    create_chunking_rlm,
    create_iterative_rlm,
    create_hierarchical_rlm,
    create_recursive_agent,
    create_compression_agent,
)

from .runner import (
    run_chunking_rlm,
    run_iterative_rlm,
    run_hierarchical_rlm,
    run_recursive_agent,
)

from .termination import (
    RLMState,
    TerminationStrategy,
    DepthTermination,
    ConvergenceTermination,
    QualityTermination,
    ChunkTermination,
    CompositeTermination,
    create_default_termination,
)

# True RLM with REPL environment
from .repl import (
    RLMREPLConfig,
    RLMREPL,
    RLMWithMCP,
    REPLEnvironment,
    run_rlm_repl,
    run_rlm_with_mcp,
)

__all__ = [
    # Pattern-based RLM
    "RLMConfig",
    "create_chunking_rlm",
    "create_iterative_rlm",
    "create_hierarchical_rlm",
    "create_recursive_agent",
    "create_compression_agent",
    "run_chunking_rlm",
    "run_iterative_rlm",
    "run_hierarchical_rlm",
    "run_recursive_agent",
    # Termination strategies
    "RLMState",
    "TerminationStrategy",
    "DepthTermination",
    "ConvergenceTermination",
    "QualityTermination",
    "ChunkTermination",
    "CompositeTermination",
    "create_default_termination",
    # True RLM with REPL
    "RLMREPLConfig",
    "RLMREPL",
    "RLMWithMCP",
    "REPLEnvironment",
    "run_rlm_repl",
    "run_rlm_with_mcp",
]
