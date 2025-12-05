"""
Termination strategies for Recursive Language Models (RLM).

Provides flexible termination conditions for recursive agent execution:
- DepthTermination: Stop at max recursion depth
- ConvergenceTermination: Stop when output stabilizes
- QualityTermination: Stop when critic approves
- CompositeTermination: Combine multiple strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Callable, Any


@dataclass
class RLMState:
    """
    State tracking for recursive language model execution.
    """
    current_depth: int = 0
    max_depth: int = 5
    outputs: List[str] = field(default_factory=list)
    chunks_processed: int = 0
    total_chunks: int = 0
    accumulated_context: str = ""
    metadata: dict = field(default_factory=dict)

    def add_output(self, output: str) -> None:
        """Add an output to the history."""
        self.outputs.append(output)

    @property
    def last_output(self) -> Optional[str]:
        """Get the most recent output."""
        return self.outputs[-1] if self.outputs else None

    @property
    def previous_output(self) -> Optional[str]:
        """Get the second most recent output."""
        return self.outputs[-2] if len(self.outputs) >= 2 else None


class TerminationStrategy(ABC):
    """
    Abstract base class for RLM termination strategies.
    """

    @abstractmethod
    def should_terminate(self, state: RLMState) -> bool:
        """
        Determine if recursion should terminate.
        
        Args:
            state: Current RLM state
            
        Returns:
            True if recursion should stop, False otherwise
        """
        pass

    @abstractmethod
    def reason(self, state: RLMState) -> str:
        """
        Provide a reason for the termination decision.
        
        Args:
            state: Current RLM state
            
        Returns:
            Human-readable reason string
        """
        pass


class DepthTermination(TerminationStrategy):
    """
    Terminate when maximum recursion depth is reached.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def should_terminate(self, state: RLMState) -> bool:
        return state.current_depth >= self.max_depth

    def reason(self, state: RLMState) -> str:
        if self.should_terminate(state):
            return f"Max depth {self.max_depth} reached"
        return f"Depth {state.current_depth}/{self.max_depth}"


class ConvergenceTermination(TerminationStrategy):
    """
    Terminate when outputs converge (become similar enough).
    
    Uses sequence matching to detect when consecutive outputs
    are similar enough to indicate convergence.
    """

    def __init__(self, similarity_threshold: float = 0.95, min_iterations: int = 2):
        """
        Args:
            similarity_threshold: Ratio (0-1) of similarity required for convergence
            min_iterations: Minimum iterations before checking convergence
        """
        self.similarity_threshold = similarity_threshold
        self.min_iterations = min_iterations

    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute similarity ratio between two strings."""
        return SequenceMatcher(None, a, b).ratio()

    def should_terminate(self, state: RLMState) -> bool:
        if len(state.outputs) < self.min_iterations:
            return False

        last = state.last_output
        previous = state.previous_output

        if last is None or previous is None:
            return False

        similarity = self._compute_similarity(last, previous)
        return similarity >= self.similarity_threshold

    def reason(self, state: RLMState) -> str:
        if len(state.outputs) < 2:
            return f"Need {self.min_iterations} iterations before checking convergence"

        last = state.last_output
        previous = state.previous_output

        if last and previous:
            similarity = self._compute_similarity(last, previous)
            if similarity >= self.similarity_threshold:
                return f"Converged at {similarity:.1%} similarity"
            return f"Similarity {similarity:.1%} < {self.similarity_threshold:.1%} threshold"

        return "Insufficient outputs for convergence check"


class QualityTermination(TerminationStrategy):
    """
    Terminate when a quality check function approves the output.
    
    The quality function receives the current state and returns
    True if the output meets quality standards.
    """

    def __init__(
        self,
        quality_fn: Optional[Callable[[RLMState], bool]] = None,
        approval_keyword: str = "APPROVED",
    ):
        """
        Args:
            quality_fn: Custom function to check quality
            approval_keyword: Keyword to look for in output (fallback)
        """
        self.quality_fn = quality_fn
        self.approval_keyword = approval_keyword

    def should_terminate(self, state: RLMState) -> bool:
        if self.quality_fn:
            return self.quality_fn(state)

        # Fallback: check for approval keyword in last output
        last = state.last_output
        if last:
            return self.approval_keyword.upper() in last.upper()
        return False

    def reason(self, state: RLMState) -> str:
        if self.should_terminate(state):
            return "Quality threshold met"
        return "Quality threshold not yet met"


class ChunkTermination(TerminationStrategy):
    """
    Terminate when all chunks have been processed.
    """

    def should_terminate(self, state: RLMState) -> bool:
        if state.total_chunks == 0:
            return False
        return state.chunks_processed >= state.total_chunks

    def reason(self, state: RLMState) -> str:
        if state.total_chunks == 0:
            return "No chunks defined"
        return f"Processed {state.chunks_processed}/{state.total_chunks} chunks"


class CompositeTermination(TerminationStrategy):
    """
    Combine multiple termination strategies.
    
    Supports both AND (all must agree) and OR (any can trigger) modes.
    """

    def __init__(
        self,
        strategies: List[TerminationStrategy],
        mode: str = "or",
    ):
        """
        Args:
            strategies: List of termination strategies to combine
            mode: "or" (any triggers termination) or "and" (all must agree)
        """
        if mode not in ("or", "and"):
            raise ValueError("mode must be 'or' or 'and'")

        self.strategies = strategies
        self.mode = mode

    def should_terminate(self, state: RLMState) -> bool:
        results = [s.should_terminate(state) for s in self.strategies]

        if self.mode == "or":
            return any(results)
        else:  # and
            return all(results)

    def reason(self, state: RLMState) -> str:
        reasons = [s.reason(state) for s in self.strategies]
        triggered = [
            r for s, r in zip(self.strategies, reasons)
            if s.should_terminate(state)
        ]

        if self.should_terminate(state):
            return f"Terminated ({self.mode.upper()}): {'; '.join(triggered)}"
        return f"Continuing: {'; '.join(reasons)}"


def create_default_termination(
    max_depth: int = 5,
    convergence_threshold: float = 0.95,
) -> CompositeTermination:
    """
    Create a sensible default termination strategy.
    
    Terminates on either max depth OR convergence.
    """
    return CompositeTermination(
        strategies=[
            DepthTermination(max_depth=max_depth),
            ConvergenceTermination(similarity_threshold=convergence_threshold),
        ],
        mode="or",
    )
