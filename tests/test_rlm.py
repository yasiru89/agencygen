"""
Tests for Recursive Language Model (RLM) primitives.

Tests cover:
- Termination strategies
- RLM configuration
- Chunking RLM creation and execution
- Iterative RLM creation and execution
- Hierarchical RLM creation and execution
- State persistence across recursive calls
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agency_gen.rlm.termination import (
    RLMState,
    DepthTermination,
    ConvergenceTermination,
    QualityTermination,
    ChunkTermination,
    CompositeTermination,
    create_default_termination,
)

from agency_gen.rlm.patterns import (
    RLMConfig,
    _chunk_text,
    create_compression_agent,
    create_chunking_rlm,
    create_iterative_rlm,
    create_hierarchical_rlm,
    create_recursive_agent,
)


class TestRLMState:
    """Tests for RLMState tracking."""

    def test_initial_state(self):
        state = RLMState()
        assert state.current_depth == 0
        assert state.max_depth == 5
        assert state.outputs == []
        assert state.chunks_processed == 0
        assert state.total_chunks == 0

    def test_add_output(self):
        state = RLMState()
        state.add_output("first output")
        state.add_output("second output")
        
        assert len(state.outputs) == 2
        assert state.last_output == "second output"
        assert state.previous_output == "first output"

    def test_last_output_empty(self):
        state = RLMState()
        assert state.last_output is None
        assert state.previous_output is None

    def test_previous_output_single(self):
        state = RLMState()
        state.add_output("only one")
        assert state.last_output == "only one"
        assert state.previous_output is None


class TestDepthTermination:
    """Tests for depth-based termination."""

    def test_terminates_at_max_depth(self):
        term = DepthTermination(max_depth=3)
        state = RLMState(current_depth=3)
        assert term.should_terminate(state) is True

    def test_continues_below_max_depth(self):
        term = DepthTermination(max_depth=5)
        state = RLMState(current_depth=2)
        assert term.should_terminate(state) is False

    def test_reason_at_termination(self):
        term = DepthTermination(max_depth=3)
        state = RLMState(current_depth=3)
        assert "Max depth 3 reached" in term.reason(state)

    def test_reason_continuing(self):
        term = DepthTermination(max_depth=5)
        state = RLMState(current_depth=2)
        assert "2/5" in term.reason(state)


class TestConvergenceTermination:
    """Tests for convergence-based termination."""

    def test_identical_outputs_converge(self):
        term = ConvergenceTermination(similarity_threshold=0.95)
        state = RLMState()
        state.add_output("This is the output")
        state.add_output("This is the output")  # Identical
        
        assert term.should_terminate(state) is True

    def test_different_outputs_continue(self):
        term = ConvergenceTermination(similarity_threshold=0.95)
        state = RLMState()
        state.add_output("First version of the output")
        state.add_output("Completely different second output")
        
        assert term.should_terminate(state) is False

    def test_similar_outputs_converge(self):
        term = ConvergenceTermination(similarity_threshold=0.90)
        state = RLMState()
        state.add_output("The answer is 42 and it is correct")
        state.add_output("The answer is 42 and it is correct!")  # Very similar
        
        assert term.should_terminate(state) is True

    def test_min_iterations_respected(self):
        term = ConvergenceTermination(similarity_threshold=0.95, min_iterations=3)
        state = RLMState()
        state.add_output("same")
        state.add_output("same")
        
        # Only 2 iterations, need 3
        assert term.should_terminate(state) is False

    def test_reason_converged(self):
        term = ConvergenceTermination(similarity_threshold=0.95)
        state = RLMState()
        state.add_output("test")
        state.add_output("test")
        
        reason = term.reason(state)
        assert "Converged" in reason


class TestQualityTermination:
    """Tests for quality-based termination."""

    def test_approval_keyword_terminates(self):
        term = QualityTermination(approval_keyword="APPROVED")
        state = RLMState()
        state.add_output("The output is APPROVED for release")
        
        assert term.should_terminate(state) is True

    def test_no_approval_continues(self):
        term = QualityTermination(approval_keyword="APPROVED")
        state = RLMState()
        state.add_output("Needs more work, not ready yet")
        
        assert term.should_terminate(state) is False

    def test_custom_quality_function(self):
        def custom_check(state: RLMState) -> bool:
            return len(state.outputs) >= 3
        
        term = QualityTermination(quality_fn=custom_check)
        
        state = RLMState()
        state.add_output("one")
        state.add_output("two")
        assert term.should_terminate(state) is False
        
        state.add_output("three")
        assert term.should_terminate(state) is True

    def test_case_insensitive_approval(self):
        term = QualityTermination(approval_keyword="APPROVED")
        state = RLMState()
        state.add_output("approved")  # lowercase
        
        assert term.should_terminate(state) is True


class TestChunkTermination:
    """Tests for chunk-based termination."""

    def test_terminates_when_all_processed(self):
        term = ChunkTermination()
        state = RLMState(chunks_processed=5, total_chunks=5)
        
        assert term.should_terminate(state) is True

    def test_continues_with_remaining_chunks(self):
        term = ChunkTermination()
        state = RLMState(chunks_processed=3, total_chunks=5)
        
        assert term.should_terminate(state) is False

    def test_no_chunks_defined(self):
        term = ChunkTermination()
        state = RLMState(total_chunks=0)
        
        assert term.should_terminate(state) is False


class TestCompositeTermination:
    """Tests for composite termination strategies."""

    def test_or_mode_any_triggers(self):
        depth_term = DepthTermination(max_depth=5)
        quality_term = QualityTermination()
        
        composite = CompositeTermination(
            strategies=[depth_term, quality_term],
            mode="or"
        )
        
        # Only depth is met
        state = RLMState(current_depth=5)
        state.add_output("not approved")
        
        assert composite.should_terminate(state) is True

    def test_and_mode_all_required(self):
        depth_term = DepthTermination(max_depth=5)
        quality_term = QualityTermination()
        
        composite = CompositeTermination(
            strategies=[depth_term, quality_term],
            mode="and"
        )
        
        # Only depth is met, not quality
        state = RLMState(current_depth=5)
        state.add_output("needs more work")  # does NOT contain "APPROVED"
        
        assert composite.should_terminate(state) is False
        
        # Both met
        state.outputs = ["APPROVED"]
        assert composite.should_terminate(state) is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            CompositeTermination(strategies=[], mode="invalid")


class TestCreateDefaultTermination:
    """Tests for default termination factory."""

    def test_creates_composite(self):
        term = create_default_termination(max_depth=10, convergence_threshold=0.9)
        assert isinstance(term, CompositeTermination)

    def test_terminates_on_depth(self):
        term = create_default_termination(max_depth=3)
        state = RLMState(current_depth=3)
        
        assert term.should_terminate(state) is True


class TestChunkText:
    """Tests for text chunking utility."""

    def test_small_text_single_chunk(self):
        text = "Short text"
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_large_text_multiple_chunks(self):
        text = "A" * 1000
        chunks = _chunk_text(text, chunk_size=300, overlap=50)
        
        assert len(chunks) > 1
        # Each chunk should be at most chunk_size
        for chunk in chunks:
            assert len(chunk) <= 300

    def test_overlap_preserved(self):
        text = "ABCDEFGHIJ" * 100  # 1000 chars
        chunks = _chunk_text(text, chunk_size=300, overlap=50)
        
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            end_of_first = chunks[0][-50:]
            start_of_second = chunks[1][:50]
            # They should share some content
            assert end_of_first == start_of_second


class TestRLMConfig:
    """Tests for RLM configuration."""

    def test_default_config(self):
        config = RLMConfig()
        
        assert config.max_depth == 5
        assert config.max_iterations == 10
        assert config.convergence_threshold == 0.95
        assert config.chunk_size == 4000

    def test_custom_config(self):
        config = RLMConfig(
            max_depth=10,
            max_iterations=20,
            chunk_size=8000,
        )
        
        assert config.max_depth == 10
        assert config.max_iterations == 20
        assert config.chunk_size == 8000

    def test_get_termination_strategy_default(self):
        config = RLMConfig()
        term = config.get_termination_strategy()
        
        assert isinstance(term, CompositeTermination)

    def test_get_termination_strategy_custom(self):
        custom_term = DepthTermination(max_depth=3)
        config = RLMConfig(termination_strategy=custom_term)
        
        term = config.get_termination_strategy()
        assert term is custom_term


class TestCreateCompressionAgent:
    """Tests for compression agent creation."""

    def test_creates_llm_agent(self):
        agent = create_compression_agent("test")
        
        assert agent.name == "test_compressor"
        assert "compress" in agent.instruction.lower()


class TestCreateChunkingRLM:
    """Tests for chunking RLM creation."""

    def test_creates_all_components(self):
        rlm = create_chunking_rlm(
            name="test",
            instruction="Process the document",
        )
        
        assert "worker" in rlm
        assert "compressor" in rlm
        assert "aggregator" in rlm
        assert "config" in rlm
        assert "chunk_fn" in rlm
        assert rlm["type"] == "chunking"

    def test_chunk_function_works(self):
        rlm = create_chunking_rlm(
            name="test",
            instruction="Process",
            config=RLMConfig(chunk_size=100, chunk_overlap=20),
        )
        
        chunks = rlm["chunk_fn"]("A" * 500)
        assert len(chunks) > 1


class TestCreateIterativeRLM:
    """Tests for iterative RLM creation."""

    def test_creates_all_components(self):
        rlm = create_iterative_rlm(
            name="test",
            instruction="Improve the text",
        )
        
        assert "worker" in rlm
        assert "critic" in rlm
        assert "loop" in rlm
        assert "config" in rlm
        assert "termination" in rlm
        assert rlm["type"] == "iterative"

    def test_worker_instruction_contains_task(self):
        rlm = create_iterative_rlm(
            name="test",
            instruction="Write a poem",
        )
        
        assert "poem" in rlm["worker"].instruction.lower()


class TestCreateHierarchicalRLM:
    """Tests for hierarchical RLM creation."""

    def test_creates_all_components(self):
        rlm = create_hierarchical_rlm(
            name="test",
            instruction="Solve the problem",
        )
        
        assert "decomposer" in rlm
        assert "solver" in rlm
        assert "aggregator" in rlm
        assert "config" in rlm
        assert "termination" in rlm
        assert rlm["type"] == "hierarchical"

    def test_decomposer_can_detect_json(self):
        rlm = create_hierarchical_rlm(
            name="test",
            instruction="Complex problem",
        )
        
        # Decomposer should mention JSON in its instruction
        assert "json" in rlm["decomposer"].instruction.lower()


class TestCreateRecursiveAgent:
    """Tests for the convenience factory function."""

    def test_creates_chunking(self):
        rlm = create_recursive_agent(
            name="test",
            instruction="Process",
            rlm_type="chunking",
        )
        assert rlm["type"] == "chunking"

    def test_creates_iterative(self):
        rlm = create_recursive_agent(
            name="test",
            instruction="Refine",
            rlm_type="iterative",
        )
        assert rlm["type"] == "iterative"

    def test_creates_hierarchical(self):
        rlm = create_recursive_agent(
            name="test",
            instruction="Decompose",
            rlm_type="hierarchical",
        )
        assert rlm["type"] == "hierarchical"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown RLM type"):
            create_recursive_agent(
                name="test",
                instruction="Task",
                rlm_type="invalid_type",
            )


class TestRLMStateIntegration:
    """Integration tests for state management across RLM operations."""

    def test_state_tracks_iterations(self):
        state = RLMState(max_depth=10)
        
        for i in range(5):
            state.current_depth = i + 1
            state.add_output(f"Output {i}")
        
        assert state.current_depth == 5
        assert len(state.outputs) == 5
        assert state.last_output == "Output 4"

    def test_state_with_termination(self):
        term = create_default_termination(max_depth=3)
        state = RLMState()
        
        # Simulate iteration loop
        for i in range(10):
            state.current_depth = i + 1
            state.add_output(f"Output {i}")
            
            if term.should_terminate(state):
                break
        
        # Should have stopped at depth 3
        assert state.current_depth == 3


# Async tests for RLM runners would go here but require mocking ADK
# These are placeholder markers for integration tests

class TestRLMRunnerIntegration:
    """Integration tests for RLM runners (require mocking)."""

    @pytest.mark.asyncio
    async def test_chunking_runner_placeholder(self):
        """Placeholder for chunking runner integration test."""
        # This would test run_chunking_rlm with mocked ADK
        pass

    @pytest.mark.asyncio
    async def test_iterative_runner_placeholder(self):
        """Placeholder for iterative runner integration test."""
        # This would test run_iterative_rlm with mocked ADK
        pass

    @pytest.mark.asyncio  
    async def test_hierarchical_runner_placeholder(self):
        """Placeholder for hierarchical runner integration test."""
        # This would test run_hierarchical_rlm with mocked ADK
        pass

