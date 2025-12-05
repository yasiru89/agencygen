"""
Tests for the true RLM (Recursive Language Model) with REPL implementation.

These tests verify:
1. Code block extraction from model output
2. FINAL answer extraction
3. REPL environment execution
4. Basic RLM flow (mocked)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

# Import the modules to test
from agency_gen.rlm.repl import (
    _extract_code_blocks,
    _extract_final_answer,
    REPLEnvironment,
    RLMREPLConfig,
    RLMREPL,
)


class TestCodeBlockExtraction:
    """Test extraction of Python code blocks from model output."""
    
    def test_extract_single_python_block(self):
        """Test extracting a single Python code block."""
        text = '''Here's some code:
```python
x = 1 + 2
print(x)
```
That's the code.'''
        
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert "x = 1 + 2" in blocks[0]
        assert "print(x)" in blocks[0]
    
    def test_extract_multiple_blocks(self):
        """Test extracting multiple code blocks."""
        text = '''First block:
```python
a = 1
```
Second block:
```python
b = 2
```
Done.'''
        
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 2
        assert "a = 1" in blocks[0]
        assert "b = 2" in blocks[1]
    
    def test_extract_block_without_python_tag(self):
        """Test extracting code block without explicit python tag."""
        text = '''Code:
```
result = 42
```'''
        
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert "result = 42" in blocks[0]
    
    def test_no_code_blocks(self):
        """Test when there are no code blocks."""
        text = "Just plain text without any code."
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 0
    
    def test_empty_code_block(self):
        """Test handling of empty code blocks."""
        text = '''Empty:
```python
```
Not empty:
```python
x = 1
```'''
        
        blocks = _extract_code_blocks(text)
        # Empty blocks should be filtered out
        assert len(blocks) == 1
        assert "x = 1" in blocks[0]


class TestFinalAnswerExtraction:
    """Test extraction of FINAL(...) and FINAL_VAR(...) answers."""
    
    def test_extract_final_value(self):
        """Test extracting FINAL(value) answer."""
        text = "After analysis, the answer is FINAL(42)"
        result = _extract_final_answer(text)
        assert result == ("value", "42")
    
    def test_extract_final_string(self):
        """Test extracting FINAL with string content."""
        text = 'The result is FINAL(The answer is blue)'
        result = _extract_final_answer(text)
        assert result == ("value", "The answer is blue")
    
    def test_extract_final_var(self):
        """Test extracting FINAL_VAR(variable_name)."""
        text = "I've stored the answer in the variable, so FINAL_VAR(my_result)"
        result = _extract_final_answer(text)
        assert result == ("var", "my_result")
    
    def test_no_final(self):
        """Test when there's no final answer."""
        text = "Still working on the problem..."
        result = _extract_final_answer(text)
        assert result is None
    
    def test_final_with_multiline(self):
        """Test FINAL with multiline content."""
        text = '''Here's the answer:
FINAL(Line 1
Line 2)'''
        result = _extract_final_answer(text)
        assert result is not None
        assert result[0] == "value"
        assert "Line 1" in result[1]


class TestREPLEnvironment:
    """Test the REPL execution environment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RLMREPLConfig()
        self.mock_llm = Mock(return_value="Mock LLM response")
        self.context = "This is the test context with some data."
        self.repl = REPLEnvironment(
            context=self.context,
            llm_fn=self.mock_llm,
            config=self.config,
        )
    
    def test_context_available(self):
        """Test that context is available as a variable."""
        result = self.repl.execute(f"print(len({self.config.context_var_name}))")
        assert result["error"] is None
        assert str(len(self.context)) in result["output"]
    
    def test_simple_expression(self):
        """Test simple expression evaluation."""
        result = self.repl.execute("1 + 2")
        assert result["error"] is None
        assert "3" in result["output"]
    
    def test_variable_assignment(self):
        """Test variable assignment and retrieval."""
        self.repl.execute("x = 42")
        result = self.repl.execute("print(x)")
        assert result["error"] is None
        assert "42" in result["output"]
    
    def test_context_manipulation(self):
        """Test manipulating the context."""
        result = self.repl.execute(f"print({self.config.context_var_name}[:10])")
        assert result["error"] is None
        assert "This is th" in result["output"]
    
    def test_error_handling(self):
        """Test that errors are captured properly."""
        result = self.repl.execute("undefined_variable")
        assert result["error"] is not None
        assert "NameError" in result["error"]
    
    def test_llm_function_available(self):
        """Test that llm() function is available."""
        self.repl.execute('response = llm("test query", "test context")')
        assert self.mock_llm.called
        self.mock_llm.assert_called_with("test query", "test context")
    
    def test_llm_depth_limit(self):
        """Test that llm() respects depth limits."""
        # Set a low depth limit
        self.config.max_recursive_depth = 1
        self.repl = REPLEnvironment(
            context=self.context,
            llm_fn=self.mock_llm,
            config=self.config,
        )
        
        # Simulate being at max depth
        self.repl.current_depth = 1
        result = self.repl.execute('result = llm("query")')
        
        # Should get error message, not actual LLM call
        stored_result = self.repl.get_variable("result")
        assert "Max recursive depth" in stored_result
    
    def test_get_variable(self):
        """Test getting variables from namespace."""
        self.repl.execute("my_var = 'hello world'")
        value = self.repl.get_variable("my_var")
        assert value == "hello world"
    
    def test_get_nonexistent_variable(self):
        """Test getting a variable that doesn't exist."""
        value = self.repl.get_variable("nonexistent")
        assert value is None
    
    def test_imports_available(self):
        """Test that pre-imported modules are available."""
        result = self.repl.execute("print(re.findall(r'\\w+', 'hello world'))")
        assert result["error"] is None
        assert "hello" in result["output"]
    
    def test_output_truncation(self):
        """Test that long outputs are truncated."""
        self.config.max_output_chars = 50
        self.repl = REPLEnvironment(
            context=self.context,
            llm_fn=self.mock_llm,
            config=self.config,
        )
        
        result = self.repl.execute("print('x' * 1000)")
        assert len(result["output"]) <= 100  # 50 chars + truncation message
        assert "truncated" in result["output"]


class TestRLMREPLConfig:
    """Test RLM REPL configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RLMREPLConfig()
        assert config.max_iterations == 20
        assert config.max_recursive_depth == 3
        assert config.context_var_name == "context"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RLMREPLConfig(
            max_iterations=5,
            max_recursive_depth=1,
            context_var_name="data",
        )
        assert config.max_iterations == 5
        assert config.max_recursive_depth == 1
        assert config.context_var_name == "data"


class TestRLMREPLIntegration:
    """Integration tests for the full RLM REPL (requires mocking LLM calls)."""
    
    @pytest.mark.asyncio
    async def test_rlm_basic_flow(self):
        """Test basic RLM flow with mocked LLM."""
        rlm = RLMREPL(name="test_rlm")
        
        # Mock the LLM call to return a simple answer
        with patch.object(rlm, '_call_llm', new_callable=AsyncMock) as mock_llm:
            # First call: model writes code
            mock_llm.return_value = '''Let me analyze this:
```python
count = context.count("test")
print(f"Found {count} occurrences")
```'''
            
            # We need to also mock the runner
            with patch('agency_gen.rlm.repl.Runner') as mock_runner_class:
                mock_runner = Mock()
                mock_runner_class.return_value = mock_runner
                
                # Create an async generator for run_async
                async def mock_run_async(*args, **kwargs):
                    mock_event = Mock()
                    mock_event.content = Mock()
                    mock_event.content.parts = [Mock(text="FINAL(2)")]
                    yield mock_event
                
                mock_runner.run_async = mock_run_async
                
                result = await rlm.run(
                    query="How many times does 'test' appear?",
                    context="This is a test. Another test here.",
                )
                
                # Should complete with a result
                assert "result" in result
                assert "iterations" in result


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
