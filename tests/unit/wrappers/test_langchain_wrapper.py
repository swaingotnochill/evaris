"""Tests for LangChainAgentWrapper."""

from typing import Any, Optional

import pytest

from evaris.wrappers.langchain import LangChainAgentWrapper


class MockLangChainAgentWithRun:
    """Mock LangChain agent with run method (legacy API)."""

    def __init__(self, response: str = "mocked response") -> None:
        self.response = response
        self.last_input: Optional[Any] = None

    def run(self, input: Any) -> str:
        """Sync run method."""
        self.last_input = input
        return self.response


class MockLangChainAgentWithArun:
    """Mock LangChain agent with arun method (legacy async API)."""

    def __init__(self, response: str = "mocked async response") -> None:
        self.response = response
        self.last_input: Optional[Any] = None

    async def arun(self, input: Any) -> str:
        """Async run method."""
        self.last_input = input
        return self.response


class MockLangChainAgentWithInvoke:
    """Mock LangChain agent with invoke method (LCEL API)."""

    def __init__(self, response_value: str = "mocked invoke response") -> None:
        self.response_value = response_value
        self.last_input: Optional[Any] = None

    def invoke(self, input: Any) -> dict[str, str]:
        """Sync invoke method."""
        self.last_input = input
        return {"output": self.response_value}


class MockLangChainAgentWithAinvoke:
    """Mock LangChain agent with ainvoke method (LCEL async API)."""

    def __init__(self, response_value: str = "mocked async invoke response") -> None:
        self.response_value = response_value
        self.last_input: Optional[Any] = None

    async def ainvoke(self, input: Any) -> dict[str, str]:
        """Async invoke method."""
        self.last_input = input
        return {"output": self.response_value}


class MockLangChainAgentBothAPIs:
    """Mock LangChain agent with both run and invoke."""

    def __init__(self) -> None:
        self.last_input: Optional[Any] = None

    def run(self, input: Any) -> str:
        self.last_input = input
        return f"run: {input}"

    async def arun(self, input: Any) -> str:
        self.last_input = input
        return f"arun: {input}"

    def invoke(self, input: Any) -> dict[str, str]:
        self.last_input = input
        return {"output": f"invoke: {input}"}

    async def ainvoke(self, input: Any) -> dict[str, str]:
        self.last_input = input
        return {"output": f"ainvoke: {input}"}


class MockMemory:
    """Mock LangChain memory."""

    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        """Clear memory."""
        self.cleared = True


class TestLangChainAgentWrapper:
    """Tests for LangChainAgentWrapper."""

    def test_wraps_agent_with_run_method(self) -> None:
        """Test wrapping agent with run() method (legacy API)."""
        mock_agent = MockLangChainAgentWithRun("test response")
        wrapper = LangChainAgentWrapper(mock_agent)

        result = wrapper.execute("test input")

        assert result == "test response"
        assert mock_agent.last_input == "test input"

    @pytest.mark.asyncio
    async def test_wraps_agent_with_arun_method(self) -> None:
        """Test wrapping agent with arun() method (legacy async API)."""
        mock_agent = MockLangChainAgentWithArun("async test response")
        wrapper = LangChainAgentWrapper(mock_agent)

        result = await wrapper.a_execute("test input")

        assert result == "async test response"
        assert mock_agent.last_input == "test input"

    def test_wraps_agent_with_invoke_method(self) -> None:
        """Test wrapping agent with invoke() method (LCEL API)."""
        mock_agent = MockLangChainAgentWithInvoke("invoke response")
        wrapper = LangChainAgentWrapper(mock_agent)

        result = wrapper.execute("test input")

        assert result == "invoke response"
        assert mock_agent.last_input == {"input": "test input"}

    @pytest.mark.asyncio
    async def test_wraps_agent_with_ainvoke_method(self) -> None:
        """Test wrapping agent with ainvoke() method (LCEL async API)."""
        mock_agent = MockLangChainAgentWithAinvoke("async invoke response")
        wrapper = LangChainAgentWrapper(mock_agent)

        result = await wrapper.a_execute("test input")

        assert result == "async invoke response"
        assert mock_agent.last_input == {"input": "test input"}

    def test_prefers_run_over_invoke(self) -> None:
        """Test that run() is preferred over invoke() when both exist."""
        mock_agent = MockLangChainAgentBothAPIs()
        wrapper = LangChainAgentWrapper(mock_agent)

        result = wrapper.execute("test")

        assert result == "run: test"

    @pytest.mark.asyncio
    async def test_prefers_arun_over_ainvoke(self) -> None:
        """Test that arun() is preferred over ainvoke() when both exist."""
        mock_agent = MockLangChainAgentBothAPIs()
        wrapper = LangChainAgentWrapper(mock_agent)

        result = await wrapper.a_execute("test")

        assert result == "arun: test"

    def test_invoke_with_dict_input(self) -> None:
        """Test invoke with dict input passes through."""
        mock_agent = MockLangChainAgentWithInvoke()
        wrapper = LangChainAgentWrapper(mock_agent)

        input_dict = {"input": "test", "extra": "data"}
        wrapper.execute(input_dict)

        assert mock_agent.last_input == input_dict

    def test_invoke_extracts_output_key(self) -> None:
        """Test that invoke extracts 'output' key from result dict."""
        mock_agent = MockLangChainAgentWithInvoke("extracted")
        wrapper = LangChainAgentWrapper(mock_agent)

        result = wrapper.execute("test")
        assert result == "extracted"

    def test_invoke_extracts_answer_key(self) -> None:
        """Test that invoke can extract 'answer' key."""

        class MockAgentWithAnswer:
            def invoke(self, input: Any) -> dict[str, str]:
                return {"answer": "the answer"}

        wrapper = LangChainAgentWrapper(MockAgentWithAnswer())
        result = wrapper.execute("test")
        assert result == "the answer"

    def test_invoke_extracts_result_key(self) -> None:
        """Test that invoke can extract 'result' key."""

        class MockAgentWithResult:
            def invoke(self, input: Any) -> dict[str, str]:
                return {"result": "the result"}

        wrapper = LangChainAgentWrapper(MockAgentWithResult())
        result = wrapper.execute("test")
        assert result == "the result"

    def test_invoke_returns_full_dict_if_no_known_key(self) -> None:
        """Test that invoke returns full dict if no known output key found."""

        class MockAgentWithUnknownKeys:
            def invoke(self, input: Any) -> dict[str, str]:
                return {"unknown": "value", "other": "data"}

        wrapper = LangChainAgentWrapper(MockAgentWithUnknownKeys())
        result = wrapper.execute("test")
        assert result == {"unknown": "value", "other": "data"}

    def test_custom_output_key(self) -> None:
        """Test using custom output_key parameter."""

        class MockAgentCustomKey:
            def invoke(self, input: Any) -> dict[str, str]:
                return {"custom_output": "custom value"}

        wrapper = LangChainAgentWrapper(MockAgentCustomKey(), output_key="custom_output")
        result = wrapper.execute("test")
        assert result == "custom value"

    def test_custom_input_key(self) -> None:
        """Test using custom input_key parameter."""
        mock_agent = MockLangChainAgentWithInvoke()
        wrapper = LangChainAgentWrapper(mock_agent, input_key="query")

        wrapper.execute("test")

        assert mock_agent.last_input == {"query": "test"}

    def test_raises_value_error_for_unsupported_agent(self) -> None:
        """Test that agent without any supported methods raises ValueError."""

        class UnsupportedAgent:
            def some_method(self) -> None:
                pass

        with pytest.raises(ValueError, match="must have at least one of"):
            LangChainAgentWrapper(UnsupportedAgent())

    @pytest.mark.asyncio
    async def test_sync_only_agent_falls_back_to_thread_pool(self) -> None:
        """Test that sync-only agent falls back to thread pool execution."""
        mock_agent = MockLangChainAgentWithRun("sync fallback")
        wrapper = LangChainAgentWrapper(mock_agent)

        # Ensure only sync methods are available
        wrapper._has_arun = False
        wrapper._has_ainvoke = False
        wrapper._has_apredict = False

        # Should fall back to running sync method in thread pool
        result = await wrapper.a_execute("test")
        assert result == "sync fallback"
        assert mock_agent.last_input == "test"

    @pytest.mark.asyncio
    async def test_raises_not_implemented_when_no_methods_available(self) -> None:
        """Test that NotImplementedError is raised when no execution methods exist."""
        mock_agent = MockLangChainAgentWithRun()
        wrapper = LangChainAgentWrapper(mock_agent)

        # Remove ALL methods
        wrapper._has_arun = False
        wrapper._has_ainvoke = False
        wrapper._has_apredict = False
        wrapper._has_run = False
        wrapper._has_invoke = False
        wrapper._has_predict = False

        with pytest.raises(NotImplementedError, match="doesn't support async"):
            await wrapper.a_execute("test")

    @pytest.mark.asyncio
    async def test_async_fallback_to_sync(self) -> None:
        """Test that async execution can fall back to sync in thread pool."""
        mock_agent = MockLangChainAgentWithRun("sync fallback")
        wrapper = LangChainAgentWrapper(mock_agent)

        # Simulate agent with only sync methods
        wrapper._has_arun = False
        wrapper._has_ainvoke = False
        wrapper._has_apredict = False

        result = await wrapper.a_execute("test")
        assert result == "sync fallback"

    def test_reset_clears_memory(self) -> None:
        """Test that reset clears agent memory."""

        class MockAgentWithMemory:
            def __init__(self) -> None:
                self.memory = MockMemory()

            def run(self, input: Any) -> str:
                return "response"

        mock_agent = MockAgentWithMemory()
        wrapper = LangChainAgentWrapper(mock_agent)

        wrapper.reset()

        assert mock_agent.memory.cleared is True

    def test_reset_handles_chat_memory(self) -> None:
        """Test that reset handles nested chat_memory attribute."""

        class MockChatMemory:
            def __init__(self) -> None:
                self.cleared = False

            def clear(self) -> None:
                self.cleared = True

        class MockMemoryWithChatMemory:
            def __init__(self) -> None:
                self.chat_memory = MockChatMemory()

        class MockAgentWithChatMemory:
            def __init__(self) -> None:
                self.memory = MockMemoryWithChatMemory()

            def run(self, input: Any) -> str:
                return "response"

        mock_agent = MockAgentWithChatMemory()
        wrapper = LangChainAgentWrapper(mock_agent)

        wrapper.reset()

        assert mock_agent.memory.chat_memory.cleared is True

    def test_reset_calls_agent_reset_if_present(self) -> None:
        """Test that reset calls agent's own reset method if present."""

        class MockAgentWithReset:
            def __init__(self) -> None:
                self.was_reset = False

            def run(self, input: Any) -> str:
                return "response"

            def reset(self) -> None:
                self.was_reset = True

        mock_agent = MockAgentWithReset()
        wrapper = LangChainAgentWrapper(mock_agent)

        wrapper.reset()

        assert mock_agent.was_reset is True

    def test_reset_handles_agent_without_memory(self) -> None:
        """Test that reset works with agent that has no memory."""
        mock_agent = MockLangChainAgentWithRun()
        wrapper = LangChainAgentWrapper(mock_agent)

        wrapper.reset()  # Should not raise

    def test_handles_exception_in_execute(self) -> None:
        """Test exception handling in execute."""

        class FailingAgent:
            def run(self, input: Any) -> str:
                raise ValueError("Agent failed")

        wrapper = LangChainAgentWrapper(FailingAgent())

        with pytest.raises(ValueError, match="Agent failed"):
            wrapper.execute("test")

    @pytest.mark.asyncio
    async def test_handles_exception_in_async_execute(self) -> None:
        """Test exception handling in async execute."""

        class FailingAsyncAgent:
            async def arun(self, input: Any) -> str:
                raise ValueError("Async agent failed")

        wrapper = LangChainAgentWrapper(FailingAsyncAgent())

        with pytest.raises(ValueError, match="Async agent failed"):
            await wrapper.a_execute("test")
