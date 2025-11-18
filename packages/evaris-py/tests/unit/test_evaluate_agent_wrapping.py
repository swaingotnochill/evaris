"""Tests for agent wrapping in evaluate.py."""

from typing import Any

import pytest

from evaris.agent_interface import SimpleAgent
from evaris.evaluate import _wrap_agent
from evaris.wrappers.langchain import LangChainAgentWrapper


class MockSyncAgent:
    """Mock sync agent implementing AgentInterface."""

    def execute(self, input: Any) -> str:
        """Execute synchronously."""
        return f"sync: {input}"


class MockAsyncAgent:
    """Mock async agent implementing AgentInterface."""

    async def a_execute(self, input: Any) -> str:
        """Execute asynchronously."""
        return f"async: {input}"


class MockBothAgent:
    """Mock agent with both sync and async methods."""

    def execute(self, input: Any) -> str:
        """Execute synchronously."""
        return f"sync: {input}"

    async def a_execute(self, input: Any) -> str:
        """Execute asynchronously."""
        return f"async: {input}"


class NotAnAgent:
    """Mock class that doesn't implement AgentInterface."""

    def run(self, input: Any) -> Any:
        """Not the right method name."""
        return input


class TestWrapAgent:
    """Tests for _wrap_agent function."""

    def test_wraps_callable_function(self) -> None:
        """Test that callable functions are returned as-is."""

        def my_function(x: Any) -> str:
            return f"result: {x}"

        wrapped = _wrap_agent(my_function)
        assert wrapped is my_function
        assert wrapped("test") == "result: test"

    def test_wraps_lambda(self) -> None:
        """Test that lambda functions are returned as-is."""

        def my_lambda(x: Any) -> str:
            return f"lambda: {x}"

        wrapped = _wrap_agent(my_lambda)
        assert wrapped is my_lambda
        assert wrapped("test") == "lambda: test"

    def test_wraps_sync_agent(self) -> None:
        """Test wrapping sync-only agent."""
        agent = MockSyncAgent()
        wrapped = _wrap_agent(agent)

        assert callable(wrapped)
        assert wrapped("test") == "sync: test"

    @pytest.mark.asyncio
    async def test_wraps_async_agent(self) -> None:
        """Test wrapping async-only agent."""
        agent = MockAsyncAgent()
        wrapped = _wrap_agent(agent)

        assert callable(wrapped)
        result = await wrapped("test")
        assert result == "async: test"

    @pytest.mark.asyncio
    async def test_wraps_both_agent_prefers_async(self) -> None:
        """Test wrapping agent with both methods prefers async."""
        agent = MockBothAgent()
        wrapped = _wrap_agent(agent)

        # Should prefer async when both are available
        result = await wrapped("test")
        assert result == "async: test"

    def test_wraps_simple_agent(self) -> None:
        """Test wrapping SimpleAgent."""
        agent = SimpleAgent(lambda x: f"simple: {x}")
        wrapped = _wrap_agent(agent)

        assert callable(wrapped)
        # SimpleAgent is async capable, so should return async wrapper
        # But we can test the sync case by checking SimpleAgent directly

    @pytest.mark.asyncio
    async def test_wraps_simple_agent_async(self) -> None:
        """Test wrapping SimpleAgent with async execution."""
        agent = SimpleAgent(lambda x: f"simple: {x}")
        wrapped = _wrap_agent(agent)

        # SimpleAgent has a_execute, so wrapped should be async
        result = await wrapped("test")
        assert result == "simple: test"

    def test_wraps_langchain_wrapper(self) -> None:
        """Test that LangChain wrappers work with _wrap_agent."""

        class MockLangChainAgent:
            def run(self, input: Any) -> str:
                return f"langchain: {input}"

        lc_agent = MockLangChainAgent()
        wrapper = LangChainAgentWrapper(lc_agent)

        # LangChainAgentWrapper implements AgentInterface
        wrapped = _wrap_agent(wrapper)

        assert callable(wrapped)
        # Wrapper has async support, but we'll just verify it's callable

    def test_raises_for_non_agent(self) -> None:
        """Test that non-agent raises TypeError."""
        not_agent = NotAnAgent()

        with pytest.raises(TypeError, match="must be either a callable"):
            _wrap_agent(not_agent)

    def test_raises_for_none(self) -> None:
        """Test that None raises TypeError."""
        with pytest.raises(TypeError, match="must be either a callable"):
            _wrap_agent(None)

    def test_raises_for_string(self) -> None:
        """Test that string raises TypeError."""
        with pytest.raises(TypeError, match="must be either a callable"):
            _wrap_agent("not an agent")

    def test_raises_for_dict(self) -> None:
        """Test that dict raises TypeError."""
        with pytest.raises(TypeError, match="must be either a callable"):
            _wrap_agent({"not": "an agent"})

    async def test_wrapped_async_function(self) -> None:
        """Test that async functions are returned as-is."""

        async def my_async_function(x: Any) -> str:
            return f"async result: {x}"

        wrapped = _wrap_agent(my_async_function)
        assert wrapped is my_async_function
        result = await wrapped("test")
        assert result == "async result: test"

    def test_wrapped_callable_class(self) -> None:
        """Test that callable classes are returned as-is."""

        class CallableClass:
            def __call__(self, x: Any) -> str:
                return f"callable: {x}"

        callable_obj = CallableClass()
        wrapped = _wrap_agent(callable_obj)
        assert wrapped is callable_obj
        assert wrapped("test") == "callable: test"
