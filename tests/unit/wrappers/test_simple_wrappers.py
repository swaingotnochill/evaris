"""Tests for simple agent wrappers (SyncAgentWrapper, AsyncAgentWrapper, CallableAgentWrapper)."""

import asyncio
from typing import Any, cast

import pytest

from evaris.wrappers.simple import AsyncAgentWrapper, CallableAgentWrapper, SyncAgentWrapper


class TestSyncAgentWrapper:
    """Tests for SyncAgentWrapper."""

    def test_wraps_sync_function(self) -> None:
        """Test wrapping a simple sync function."""

        def my_agent(input: Any) -> Any:
            return f"processed: {input}"

        wrapped = SyncAgentWrapper(my_agent)
        result = wrapped.execute("test")
        assert result == "processed: test"

    @pytest.mark.asyncio
    async def test_async_execute_uses_thread_pool(self) -> None:
        """Test that async execution runs sync agent in thread pool."""

        def my_agent(input: Any) -> Any:
            return f"processed: {input}"

        wrapped = SyncAgentWrapper(my_agent)
        result = await wrapped.a_execute("test")
        assert result == "processed: test"

    def test_raises_type_error_for_non_callable(self) -> None:
        """Test that non-callable raises TypeError."""
        with pytest.raises(TypeError, match="Agent must be callable"):
            SyncAgentWrapper(cast(Any, "not a function"))

    def test_handles_dict_input_output(self) -> None:
        """Test handling dict input and output."""

        def my_agent(input: Any) -> Any:
            return {"result": input["value"] * 2}

        wrapped = SyncAgentWrapper(my_agent)
        result = wrapped.execute({"value": 5})
        assert result == {"result": 10}

    def test_handles_list_input(self) -> None:
        """Test handling list input."""

        def my_agent(input: Any) -> Any:
            return [x.upper() for x in input]

        wrapped = SyncAgentWrapper(my_agent)
        result = wrapped.execute(["a", "b", "c"])
        assert result == ["A", "B", "C"]

    def test_handles_exception_in_agent(self) -> None:
        """Test that exceptions in agent are propagated."""

        def my_agent(input: Any) -> Any:
            raise ValueError("Agent error")

        wrapped = SyncAgentWrapper(my_agent)
        with pytest.raises(ValueError, match="Agent error"):
            wrapped.execute("test")

    @pytest.mark.asyncio
    async def test_async_execute_propagates_exception(self) -> None:
        """Test that async execution propagates exceptions."""

        def my_agent(input: Any) -> Any:
            raise ValueError("Agent error")

        wrapped = SyncAgentWrapper(my_agent)
        with pytest.raises(ValueError, match="Agent error"):
            await wrapped.a_execute("test")

    def test_reset_is_noop(self) -> None:
        """Test that reset is a no-op for stateless wrapper."""
        wrapped = SyncAgentWrapper(lambda x: x)
        wrapped.reset()  # Should not raise


class TestAsyncAgentWrapper:
    """Tests for AsyncAgentWrapper."""

    @pytest.mark.asyncio
    async def test_wraps_async_function(self) -> None:
        """Test wrapping a simple async function."""

        async def my_agent(input: Any) -> Any:
            await asyncio.sleep(0.01)
            return f"processed: {input}"

        wrapped = AsyncAgentWrapper(my_agent)
        result = await wrapped.a_execute("test")
        assert result == "processed: test"

    def test_sync_execute_runs_in_event_loop(self) -> None:
        """Test that sync execution creates event loop."""

        async def my_agent(input: Any) -> Any:
            await asyncio.sleep(0.01)
            return f"processed: {input}"

        wrapped = AsyncAgentWrapper(my_agent)
        result = wrapped.execute("test")
        assert result == "processed: test"

    def test_raises_type_error_for_non_callable(self) -> None:
        """Test that non-callable raises TypeError."""
        with pytest.raises(TypeError, match="Agent must be callable"):
            AsyncAgentWrapper(cast(Any, "not a function"))

    def test_raises_type_error_for_sync_function(self) -> None:
        """Test that sync function raises TypeError."""

        def sync_agent(input: Any) -> Any:
            return input

        with pytest.raises(TypeError, match="must be an async function"):
            AsyncAgentWrapper(sync_agent)

    @pytest.mark.asyncio
    async def test_handles_dict_input_output(self) -> None:
        """Test handling dict input and output."""

        async def my_agent(input: Any) -> Any:
            return {"result": input["value"] * 2}

        wrapped = AsyncAgentWrapper(my_agent)
        result = await wrapped.a_execute({"value": 5})
        assert result == {"result": 10}

    @pytest.mark.asyncio
    async def test_handles_exception_in_agent(self) -> None:
        """Test that exceptions in agent are propagated."""

        async def my_agent(input: Any) -> Any:
            raise ValueError("Agent error")

        wrapped = AsyncAgentWrapper(my_agent)
        with pytest.raises(ValueError, match="Agent error"):
            await wrapped.a_execute("test")

    def test_sync_execute_propagates_exception(self) -> None:
        """Test that sync execution propagates exceptions."""

        async def my_agent(input: Any) -> Any:
            raise ValueError("Agent error")

        wrapped = AsyncAgentWrapper(my_agent)
        with pytest.raises(ValueError, match="Agent error"):
            wrapped.execute("test")

    def test_reset_is_noop(self) -> None:
        """Test that reset is a no-op for stateless wrapper."""

        async def my_agent(input: Any) -> Any:
            return input

        wrapped = AsyncAgentWrapper(my_agent)
        wrapped.reset()  # Should not raise


class TestCallableAgentWrapper:
    """Tests for CallableAgentWrapper (auto-detect sync/async)."""

    def test_detects_and_wraps_sync_function(self) -> None:
        """Test that it correctly detects and wraps sync function."""

        def my_agent(input: Any) -> Any:
            return f"sync: {input}"

        wrapped = CallableAgentWrapper(my_agent)
        result = wrapped.execute("test")
        assert result == "sync: test"

    @pytest.mark.asyncio
    async def test_detects_and_wraps_async_function(self) -> None:
        """Test that it correctly detects and wraps async function."""

        async def my_agent(input: Any) -> Any:
            return f"async: {input}"

        wrapped = CallableAgentWrapper(my_agent)
        result = await wrapped.a_execute("test")
        assert result == "async: test"

    def test_sync_function_sync_execute(self) -> None:
        """Test sync function with sync execute."""

        def my_agent(input: Any) -> Any:
            return f"processed: {input}"

        wrapped = CallableAgentWrapper(my_agent)
        result = wrapped.execute("test")
        assert result == "processed: test"

    @pytest.mark.asyncio
    async def test_sync_function_async_execute(self) -> None:
        """Test sync function with async execute (uses thread pool)."""

        def my_agent(input: Any) -> Any:
            return f"processed: {input}"

        wrapped = CallableAgentWrapper(my_agent)
        result = await wrapped.a_execute("test")
        assert result == "processed: test"

    def test_async_function_sync_execute(self) -> None:
        """Test async function with sync execute (creates event loop)."""

        async def my_agent(input: Any) -> Any:
            return f"processed: {input}"

        wrapped = CallableAgentWrapper(my_agent)
        result = wrapped.execute("test")
        assert result == "processed: test"

    @pytest.mark.asyncio
    async def test_async_function_async_execute(self) -> None:
        """Test async function with async execute."""

        async def my_agent(input: Any) -> Any:
            return f"processed: {input}"

        wrapped = CallableAgentWrapper(my_agent)
        result = await wrapped.a_execute("test")
        assert result == "processed: test"

    def test_raises_type_error_for_non_callable(self) -> None:
        """Test that non-callable raises TypeError."""
        with pytest.raises(TypeError, match="Agent must be callable"):
            CallableAgentWrapper(cast(Any, "not a function"))

    def test_handles_complex_inputs(self) -> None:
        """Test handling complex multimodal inputs."""

        def my_agent(input: Any) -> Any:
            if isinstance(input, dict):
                return {"processed": True, **input}
            return f"processed: {input}"

        wrapped = CallableAgentWrapper(my_agent)

        # Dict input
        result = wrapped.execute({"key": "value"})
        assert result == {"processed": True, "key": "value"}

        # String input
        result = wrapped.execute("test")
        assert result == "processed: test"

    @pytest.mark.asyncio
    async def test_concurrent_async_execution(self) -> None:
        """Test that multiple async executions can run concurrently."""

        async def my_agent(input: Any) -> Any:
            await asyncio.sleep(0.01)
            return f"processed: {input}"

        wrapped = CallableAgentWrapper(my_agent)

        # Run 5 concurrent executions
        results = await asyncio.gather(*[wrapped.a_execute(f"test{i}") for i in range(5)])

        assert len(results) == 5
        assert all(isinstance(r, str) and r.startswith("processed: test") for r in results)

    def test_exception_handling_sync(self) -> None:
        """Test exception handling in sync mode."""

        def my_agent(input: Any) -> Any:
            if input == "error":
                raise ValueError("Test error")
            return input

        wrapped = CallableAgentWrapper(my_agent)

        # Normal execution
        assert wrapped.execute("ok") == "ok"

        # Error case
        with pytest.raises(ValueError, match="Test error"):
            wrapped.execute("error")

    @pytest.mark.asyncio
    async def test_exception_handling_async(self) -> None:
        """Test exception handling in async mode."""

        async def my_agent(input: Any) -> Any:
            if input == "error":
                raise ValueError("Test error")
            return input

        wrapped = CallableAgentWrapper(my_agent)

        # Normal execution
        assert await wrapped.a_execute("ok") == "ok"

        # Error case
        with pytest.raises(ValueError, match="Test error"):
            await wrapped.a_execute("error")
