"""Tests for AgentInterface protocol and helper functions."""

from typing import Any

import pytest

from evaris.agent_interface import (
    SimpleAgent,
    is_async_agent,
    is_stateful_agent,
    is_sync_agent,
    validate_agent,
)


class TestAgentInterfaceProtocol:
    """Tests for AgentInterface protocol."""

    def test_sync_only_agent_is_valid(self) -> None:
        """Test that sync-only agent is valid via helper functions."""

        class SyncOnlyAgent:
            def execute(self, input: Any) -> Any:
                return f"sync: {input}"

        agent = SyncOnlyAgent()
        assert is_sync_agent(agent)
        assert not is_async_agent(agent)
        validate_agent(agent)  # Should not raise

    def test_async_only_agent_is_valid(self) -> None:
        """Test that async-only agent is valid via helper functions."""

        class AsyncOnlyAgent:
            async def a_execute(self, input: Any) -> Any:
                return f"async: {input}"

        agent = AsyncOnlyAgent()
        assert is_async_agent(agent)
        assert not is_sync_agent(agent)
        validate_agent(agent)  # Should not raise

    def test_both_sync_async_agent_is_valid(self) -> None:
        """Test that agent with both methods is valid."""

        class BothAgent:
            def execute(self, input: Any) -> Any:
                return f"sync: {input}"

            async def a_execute(self, input: Any) -> Any:
                return f"async: {input}"

        agent = BothAgent()
        assert is_sync_agent(agent)
        assert is_async_agent(agent)
        validate_agent(agent)  # Should not raise

    def test_stateful_agent_with_reset(self) -> None:
        """Test that agent with reset method is recognized."""

        class StatefulAgent:
            def __init__(self) -> None:
                self.state: list[Any] = []

            def execute(self, input: Any) -> Any:
                self.state.append(input)
                return str(self.state)

            def reset(self) -> None:
                self.state = []

        agent = StatefulAgent()
        assert is_sync_agent(agent)
        assert is_stateful_agent(agent)
        validate_agent(agent)  # Should not raise

    def test_non_agent_is_invalid(self) -> None:
        """Test that object without execute/a_execute is invalid."""

        class NotAnAgent:
            def run(self, input: Any) -> Any:
                return input

        obj = NotAnAgent()
        assert not is_sync_agent(obj)
        assert not is_async_agent(obj)
        with pytest.raises(TypeError):
            validate_agent(obj)


class TestIsAsyncAgent:
    """Tests for is_async_agent helper."""

    def test_async_agent_returns_true(self) -> None:
        """Test that agent with a_execute returns True."""

        class AsyncAgent:
            async def a_execute(self, input: Any) -> Any:
                return input

        agent = AsyncAgent()
        assert is_async_agent(agent) is True

    def test_sync_only_agent_returns_false(self) -> None:
        """Test that agent without a_execute returns False."""

        class SyncAgent:
            def execute(self, input: Any) -> Any:
                return input

        agent = SyncAgent()
        assert is_async_agent(agent) is False

    def test_both_methods_returns_true(self) -> None:
        """Test that agent with both methods returns True."""

        class BothAgent:
            def execute(self, input: Any) -> Any:
                return input

            async def a_execute(self, input: Any) -> Any:
                return input

        agent = BothAgent()
        assert is_async_agent(agent) is True

    def test_non_callable_a_execute_returns_false(self) -> None:
        """Test that non-callable a_execute attribute returns False."""

        class FakeAsyncAgent:
            a_execute = "not a function"

        agent = FakeAsyncAgent()
        assert is_async_agent(agent) is False


class TestIsSyncAgent:
    """Tests for is_sync_agent helper."""

    def test_sync_agent_returns_true(self) -> None:
        """Test that agent with execute returns True."""

        class SyncAgent:
            def execute(self, input: Any) -> Any:
                return input

        agent = SyncAgent()
        assert is_sync_agent(agent) is True

    def test_async_only_agent_returns_false(self) -> None:
        """Test that agent without execute returns False."""

        class AsyncAgent:
            async def a_execute(self, input: Any) -> Any:
                return input

        agent = AsyncAgent()
        assert is_sync_agent(agent) is False

    def test_both_methods_returns_true(self) -> None:
        """Test that agent with both methods returns True."""

        class BothAgent:
            def execute(self, input: Any) -> Any:
                return input

            async def a_execute(self, input: Any) -> Any:
                return input

        agent = BothAgent()
        assert is_sync_agent(agent) is True

    def test_non_callable_execute_returns_false(self) -> None:
        """Test that non-callable execute attribute returns False."""

        class FakeSyncAgent:
            execute = "not a function"

        agent = FakeSyncAgent()
        assert is_sync_agent(agent) is False


class TestIsStatefulAgent:
    """Tests for is_stateful_agent helper."""

    def test_agent_with_reset_returns_true(self) -> None:
        """Test that agent with reset method returns True."""

        class StatefulAgent:
            def execute(self, input: Any) -> Any:
                return input

            def reset(self) -> None:
                pass

        agent = StatefulAgent()
        assert is_stateful_agent(agent) is True

    def test_agent_without_reset_returns_false(self) -> None:
        """Test that agent without reset returns False."""

        class StatelessAgent:
            def execute(self, input: Any) -> Any:
                return input

        agent = StatelessAgent()
        assert is_stateful_agent(agent) is False

    def test_non_callable_reset_returns_false(self) -> None:
        """Test that non-callable reset attribute returns False."""

        class FakeStatefulAgent:
            def execute(self, input: Any) -> Any:
                return input

            reset = "not a function"

        agent = FakeStatefulAgent()
        assert is_stateful_agent(agent) is False


class TestValidateAgent:
    """Tests for validate_agent function."""

    def test_valid_sync_agent_passes(self) -> None:
        """Test that valid sync agent passes validation."""

        class SyncAgent:
            def execute(self, input: Any) -> Any:
                return input

        agent = SyncAgent()
        validate_agent(agent)  # Should not raise

    def test_valid_async_agent_passes(self) -> None:
        """Test that valid async agent passes validation."""

        class AsyncAgent:
            async def a_execute(self, input: Any) -> Any:
                return input

        agent = AsyncAgent()
        validate_agent(agent)  # Should not raise

    def test_invalid_agent_raises_type_error(self) -> None:
        """Test that invalid agent raises TypeError."""

        class InvalidAgent:
            def run(self, input: Any) -> Any:
                return input

        agent = InvalidAgent()
        with pytest.raises(TypeError, match="must implement at least one of execute"):
            validate_agent(agent)

    def test_error_message_includes_agent_type(self) -> None:
        """Test that error message includes agent type name."""

        class MyInvalidAgent:
            pass

        agent = MyInvalidAgent()
        with pytest.raises(TypeError, match="MyInvalidAgent"):
            validate_agent(agent)


class TestSimpleAgent:
    """Tests for SimpleAgent implementation."""

    def test_simple_agent_with_default_handler(self) -> None:
        """Test SimpleAgent with default echo handler."""
        agent = SimpleAgent()
        result = agent.execute("test")
        assert result == "test"

    def test_simple_agent_with_custom_handler(self) -> None:
        """Test SimpleAgent with custom handler."""
        agent = SimpleAgent(lambda x: f"processed: {x}")
        result = agent.execute("test")
        assert result == "processed: test"

    @pytest.mark.asyncio
    async def test_simple_agent_async_execute(self) -> None:
        """Test SimpleAgent async execution."""
        agent = SimpleAgent(lambda x: f"processed: {x}")
        result = await agent.a_execute("test")
        assert result == "processed: test"

    def test_simple_agent_reset_is_noop(self) -> None:
        """Test that SimpleAgent reset is a no-op."""
        agent = SimpleAgent()
        agent.reset()  # Should not raise

    def test_simple_agent_is_valid(self) -> None:
        """Test that SimpleAgent is valid agent."""
        agent = SimpleAgent()
        validate_agent(agent)  # Should not raise

    def test_simple_agent_is_sync(self) -> None:
        """Test that SimpleAgent is recognized as sync agent."""
        agent = SimpleAgent()
        assert is_sync_agent(agent) is True

    def test_simple_agent_is_async(self) -> None:
        """Test that SimpleAgent is recognized as async agent."""
        agent = SimpleAgent()
        assert is_async_agent(agent) is True

    def test_simple_agent_is_stateful(self) -> None:
        """Test that SimpleAgent is stateful (has reset method)."""
        agent = SimpleAgent()
        assert is_stateful_agent(agent) is True  # Has reset method

    def test_simple_agent_handles_dict_input(self) -> None:
        """Test SimpleAgent with dict input."""
        agent = SimpleAgent(lambda x: {"result": x})
        result = agent.execute({"key": "value"})
        assert result == {"result": {"key": "value"}}

    def test_simple_agent_handles_list_input(self) -> None:
        """Test SimpleAgent with list input."""
        agent = SimpleAgent(lambda x: [item.upper() for item in x])
        result = agent.execute(["a", "b", "c"])
        assert result == ["A", "B", "C"]
