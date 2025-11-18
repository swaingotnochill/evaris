"""Tests for BaseAgentWrapper."""

from typing import Any, Optional, cast

import pytest

from evaris.types import MultiModalInput, MultiModalOutput
from evaris.wrappers.base import BaseAgentWrapper


class ConcreteWrapper(BaseAgentWrapper):
    """Concrete implementation for testing."""

    def execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent synchronously."""
        normalized = self._normalize_input(input)
        result = f"executed: {normalized}"
        return self._normalize_output(result)

    async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
        """Execute agent asynchronously."""
        normalized = self._normalize_input(input)
        result = f"async executed: {normalized}"
        return self._normalize_output(result)


class TestBaseAgentWrapper:
    """Tests for BaseAgentWrapper base class."""

    def test_initialization_with_defaults(self) -> None:
        """Test wrapper initialization with default parameters."""

        class DummyAgent:
            def run(self, x: Any) -> Any:
                return x

        agent = DummyAgent()
        wrapper = ConcreteWrapper(agent)

        assert wrapper.agent is agent
        assert wrapper.trace_llm_calls is False
        assert wrapper.trace_tool_calls is False
        assert wrapper.track_cost is False

    def test_initialization_with_custom_params(self) -> None:
        """Test wrapper initialization with custom parameters."""

        class DummyAgent:
            def run(self, x: Any) -> Any:
                return x

        agent = DummyAgent()
        wrapper = ConcreteWrapper(
            agent, trace_llm_calls=True, trace_tool_calls=True, track_cost=True
        )

        assert wrapper.trace_llm_calls is True
        assert wrapper.trace_tool_calls is True
        assert wrapper.track_cost is True

    def test_normalize_input_string(self) -> None:
        """Test input normalization for string."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        result = wrapper._normalize_input("test input")
        assert result == "test input"

    def test_normalize_input_dict(self) -> None:
        """Test input normalization for dict."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        input_dict = {"key": "value"}
        result = wrapper._normalize_input(input_dict)
        assert result == input_dict

    def test_normalize_input_list(self) -> None:
        """Test input normalization for list."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        input_list = ["a", "b", "c"]
        result = wrapper._normalize_input(cast(Any, input_list))
        assert result == input_list

    def test_normalize_input_path(self) -> None:
        """Test input normalization for Path object."""
        from pathlib import Path

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        input_path = Path("/tmp/test.txt")
        result = wrapper._normalize_input(input_path)
        assert result == str(input_path)
        assert isinstance(result, str)

    def test_normalize_input_other_types(self) -> None:
        """Test input normalization for other types (converts to string)."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())

        # Integer
        assert wrapper._normalize_input(cast(Any, 42)) == "42"

        # Float
        assert wrapper._normalize_input(cast(Any, 3.14)) == "3.14"

        # Custom object
        class CustomObj:
            def __str__(self) -> str:
                return "custom"

        assert wrapper._normalize_input(cast(Any, CustomObj())) == "custom"

    def test_normalize_output_string(self) -> None:
        """Test output normalization for string."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        result = wrapper._normalize_output("test output")
        assert result == "test output"

    def test_normalize_output_dict(self) -> None:
        """Test output normalization for dict."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        output_dict = {"result": "value"}
        result = wrapper._normalize_output(output_dict)
        assert result == output_dict

    def test_normalize_output_list(self) -> None:
        """Test output normalization for list."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        output_list = ["x", "y", "z"]
        result = wrapper._normalize_output(output_list)
        assert result == output_list

    def test_normalize_output_other_types(self) -> None:
        """Test output normalization for other types (converts to string)."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())

        # Integer
        assert wrapper._normalize_output(cast(Any, 42)) == "42"

        # Custom object
        class CustomObj:
            def __str__(self) -> str:
                return "output"

        assert wrapper._normalize_output(cast(Any, CustomObj())) == "output"

    def test_reset_is_noop_by_default(self) -> None:
        """Test that reset is a no-op in base class."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        wrapper.reset()  # Should not raise

    def test_extract_cost_returns_none_by_default(self) -> None:
        """Test that _extract_cost returns None by default."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        cost = wrapper._extract_cost("some response")
        assert cost is None

    def test_extract_tokens_returns_none_by_default(self) -> None:
        """Test that _extract_tokens returns None by default."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        tokens = wrapper._extract_tokens("some response")
        assert tokens is None

    def test_concrete_execute(self) -> None:
        """Test concrete implementation of execute."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        result = wrapper.execute("test")
        assert result == "executed: test"

    @pytest.mark.asyncio
    async def test_concrete_async_execute(self) -> None:
        """Test concrete implementation of a_execute."""

        class DummyAgent:
            pass

        wrapper = ConcreteWrapper(DummyAgent())
        result = await wrapper.a_execute("test")
        assert result == "async executed: test"

    def test_custom_wrapper_can_override_normalization(self) -> None:
        """Test that subclass can override normalization methods."""

        class CustomWrapper(BaseAgentWrapper):
            def _normalize_input(self, input: Any) -> str:
                return f"custom_in: {input}"

            def _normalize_output(self, output: Any) -> str:
                return f"custom_out: {output}"

            def execute(self, input: MultiModalInput) -> MultiModalOutput:
                normalized = self._normalize_input(input)
                return self._normalize_output(normalized)

            async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
                return self.execute(input)

        class DummyAgent:
            pass

        wrapper = CustomWrapper(DummyAgent())
        result = wrapper.execute("test")
        assert result == "custom_out: custom_in: test"

    def test_custom_wrapper_can_override_cost_extraction(self) -> None:
        """Test that subclass can override cost extraction."""

        class CostTrackingWrapper(BaseAgentWrapper):
            def _extract_cost(self, response: Any) -> Optional[float]:
                if isinstance(response, dict) and "cost" in response:
                    return float(response["cost"])
                return None

            def _extract_tokens(self, response: Any) -> Optional[int]:
                if isinstance(response, dict) and "tokens" in response:
                    return int(response["tokens"])
                return None

            def execute(self, input: MultiModalInput) -> MultiModalOutput:
                return {"result": "ok", "cost": 0.05, "tokens": 100}

            async def a_execute(self, input: MultiModalInput) -> MultiModalOutput:
                return self.execute(input)

        class DummyAgent:
            pass

        wrapper = CostTrackingWrapper(DummyAgent())
        response = wrapper.execute("test")

        cost = wrapper._extract_cost(response)
        assert cost == 0.05

        tokens = wrapper._extract_tokens(response)
        assert tokens == 100
