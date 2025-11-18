"""Integration tests for evaluating agents with AgentInterface."""

from typing import Any

import pytest

from evaris import evaluate, evaluate_async, evaluate_sync
from evaris.agent_interface import SimpleAgent
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.wrappers.langchain import LangChainAgentWrapper


class MockGreetingAgent:
    """Mock agent that greets names."""

    def execute(self, name: str) -> str:
        """Execute greeting synchronously."""
        return f"Hello {name}"


class MockAsyncGreetingAgent:
    """Mock async agent that greets names."""

    async def a_execute(self, name: str) -> str:
        """Execute greeting asynchronously."""
        return f"Hello {name}"


class MockBothGreetingAgent:
    """Mock agent with both sync and async methods."""

    def execute(self, name: str) -> str:
        """Execute greeting synchronously."""
        return f"Hello {name}"

    async def a_execute(self, name: str) -> str:
        """Execute greeting asynchronously."""
        return f"Hello {name}"


class MockLangChainAgent:
    """Mock LangChain agent for testing."""

    def run(self, input: str) -> str:
        """Legacy LangChain run method."""
        return f"LangChain says: {input}"


class TestSyncAgentEvaluation:
    """Tests for evaluating sync agents."""

    def test_evaluate_sync_agent_with_exact_match(self) -> None:
        """Test evaluating a sync agent with exact match metric."""
        agent = MockGreetingAgent()

        data = [
            {"input": "World", "expected": "Hello World"},
            {"input": "Alice", "expected": "Hello Alice"},
            {"input": "Bob", "expected": "Hello Bob"},
        ]

        result = evaluate(
            name="greeting-test",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.name == "greeting-test"
        assert result.accuracy == 1.0
        assert len(result.results) == 3
        assert all(r.metrics[0].passed for r in result.results)

    def test_evaluate_sync_explicit(self) -> None:
        """Test using evaluate_sync explicitly with agent."""
        agent = MockGreetingAgent()

        data = [{"input": "World", "expected": "Hello World"}]

        result = evaluate_sync(
            name="sync-explicit",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0

    def test_evaluate_sync_agent_with_metric_instance(self) -> None:
        """Test evaluating with metric instance instead of string."""
        agent = MockGreetingAgent()

        data = [{"input": "World", "expected": "Hello World"}]

        result = evaluate(
            name="metric-instance-test",
            task=agent,
            data=data,
            metrics=[ExactMatchMetric()],
        )

        assert result.accuracy == 1.0

    def test_evaluate_sync_agent_partial_match(self) -> None:
        """Test evaluating agent with some failures."""
        agent = MockGreetingAgent()

        data = [
            {"input": "World", "expected": "Hello World"},  # Pass
            {"input": "Alice", "expected": "Hi Alice"},  # Fail
            {"input": "Bob", "expected": "Hello Bob"},  # Pass
        ]

        result = evaluate(
            name="partial-match",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 2 / 3  # 2 out of 3 passed
        assert result.results[0].metrics[0].passed is True
        assert result.results[1].metrics[0].passed is False
        assert result.results[2].metrics[0].passed is True

    def test_evaluate_simple_agent(self) -> None:
        """Test evaluating SimpleAgent."""
        agent = SimpleAgent(lambda name: f"Hello {name}")

        data = [{"input": "World", "expected": "Hello World"}]

        result = evaluate(
            name="simple-agent-test",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0


class TestAsyncAgentEvaluation:
    """Tests for evaluating async agents."""

    @pytest.mark.asyncio
    async def test_evaluate_async_agent_with_await(self) -> None:
        """Test evaluating async agent with await."""
        agent = MockAsyncGreetingAgent()

        data = [
            {"input": "World", "expected": "Hello World"},
            {"input": "Alice", "expected": "Hello Alice"},
        ]

        result = await evaluate_async(
            name="async-greeting",
            task=agent,
            data=data,
            metrics=["exact_match"],
            max_concurrency=2,
        )

        assert result.accuracy == 1.0
        assert len(result.results) == 2

    def test_evaluate_async_agent_smart_routing(self) -> None:
        """Test that async agent triggers smart routing."""
        agent = MockAsyncGreetingAgent()

        data = [{"input": "World", "expected": "Hello World"}]

        # Smart routing should detect async and use asyncio.run()
        result = evaluate(
            name="smart-routing-async",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_both_agent_prefers_async(self) -> None:
        """Test that agent with both methods prefers async."""
        agent = MockBothGreetingAgent()

        data = [{"input": "World", "expected": "Hello World"}]

        result = await evaluate_async(
            name="both-methods",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0


class TestLangChainAgentEvaluation:
    """Tests for evaluating LangChain agents."""

    def test_evaluate_langchain_wrapped_agent(self) -> None:
        """Test evaluating LangChain agent with wrapper."""
        lc_agent = MockLangChainAgent()
        wrapped = LangChainAgentWrapper(lc_agent)

        data = [
            {"input": "Hello", "expected": "LangChain says: Hello"},
            {"input": "World", "expected": "LangChain says: World"},
        ]

        result = evaluate(
            name="langchain-test",
            task=wrapped,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0
        assert len(result.results) == 2


class TestFunctionAgentEvaluation:
    """Tests for evaluating plain functions (backward compatibility)."""

    def test_evaluate_plain_function(self) -> None:
        """Test that plain functions still work."""

        def my_agent(name: str) -> str:
            return f"Hello {name}"

        data = [{"input": "World", "expected": "Hello World"}]

        result = evaluate(
            name="function-test",
            task=my_agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0

    async def test_evaluate_async_function(self) -> None:
        """Test that async functions still work."""

        async def my_async_agent(name: str) -> str:
            return f"Hello {name}"

        data = [{"input": "World", "expected": "Hello World"}]

        result = await evaluate_async(
            name="async-function-test",
            task=my_async_agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0


class TestAgentWithLatency:
    """Tests for latency tracking with agents."""

    def test_sync_agent_tracks_latency(self) -> None:
        """Test that latency is tracked for sync agents."""
        agent = MockGreetingAgent()

        data = [{"input": "World", "expected": "Hello World"}]

        result = evaluate(
            name="latency-test",
            task=agent,
            data=data,
            metrics=["exact_match", "latency"],
            baselines=False,  # Disable baselines for latency test
        )

        assert result.accuracy == 1.0
        # Check that latency was recorded
        assert result.results[0].latency_ms > 0
        latency_metric = [m for m in result.results[0].metrics if m.name == "latency"][0]
        assert latency_metric.passed is True
        assert "latency_ms" in latency_metric.metadata


class TestErrorHandling:
    """Tests for error handling during agent evaluation."""

    def test_agent_error_is_recorded(self) -> None:
        """Test that agent errors are properly recorded."""

        class FailingAgent:
            def execute(self, input: Any) -> str:
                raise ValueError("Agent failed!")

        agent = FailingAgent()

        data = [{"input": "test", "expected": "anything"}]

        result = evaluate(
            name="error-test",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        # Error should be recorded in test result
        assert result.results[0].error is not None
        assert "Agent failed!" in result.results[0].error
        # Agent error means no metrics are computed
        assert len(result.results[0].metrics) == 0


class TestMixedDataFormats:
    """Tests for evaluating agents with different data formats."""

    def test_evaluate_with_golden_data(self) -> None:
        """Test evaluating with Golden objects (no actual_output)."""
        agent = MockGreetingAgent()

        # Data without actual_output - agent will be called
        data = [
            {"input": "World", "expected": "Hello World"},
            {"input": "Alice", "expected": "Hello Alice"},
        ]

        result = evaluate(
            name="golden-test",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0

    def test_evaluate_with_testcase_data(self) -> None:
        """Test evaluating with TestCase objects (with actual_output)."""

        # Data with actual_output - agent won't be called
        data = [
            {"input": "World", "actual_output": "Hello World", "expected": "Hello World"},
            {"input": "Alice", "actual_output": "Hello Alice", "expected": "Hello Alice"},
        ]

        # Pass a dummy agent - it won't be called since actual_output is provided
        agent = MockGreetingAgent()

        result = evaluate(
            name="testcase-test",
            task=agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.accuracy == 1.0
