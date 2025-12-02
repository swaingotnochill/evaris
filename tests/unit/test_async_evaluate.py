"""Tests for async evaluation functionality."""

import asyncio

import pytest

from evaris import evaluate, evaluate_async, evaluate_stream, evaluate_sync
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.types import EvalResult, TestResult


class TestAsyncEvaluate:
    """Test async evaluate functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_async_basic(self) -> None:
        """Test basic async evaluation."""

        async def async_agent(input_text: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return input_text.upper()

        result = await evaluate_async(
            name="test-async-basic",
            task=async_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        )

        assert isinstance(result, EvalResult)
        assert result.total == 2
        assert result.passed == 2
        assert result.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_async_with_sync_agent(self) -> None:
        """Test async evaluation with sync agent (should run in thread pool)."""

        def sync_agent(input_text: str) -> str:
            return input_text.upper()

        result = await evaluate_async(
            name="test-async-sync-agent",
            task=sync_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        )

        assert isinstance(result, EvalResult)
        assert result.total == 2
        assert result.passed == 2
        assert result.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_async_concurrency(self) -> None:
        """Test async evaluation with concurrency control."""
        execution_order = []

        async def async_agent(input_text: str) -> str:
            execution_order.append(f"start-{input_text}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end-{input_text}")
            return input_text

        result = await evaluate_async(
            name="test-async-concurrency",
            task=async_agent,
            data=[{"input": str(i), "expected": str(i)} for i in range(5)],
            metrics=[ExactMatchMetric()],
            max_concurrency=2,  # Limit to 2 concurrent executions
        )

        assert result.total == 5
        assert result.passed == 5
        # Verify that not all started before any ended (indicating concurrency control)
        start_count_before_first_end = 0
        for item in execution_order:
            if item.startswith("end-"):
                break
            if item.startswith("start-"):
                start_count_before_first_end += 1
        # Should be at most 2 (concurrency limit)
        assert start_count_before_first_end <= 2

    @pytest.mark.asyncio
    async def test_evaluate_stream_basic(self) -> None:
        """Test streaming evaluation results."""

        async def async_agent(input_text: str) -> str:
            await asyncio.sleep(0.01)
            return input_text.upper()

        results = []
        async for result in evaluate_stream(
            name="test-stream-basic",
            task=async_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        ):
            assert isinstance(result, TestResult)
            results.append(result)

        assert len(results) == 2
        assert all(r.error is None for r in results)
        assert all(m.passed for r in results for m in r.metrics)

    @pytest.mark.asyncio
    async def test_evaluate_stream_early_access(self) -> None:
        """Test that streaming gives early access to results."""

        async def async_agent(input_text: str) -> str:
            await asyncio.sleep(0.01)
            return input_text

        first_result = None
        result_count = 0

        async for result in evaluate_stream(
            name="test-stream-early",
            task=async_agent,
            data=[{"input": str(i), "expected": str(i)} for i in range(10)],
            metrics=[ExactMatchMetric()],
        ):
            if first_result is None:
                first_result = result
            result_count += 1

            # If we can access the first result before all 10 complete, streaming works
            if result_count == 1:
                assert first_result is not None
                break

        assert first_result is not None
        assert result_count >= 1


class TestSmartRouting:
    """Test smart routing between sync and async."""

    def test_evaluate_with_sync_agent_uses_sync(self) -> None:
        """Test that sync agent routes to sync evaluation."""

        def sync_agent(input_text: str) -> str:
            return input_text.upper()

        result = evaluate(
            name="test-sync-routing",
            task=sync_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        )

        assert isinstance(result, EvalResult)
        assert result.total == 2
        assert result.passed == 2

    def test_evaluate_with_async_agent_uses_async(self) -> None:
        """Test that async agent routes to async evaluation."""

        async def async_agent(input_text: str) -> str:
            await asyncio.sleep(0.01)
            return input_text.upper()

        result = evaluate(
            name="test-async-routing",
            task=async_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        )

        assert isinstance(result, EvalResult)
        assert result.total == 2
        assert result.passed == 2

    def test_evaluate_sync_explicit(self) -> None:
        """Test explicit sync evaluation."""

        def sync_agent(input_text: str) -> str:
            return input_text.upper()

        result = evaluate_sync(
            name="test-explicit-sync",
            task=sync_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        )

        assert isinstance(result, EvalResult)
        assert result.total == 2
        assert result.passed == 2

    @pytest.mark.asyncio
    async def test_evaluate_async_explicit(self) -> None:
        """Test explicit async evaluation."""

        async def async_agent(input_text: str) -> str:
            await asyncio.sleep(0.01)
            return input_text.upper()

        result = await evaluate_async(
            name="test-explicit-async",
            task=async_agent,
            data=[
                {"input": "hello", "expected": "HELLO"},
                {"input": "world", "expected": "WORLD"},
            ],
            metrics=[ExactMatchMetric()],
        )

        assert isinstance(result, EvalResult)
        assert result.total == 2
        assert result.passed == 2


class TestAsyncErrorHandling:
    """Test error handling in async evaluation."""

    @pytest.mark.asyncio
    async def test_async_agent_exception_captured(self) -> None:
        """Test that async agent exceptions are captured in results."""

        async def failing_agent(input_text: str) -> str:
            if input_text == "fail":
                raise ValueError("Test error")
            return input_text

        # Should not raise, even though agent fails
        result = await evaluate_async(
            name="test-async-error",
            task=failing_agent,
            data=[
                {"input": "pass", "expected": "pass"},
                {"input": "fail", "expected": "fail"},
            ],
            metrics=[ExactMatchMetric()],
        )

        # Evaluation should complete
        assert result.total == 2
        # At least one test should pass (the "pass" input)
        assert result.passed >= 1

    @pytest.mark.asyncio
    async def test_async_empty_data_raises(self) -> None:
        """Test that empty data raises ValueError."""

        async def async_agent(input_text: str) -> str:
            return input_text

        with pytest.raises(ValueError, match="at least one test case"):
            await evaluate_async(
                name="test-empty-data",
                task=async_agent,
                data=[],
                metrics=[ExactMatchMetric()],
            )
