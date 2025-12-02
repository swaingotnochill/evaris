"""Integration tests for the evaluate() function.

These tests verify the complete evaluation workflow from input to output.
"""

from typing import Any

import pytest

from evaris import EvalResult, TestCase, evaluate


# Test agent functions
def simple_echo_agent(input_data: Any) -> str:
    """Simple agent that echoes input."""
    return str(input_data)


def greeting_agent(name: str) -> str:
    """Agent that greets by name."""
    return f"Hello {name}"


def math_agent(query: str) -> str:
    """Agent that answers simple math questions."""
    if "2+2" in query:
        return "4"
    elif "3+3" in query:
        return "6"
    return "I don't know"


class TestEvaluateEndToEnd:
    """End-to-end integration tests for evaluate()."""

    def test_evaluate_single_test_case_exact_match(self) -> None:
        """Test complete evaluation with single test case."""
        result = evaluate(
            name="test-eval",
            task=greeting_agent,
            data=[{"input": "World", "expected": "Hello World"}],
            metrics=["exact_match"],
        )

        assert isinstance(result, EvalResult)
        assert result.name == "test-eval"
        assert result.total == 1
        assert result.passed == 1
        assert result.failed == 0
        assert result.accuracy == 1.0
        assert len(result.results) == 1
        assert result.results[0].output == "Hello World"

    def test_evaluate_multiple_test_cases(self) -> None:
        """Test evaluation with multiple test cases."""
        test_data = [
            {"input": "Alice", "expected": "Hello Alice"},
            {"input": "Bob", "expected": "Hello Bob"},
            {"input": "Charlie", "expected": "Hello Charlie"},
        ]

        result = evaluate(
            name="multi-test",
            task=greeting_agent,
            data=test_data,
            metrics=["exact_match"],
        )

        assert result.total == 3
        assert result.passed == 3
        assert result.failed == 0
        assert result.accuracy == 1.0
        assert len(result.results) == 3

    def test_evaluate_with_failures(self) -> None:
        """Test evaluation where some test cases fail."""
        test_data = [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "What is 3+3?", "expected": "6"},
            {"input": "What is 5+5?", "expected": "10"},
        ]

        result = evaluate(
            name="math-test",
            task=math_agent,
            data=test_data,
            metrics=["exact_match"],
        )

        assert result.total == 3
        assert result.passed == 2
        assert result.failed == 1
        assert result.accuracy == pytest.approx(2 / 3, rel=1e-2)
        assert result.results[2].metrics[0].passed is False

    def test_evaluate_tracks_latency(self) -> None:
        """Test that evaluation tracks latency for each test case."""
        result = evaluate(
            name="latency-test",
            task=greeting_agent,
            data=[{"input": "Test", "expected": "Hello Test"}],
            metrics=["exact_match"],
        )

        assert result.results[0].latency_ms > 0
        assert result.avg_latency_ms > 0

    def test_evaluate_with_testcase_objects(self) -> None:
        """Test evaluation with TestCase objects instead of dicts."""
        test_cases = [
            TestCase(input="Alice", expected="Hello Alice", actual_output="Hello Alice"),
            TestCase(input="Bob", expected="Hello Bob", actual_output="Hello Bob"),
        ]

        result = evaluate(
            name="testcase-obj-test",
            task=greeting_agent,
            data=test_cases,
            metrics=["exact_match"],
        )

        assert result.total == 2
        assert result.passed == 2
        assert result.accuracy == 1.0

    def test_multiple_metrics(self) -> None:
        """Test evaluation with multiple metrics."""
        result = evaluate(
            name="multi-metric",
            task=greeting_agent,
            data=[{"input": "Alice", "expected": "Hello Alice"}],
            metrics=["exact_match", "latency"],
        )

        assert len(result.results[0].metrics) == 2
        metric_names = {m.name for m in result.results[0].metrics}
        assert "exact_match" in metric_names
        assert "latency" in metric_names


class TestEvaluateLatencyMetric:
    """Integration tests for latency metric."""

    def test_latency_metric_measured(self) -> None:
        """Test that latency metric is properly measured."""
        result = evaluate(
            name="latency-only",
            task=greeting_agent,
            data=[{"input": "Test"}],
            metrics=["latency"],
        )

        assert result.total == 1
        assert result.results[0].latency_ms > 0
        latency_metric = next(m for m in result.results[0].metrics if m.name == "latency")
        assert latency_metric.passed is True


class TestEvaluateErrorHandling:
    """Integration tests for error handling."""

    def test_evaluate_empty_data_raises_error(self) -> None:
        """Test that empty data list raises appropriate error."""
        with pytest.raises(ValueError, match="at least one test case"):
            evaluate(
                name="empty-test",
                task=greeting_agent,
                data=[],
                metrics=["exact_match"],
            )

    def test_evaluate_invalid_metric_raises_error(self) -> None:
        """Test that invalid metric name raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate(
                name="invalid-metric",
                task=greeting_agent,
                data=[{"input": "test"}],
                metrics=["invalid_metric_name"],
            )

    def test_evaluate_handles_agent_exceptions(self) -> None:
        """Test that agent exceptions are caught and recorded."""

        def failing_agent(input_data: Any) -> str:
            raise RuntimeError("Agent failed!")

        result = evaluate(
            name="failing-agent",
            task=failing_agent,
            data=[{"input": "test", "expected": "output"}],
            metrics=["exact_match"],
        )

        assert result.total == 1
        assert result.failed == 1
        assert result.results[0].error is not None
        assert "Agent failed!" in result.results[0].error

    def test_evaluate_test_case_without_expected_for_exact_match(self) -> None:
        """Test that exact_match metric handles missing expected value gracefully."""
        result = evaluate(
            name="no-expected",
            task=greeting_agent,
            data=[{"input": "test"}],
            metrics=["exact_match"],
        )

        assert result.total == 1
        assert result.failed == 1
        assert len(result.results[0].metrics) == 1
        assert result.results[0].metrics[0].passed is False
        metadata = result.results[0].metrics[0].metadata
        assert metadata is not None
        assert "expected" in metadata.get("error", "")


class TestEvaluateInputValidation:
    """Integration tests for input validation."""

    def test_evaluate_requires_name(self) -> None:
        """Test that evaluation name is required."""
        with pytest.raises(TypeError):
            evaluate(  # type: ignore
                task=greeting_agent,
                data=[{"input": "test"}],
                metrics=["exact_match"],
            )

    def test_evaluate_requires_task(self) -> None:
        """Test that task function is required."""
        with pytest.raises(TypeError):
            evaluate(  # type: ignore
                name="test",
                data=[{"input": "test"}],
                metrics=["exact_match"],
            )

    def test_evaluate_requires_data(self) -> None:
        """Test that data is required."""
        with pytest.raises(TypeError):
            evaluate(  # type: ignore
                name="test",
                task=greeting_agent,
                metrics=["exact_match"],
            )

    def test_evaluate_requires_metrics(self) -> None:
        """Test that metrics list is required."""
        with pytest.raises(TypeError):
            evaluate(  # type: ignore
                name="test",
                task=greeting_agent,
                data=[{"input": "test"}],
            )
