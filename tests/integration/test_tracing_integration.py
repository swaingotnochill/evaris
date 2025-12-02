"""Integration tests for tracing with evaluation."""

from evaris import evaluate
from evaris.tracing import configure_debug_logging, configure_tracing, get_debug_logger, get_tracer
from evaris.types import TestCase


def simple_agent(input_text: str) -> str:
    """Simple test agent that echoes input."""
    return f"Response: {input_text}"


class TestTracingIntegration:
    """Integration tests for tracing with evaluation."""

    def setup_method(self) -> None:
        """Setup for each test."""
        # Reset global state
        import evaris.tracing

        evaris.tracing._global_tracer = None
        evaris.tracing._global_debug_logger = None

    def test_evaluate_with_tracing_disabled(self) -> None:
        """Test evaluation with tracing disabled."""
        configure_tracing(enabled=False)
        configure_debug_logging(enabled=False)

        data = [
            {"input": "test1", "expected": "Response: test1"},
            {"input": "test2", "expected": "Response: test2"},
        ]

        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=data,
            metrics=["exact_match"],
        )

        assert result.total == 2
        assert result.passed == 2
        assert result.accuracy == 1.0

    def test_evaluate_with_tracing_enabled_no_otel(self) -> None:
        """Test evaluation with tracing enabled but no OpenTelemetry."""
        # This should work gracefully without OpenTelemetry
        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=[{"input": "test", "expected": "Response: test"}],
            metrics=["exact_match"],
            enable_tracing=True,
        )

        assert result.total == 1
        assert result.passed == 1

    def test_evaluate_with_debug_logging_enabled(self) -> None:
        """Test evaluation with debug logging enabled."""
        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=[
                {"input": "test1", "expected": "Response: test1"},
                {"input": "test2", "expected": "Response: test2"},
            ],
            metrics=["exact_match"],
            enable_debug=True,
        )

        assert result.total == 2
        assert result.passed == 2

        # Verify debug logger was configured
        debug_logger = get_debug_logger()
        assert debug_logger.enabled

    def test_evaluate_with_both_tracing_and_debug(self) -> None:
        """Test evaluation with both tracing and debug logging."""
        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=[{"input": "test", "expected": "Response: test"}],
            metrics=["exact_match", "latency"],
            enable_tracing=True,
            enable_debug=True,
        )

        assert result.total == 1
        assert result.passed == 1

        # Verify both were configured
        get_tracer()
        debug_logger = get_debug_logger()
        assert debug_logger.enabled

    def test_evaluate_preserves_tracing_config(self) -> None:
        """Test that evaluate() preserves custom tracing configuration."""
        # Configure custom tracer
        custom_tracer = configure_tracing(service_name="custom-service", enabled=False)

        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=[{"input": "test", "expected": "Response: test"}],
            metrics=["exact_match"],
        )

        # Should still be the same custom tracer
        assert get_tracer() is custom_tracer
        assert result.total == 1

    def test_evaluate_with_latency_metric_includes_timing(self) -> None:
        """Test that latency metric works correctly with tracing."""
        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=[
                {"input": "test1", "expected": "Response: test1"},
                {"input": "test2", "expected": "Response: test2"},
            ],
            metrics=["latency"],
            enable_tracing=True,
        )

        assert result.total == 2
        assert result.avg_latency_ms >= 0

        # Check each test result has latency
        for test_result in result.results:
            assert test_result.latency_ms >= 0
            assert len(test_result.metrics) == 1
            assert test_result.metrics[0].name == "latency"
            assert test_result.metrics[0].passed is True

    def test_evaluate_with_goldens_generates_test_cases(self) -> None:
        """Test that Golden data is converted to TestCases with tracing."""
        data = [
            {"input": "test1", "expected": "Response: test1"},  # Golden
            {"input": "test2", "expected": "Response: test2"},  # Golden
        ]

        result = evaluate(
            name="test-eval",
            task=simple_agent,
            data=data,
            metrics=["exact_match"],
            enable_tracing=True,
        )

        assert result.total == 2
        assert result.passed == 2

    def test_evaluate_with_existing_test_cases(self) -> None:
        """Test evaluation with pre-generated TestCases."""
        test_cases = [
            TestCase(input="test1", expected="Response: test1", actual_output="Response: test1"),
            TestCase(input="test2", expected="Response: test2", actual_output="Response: test2"),
        ]

        result = evaluate(
            name="test-eval",
            task=simple_agent,  # Won't be called since actual_output exists
            data=test_cases,
            metrics=["exact_match"],
            enable_tracing=True,
        )

        assert result.total == 2
        assert result.passed == 2

    def test_evaluate_handles_agent_errors_with_tracing(self) -> None:
        """Test that agent errors are properly traced."""

        def failing_agent(input_text: str) -> str:
            """Agent that always fails."""
            raise ValueError("Simulated error")

        result = evaluate(
            name="error-test",
            task=failing_agent,
            data=[{"input": "test", "expected": "anything"}],
            metrics=["exact_match"],
            enable_tracing=True,
        )

        assert result.total == 1
        assert result.failed == 1
        assert result.results[0].error is not None

    def test_multiple_evaluations_with_tracing(self) -> None:
        """Test multiple sequential evaluations with tracing."""
        configure_tracing(enabled=False)
        configure_debug_logging(enabled=True)

        # Run first evaluation
        result1 = evaluate(
            name="eval-1",
            task=simple_agent,
            data=[{"input": "test1", "expected": "Response: test1"}],
            metrics=["exact_match"],
        )

        # Run second evaluation
        result2 = evaluate(
            name="eval-2",
            task=simple_agent,
            data=[{"input": "test2", "expected": "Response: test2"}],
            metrics=["exact_match"],
        )

        assert result1.total == 1
        assert result2.total == 1
        assert result1.passed == 1
        assert result2.passed == 1
