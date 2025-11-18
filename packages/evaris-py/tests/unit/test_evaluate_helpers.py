"""Unit tests for evaluate() helper functions."""

import pytest

from evaris.evaluate import _normalize_data, _resolve_metrics
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.types import Golden, TestCase


class TestNormalizeData:
    """Tests for _normalize_data() function."""

    def test_normalize_dict_with_actual_output_to_testcase(self) -> None:
        """Test normalizing dict with actual_output creates TestCase."""
        data = {
            "input": "test input",
            "expected": "test output",
            "actual_output": "actual output",
            "metadata": {"key": "value"},
        }

        result = _normalize_data(data)

        assert isinstance(result, TestCase)
        assert result.input == "test input"
        assert result.expected == "test output"
        assert result.actual_output == "actual output"
        assert result.metadata == {"key": "value"}

    def test_normalize_dict_without_actual_output_to_golden(self) -> None:
        """Test normalizing dict without actual_output creates Golden."""
        data = {
            "input": "test input",
            "expected": "test output",
            "metadata": {"key": "value"},
        }

        result = _normalize_data(data)

        assert isinstance(result, Golden)
        assert result.input == "test input"
        assert result.expected == "test output"
        assert result.metadata == {"key": "value"}

    def test_normalize_golden_object_passthrough(self) -> None:
        """Test that Golden objects pass through unchanged."""
        golden = Golden(input="test", expected="output", metadata={"x": 1})

        result = _normalize_data(golden)

        assert result is golden
        assert result.input == "test"
        assert result.expected == "output"
        assert result.metadata == {"x": 1}

    def test_normalize_testcase_object_passthrough(self) -> None:
        """Test that TestCase objects pass through unchanged."""
        test_case = TestCase(
            input="test", expected="output", actual_output="actual", metadata={"x": 1}
        )

        result = _normalize_data(test_case)

        assert result is test_case
        assert result.input == "test"
        assert result.expected == "output"
        assert result.actual_output == "actual"
        assert result.metadata == {"x": 1}

    def test_normalize_handles_missing_expected(self) -> None:
        """Test normalization when expected field is missing."""
        data = {"input": "test"}

        result = _normalize_data(data)

        assert isinstance(result, Golden)
        assert result.expected is None

    def test_normalize_handles_missing_metadata(self) -> None:
        """Test normalization when metadata field is missing."""
        data = {"input": "test", "expected": "output"}

        result = _normalize_data(data)

        assert isinstance(result, Golden)
        assert result.metadata == {}


class TestResolveMetrics:
    """Tests for _resolve_metrics() function."""

    def test_resolve_single_metric(self) -> None:
        """Test resolving a single metric by name."""
        metrics = _resolve_metrics(["exact_match"])

        assert len(metrics) == 1
        assert isinstance(metrics[0], ExactMatchMetric)

    def test_resolve_multiple_metrics(self) -> None:
        """Test resolving multiple metrics."""
        metrics = _resolve_metrics(["exact_match", "latency"])

        assert len(metrics) == 2
        assert any(isinstance(m, ExactMatchMetric) for m in metrics)

    def test_resolve_unknown_metric_raises_error(self) -> None:
        """Test that unknown metric name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric 'unknown_metric'"):
            _resolve_metrics(["unknown_metric"])

    def test_resolve_partial_unknown_metrics(self) -> None:
        """Test error when some metrics are unknown."""
        with pytest.raises(ValueError, match="Unknown metric"):
            _resolve_metrics(["exact_match", "invalid_metric"])

    def test_resolve_empty_list(self) -> None:
        """Test resolving empty metrics list."""
        metrics = _resolve_metrics([])

        assert len(metrics) == 0

    def test_error_message_shows_available_metrics(self) -> None:
        """Test that error message lists available metrics."""
        with pytest.raises(ValueError, match="Available metrics"):
            _resolve_metrics(["bad_metric"])
