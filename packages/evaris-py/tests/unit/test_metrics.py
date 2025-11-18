"""Unit tests for built-in metrics."""

import pytest

from evaris.metrics.exact_match import ExactMatchMetric
from evaris.types import MetricResult, TestCase


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.metric = ExactMatchMetric()

    def test_exact_match_strings(self) -> None:
        """Test exact string matching."""
        tc = TestCase(input="test", expected="output", actual_output="output")
        result = self.metric.score(tc, "output")

        assert isinstance(result, MetricResult)
        assert result.name == "exact_match"
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_mismatch_strings(self) -> None:
        """Test string mismatch."""
        tc = TestCase(input="test", expected="expected", actual_output="actual")
        result = self.metric.score(tc, "actual")

        assert result.score == 0.0
        assert result.passed is False

    def test_case_sensitive_by_default(self) -> None:
        """Test that matching is case-sensitive by default."""
        tc = TestCase(input="test", expected="Hello", actual_output="hello")
        result = self.metric.score(tc, "hello")

        assert result.score == 0.0
        assert result.passed is False

    def test_case_insensitive_mode(self) -> None:
        """Test case-insensitive matching when enabled."""
        metric = ExactMatchMetric(case_sensitive=False)
        tc = TestCase(input="test", expected="Hello World", actual_output="hello world")
        result = metric.score(tc, "hello world")

        assert result.score == 1.0
        assert result.passed is True

    def test_whitespace_sensitive_by_default(self) -> None:
        """Test that whitespace is significant by default."""
        tc = TestCase(input="test", expected="hello", actual_output=" hello ")
        result = self.metric.score(tc, " hello ")

        assert result.score == 0.0
        assert result.passed is False

    def test_strip_whitespace_mode(self) -> None:
        """Test whitespace stripping when enabled."""
        metric = ExactMatchMetric(strip_whitespace=True)
        tc = TestCase(input="test", expected="hello", actual_output="  hello  ")
        result = metric.score(tc, "  hello  ")

        assert result.score == 1.0
        assert result.passed is True

    def test_combined_case_and_whitespace_normalization(self) -> None:
        """Test combining case-insensitive and whitespace stripping."""
        metric = ExactMatchMetric(case_sensitive=False, strip_whitespace=True)
        tc = TestCase(input="test", expected="Hello World", actual_output="  HELLO WORLD  ")
        result = metric.score(tc, "  HELLO WORLD  ")

        assert result.score == 1.0
        assert result.passed is True

    def test_numeric_values(self) -> None:
        """Test matching numeric values."""
        tc = TestCase(input="2+2", expected=4, actual_output=4)
        result = self.metric.score(tc, 4)

        assert result.score == 1.0
        assert result.passed is True

    def test_numeric_mismatch(self) -> None:
        """Test numeric value mismatch."""
        tc = TestCase(input="2+2", expected=4, actual_output=5)
        result = self.metric.score(tc, 5)

        assert result.score == 0.0
        assert result.passed is False

    def test_none_expected_raises_error(self) -> None:
        """Test that expected=None raises error for exact match."""
        tc = TestCase(input="test", expected=None, actual_output=None)

        with pytest.raises(ValueError, match="expected"):
            self.metric.score(tc, None)

    def test_empty_string_match(self) -> None:
        """Test matching empty strings."""
        tc = TestCase(input="test", expected="", actual_output="")
        result = self.metric.score(tc, "")

        assert result.score == 1.0
        assert result.passed is True

    def test_dict_values(self) -> None:
        """Test matching dictionary values."""
        tc = TestCase(input="test", expected={"key": "value"}, actual_output={"key": "value"})
        result = self.metric.score(tc, {"key": "value"})

        assert result.score == 1.0
        assert result.passed is True

    def test_list_values(self) -> None:
        """Test matching list values."""
        tc = TestCase(input="test", expected=[1, 2, 3], actual_output=[1, 2, 3])
        result = self.metric.score(tc, [1, 2, 3])

        assert result.score == 1.0
        assert result.passed is True

    def test_list_order_matters(self) -> None:
        """Test that list order is significant."""
        tc = TestCase(input="test", expected=[1, 2, 3], actual_output=[3, 2, 1])
        result = self.metric.score(tc, [3, 2, 1])

        assert result.score == 0.0
        assert result.passed is False

    def test_missing_expected_raises_error(self) -> None:
        """Test that missing expected value raises error."""
        tc = TestCase(input="test", actual_output="any output")  # No expected value

        with pytest.raises(ValueError, match="expected"):
            self.metric.score(tc, "any output")

    def test_metadata_contains_comparison_info(self) -> None:
        """Test that result metadata contains comparison information."""
        tc = TestCase(input="test", expected="expected", actual_output="actual")
        result = self.metric.score(tc, "actual")

        assert result.metadata is not None
        assert "expected" in result.metadata
        assert "actual" in result.metadata
        assert result.metadata["expected"] == "expected"
        assert result.metadata["actual"] == "actual"
