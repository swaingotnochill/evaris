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


class TestExactMatchMetricReasoning:
    """Tests for ExactMatchMetric with reasoning capabilities."""

    def test_exact_match_generates_reasoning_steps(self) -> None:
        """Test that reasoning steps are generated for exact match."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="output", actual_output="output")
        result = metric.score(tc, "output")

        # Should have reasoning steps
        assert result.reasoning_steps is not None
        assert len(result.reasoning_steps) > 0

        # Should have reasoning summary
        assert result.reasoning is not None
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

        # Should indicate logic-based reasoning
        assert result.reasoning_type == "logic"

    def test_reasoning_steps_show_normalization(self) -> None:
        """Test that reasoning steps show normalization process."""
        metric = ExactMatchMetric(case_sensitive=False, strip_whitespace=True)
        tc = TestCase(input="test", expected="Hello World", actual_output="  hello world  ")
        result = metric.score(tc, "  hello world  ")

        # Find normalization step
        assert result.reasoning_steps is not None
        norm_steps = [s for s in result.reasoning_steps if "normal" in s.operation.lower()]
        assert len(norm_steps) > 0

        # Check that normalization step shows the transformation
        norm_step = norm_steps[0]
        assert norm_step.inputs is not None
        assert norm_step.outputs is not None

    def test_reasoning_shows_final_comparison(self) -> None:
        """Test that reasoning includes final comparison step."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="Paris", actual_output="Paris")
        result = metric.score(tc, "Paris")

        # Should have a comparison step
        assert result.reasoning_steps is not None
        comp_steps = [s for s in result.reasoning_steps if "compar" in s.operation.lower()]
        assert len(comp_steps) > 0

        # Comparison step should show match result
        comp_step = comp_steps[0]
        assert "match" in comp_step.outputs or "score" in comp_step.outputs

    def test_reasoning_for_mismatch(self) -> None:
        """Test reasoning when values don't match."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="expected", actual_output="actual")
        result = metric.score(tc, "actual")

        # Should have reasoning even for mismatch
        assert result.reasoning is not None
        assert result.reasoning_steps is not None

        # Reasoning should indicate mismatch
        assert "match" in result.reasoning.lower() or "mismatch" in result.reasoning.lower()

        # Should show it failed
        assert result.passed is False

    def test_reasoning_with_case_normalization(self) -> None:
        """Test reasoning shows case normalization details."""
        metric = ExactMatchMetric(case_sensitive=False)
        tc = TestCase(input="test", expected="HELLO", actual_output="hello")
        result = metric.score(tc, "hello")

        # Should show normalization in reasoning steps
        assert result.reasoning_steps is not None

        # Check that normalization step exists and shows case conversion
        norm_step = None
        for step in result.reasoning_steps:
            if "normal" in step.operation.lower():
                norm_step = step
                break

        assert norm_step is not None
        # Should show original and normalized values
        assert norm_step.inputs is not None
        assert norm_step.outputs is not None

    def test_reasoning_with_whitespace_stripping(self) -> None:
        """Test reasoning shows whitespace stripping."""
        metric = ExactMatchMetric(strip_whitespace=True)
        tc = TestCase(input="test", expected="hello", actual_output="  hello  ")
        result = metric.score(tc, "  hello  ")

        # Should show whitespace handling in reasoning
        assert result.reasoning_steps is not None

        # Find step that handles whitespace
        ws_step = None
        for step in result.reasoning_steps:
            if "normal" in step.operation.lower() or "strip" in step.operation.lower():
                ws_step = step
                break

        assert ws_step is not None

    def test_reasoning_summary_is_human_readable(self) -> None:
        """Test that reasoning summary is clear and human-readable."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="42", actual_output="42")
        result = metric.score(tc, "42")

        # Should have clear reasoning
        assert result.reasoning is not None
        assert len(result.reasoning) > 10  # Not just "match" or "yes"

        # Should be a proper sentence (capital letter)
        assert result.reasoning[0].isupper() or result.reasoning[0].isdigit()

    def test_reasoning_steps_are_sequential(self) -> None:
        """Test that reasoning steps are numbered sequentially."""
        metric = ExactMatchMetric(case_sensitive=False, strip_whitespace=True)
        tc = TestCase(input="test", expected="TEST", actual_output="  test  ")
        result = metric.score(tc, "  test  ")

        assert result.reasoning_steps is not None
        assert len(result.reasoning_steps) > 0

        # Check sequential numbering
        for i, step in enumerate(result.reasoning_steps, start=1):
            assert step.step_number == i

    def test_reasoning_steps_have_descriptions(self) -> None:
        """Test that all reasoning steps have clear descriptions."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="value", actual_output="value")
        result = metric.score(tc, "value")

        assert result.reasoning_steps is not None

        # All steps should have meaningful descriptions
        for step in result.reasoning_steps:
            assert step.description is not None
            assert len(step.description) > 5  # Not empty or too short
            assert isinstance(step.description, str)

    def test_reasoning_includes_configuration(self) -> None:
        """Test that reasoning reflects metric configuration."""
        metric = ExactMatchMetric(case_sensitive=False, strip_whitespace=True)
        tc = TestCase(input="test", expected="Hello", actual_output="hello")
        result = metric.score(tc, "hello")

        # Reasoning should mention or reflect the configuration
        # Either in reasoning summary or in step metadata
        assert result.reasoning is not None
        assert result.reasoning_steps is not None
        reasoning_text = result.reasoning.lower()
        has_config_info = (
            "case" in reasoning_text
            or "whitespace" in reasoning_text
            or any("case_sensitive" in str(step.metadata) for step in result.reasoning_steps)
        )
        assert has_config_info

    def test_reasoning_for_non_string_values(self) -> None:
        """Test reasoning works for non-string values."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected=42, actual_output=42)
        result = metric.score(tc, 42)

        # Should still have reasoning
        assert result.reasoning is not None
        assert result.reasoning_steps is not None

        # Should indicate no normalization needed for non-strings
        assert len(result.reasoning_steps) >= 1
