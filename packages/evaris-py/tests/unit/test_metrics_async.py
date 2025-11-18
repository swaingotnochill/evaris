"""Unit tests for async metric methods (a_measure)."""

import pytest

from evaris.metrics.answer_match import AnswerMatchConfig, AnswerMatchMetric
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.metrics.fuzz_test import FuzzTestConfig, FuzzTestMetric
from evaris.metrics.semantic_similarity import SemanticSimilarityConfig, SemanticSimilarityMetric
from evaris.metrics.state_match import StateMatchConfig, StateMatchMetric
from evaris.metrics.unit_test import UnitTestMetric
from evaris.types import MetricResult, TestCase


class TestExactMatchAsync:
    """Tests for ExactMatchMetric.a_measure()."""

    @pytest.mark.asyncio
    async def test_a_measure_basic(self) -> None:
        """Test async exact match with basic case."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="output", actual_output="output")

        result = await metric.a_measure(tc)

        assert isinstance(result, MetricResult)
        assert result.name == "exact_match"
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_mismatch(self) -> None:
        """Test async exact match with mismatch."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="expected", actual_output="actual")

        result = await metric.a_measure(tc)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_a_measure_case_insensitive(self) -> None:
        """Test async exact match with case-insensitive mode."""
        metric = ExactMatchMetric(case_sensitive=False)
        tc = TestCase(input="test", expected="Hello", actual_output="hello")

        result = await metric.a_measure(tc)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_strip_whitespace(self) -> None:
        """Test async exact match with whitespace stripping."""
        metric = ExactMatchMetric(strip_whitespace=True)
        tc = TestCase(input="test", expected="hello", actual_output="  hello  ")

        result = await metric.a_measure(tc)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_none_actual_output(self) -> None:
        """Test that None actual_output raises error."""
        metric = ExactMatchMetric()
        tc = TestCase(input="test", expected="output", actual_output=None)

        with pytest.raises(ValueError, match="actual_output"):
            await metric.a_measure(tc)


class TestAnswerMatchAsync:
    """Tests for AnswerMatchMetric.a_measure()."""

    @pytest.mark.asyncio
    async def test_a_measure_delimited_basic(self) -> None:
        """Test async answer match with delimited format."""
        metric = AnswerMatchMetric()
        tc = TestCase(
            input="What is 2+2?",
            expected="4",
            actual_output="Let me think... Answer: 4",
        )

        result = await metric.a_measure(tc)

        assert result.name == "answer_match"
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_delimited_mismatch(self) -> None:
        """Test async answer match with wrong answer."""
        metric = AnswerMatchMetric()
        tc = TestCase(
            input="What is 2+2?",
            expected="4",
            actual_output="Answer: 5",
        )

        result = await metric.a_measure(tc)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_a_measure_missing_delimiter(self) -> None:
        """Test async answer match when delimiter is missing."""
        metric = AnswerMatchMetric()
        tc = TestCase(
            input="What is 2+2?",
            expected="4",
            actual_output="The result is 4",  # No "Answer:" delimiter
        )

        result = await metric.a_measure(tc)

        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_custom_delimiter(self) -> None:
        """Test async answer match with custom delimiter."""
        config = AnswerMatchConfig(delimiter="Result:")
        metric = AnswerMatchMetric(config)
        tc = TestCase(
            input="Calculate 5*3",
            expected="15",
            actual_output="Result: 15",
        )

        result = await metric.a_measure(tc)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_none_actual_output(self) -> None:
        """Test that None actual_output raises error."""
        metric = AnswerMatchMetric()
        tc = TestCase(input="test", expected="answer", actual_output=None)

        with pytest.raises(ValueError, match="actual_output"):
            await metric.a_measure(tc)


class TestStateMatchAsync:
    """Tests for StateMatchMetric.a_measure()."""

    @pytest.mark.asyncio
    async def test_a_measure_exact_match(self) -> None:
        """Test async state match with exact match."""
        metric = StateMatchMetric()
        tc = TestCase(
            input="test",
            expected={"state": {"a": 1, "b": 2}},
            actual_output={"a": 1, "b": 2},
        )

        result = await metric.a_measure(tc)

        assert result.name == "state_match"
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_mismatch(self) -> None:
        """Test async state match with mismatch."""
        metric = StateMatchMetric()
        tc = TestCase(
            input="test",
            expected={"state": {"a": 1, "b": 2}},
            actual_output={"a": 1, "b": 3},
        )

        result = await metric.a_measure(tc)

        assert result.passed is False
        assert "differences" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_subset_mode(self) -> None:
        """Test async state match with subset comparison."""
        config = StateMatchConfig(comparison_mode="subset", check_side_effects=False)
        metric = StateMatchMetric(config)
        tc = TestCase(
            input="test",
            expected={"state": {"a": 1}},
            actual_output={"a": 1, "b": 2},  # Extra key is OK in subset mode
        )

        result = await metric.a_measure(tc)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_side_effects_detected(self) -> None:
        """Test async state match detects side effects."""
        config = StateMatchConfig(check_side_effects=True)
        metric = StateMatchMetric(config)
        tc = TestCase(
            input="test",
            expected={"state": {"a": 1}},
            actual_output={"a": 1, "unexpected": "value"},
        )

        result = await metric.a_measure(tc)

        assert result.passed is False
        assert "side effect" in str(result.metadata.get("differences", []))

    @pytest.mark.asyncio
    async def test_a_measure_none_actual_output(self) -> None:
        """Test that None actual_output raises error."""
        metric = StateMatchMetric()
        tc = TestCase(input="test", expected={"state": {"a": 1}}, actual_output=None)

        with pytest.raises(ValueError, match="actual_output"):
            await metric.a_measure(tc)


class TestSemanticSimilarityAsync:
    """Tests for SemanticSimilarityMetric.a_measure()."""

    @pytest.mark.asyncio
    async def test_a_measure_identical_text(self) -> None:
        """Test async semantic similarity with identical text."""
        metric = SemanticSimilarityMetric()
        tc = TestCase(
            input="What is the capital of France?",
            expected="Paris",
            actual_output="Paris",
        )

        result = await metric.a_measure(tc)

        assert result.name == "semantic_similarity"
        assert result.score > 0.9  # Should be very high for identical text
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_a_measure_similar_meaning(self) -> None:
        """Test async semantic similarity with similar meaning."""
        metric = SemanticSimilarityMetric()
        tc = TestCase(
            input="What is the capital of France?",
            expected="Paris",
            actual_output="The capital is Paris",
        )

        result = await metric.a_measure(tc)

        assert result.name == "semantic_similarity"
        assert result.score > 0.7  # Should have high similarity
        assert "similarity" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_different_meaning(self) -> None:
        """Test async semantic similarity with different meaning."""
        metric = SemanticSimilarityMetric()
        tc = TestCase(
            input="What is the capital of France?",
            expected="Paris",
            actual_output="London",
        )

        result = await metric.a_measure(tc)

        assert result.score < 0.8  # Should have lower similarity
        assert "similarity" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_custom_threshold(self) -> None:
        """Test async semantic similarity with custom threshold."""
        config = SemanticSimilarityConfig(threshold=0.95)
        metric = SemanticSimilarityMetric(config)
        tc = TestCase(
            input="test",
            expected="hello",
            actual_output="hello world",  # Similar but not identical
        )

        result = await metric.a_measure(tc)

        # May pass or fail depending on similarity, but should run without error
        assert result.name == "semantic_similarity"
        assert "similarity" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_none_actual_output(self) -> None:
        """Test that None actual_output raises error."""
        metric = SemanticSimilarityMetric()
        tc = TestCase(input="test", expected="output", actual_output=None)

        with pytest.raises(ValueError, match="actual_output"):
            await metric.a_measure(tc)


class TestUnitTestAsync:
    """Tests for UnitTestMetric.a_measure()."""

    @pytest.mark.asyncio
    async def test_a_measure_passing_tests(self) -> None:
        """Test async unit test metric with passing tests."""
        metric = UnitTestMetric()
        tc = TestCase(
            input="Write a function to add two numbers",
            expected={"tests": ["def test_add():\n    assert add(2, 3) == 5"]},
            actual_output="def add(a, b):\n    return a + b",
        )

        result = await metric.a_measure(tc)

        assert result.name == "unit_test"
        assert result.score == 1.0
        assert result.passed is True
        assert "num_tests" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_failing_tests(self) -> None:
        """Test async unit test metric with failing tests."""
        metric = UnitTestMetric()
        tc = TestCase(
            input="Write a function to add two numbers",
            expected={"tests": ["def test_add():\n    assert add(2, 3) == 5"]},
            actual_output="def add(a, b):\n    return a - b",  # Wrong implementation
        )

        result = await metric.a_measure(tc)

        assert result.score == 0.0
        assert result.passed is False
        assert "test_output" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_none_actual_output(self) -> None:
        """Test that None actual_output raises error."""
        metric = UnitTestMetric()
        tc = TestCase(
            input="test",
            expected={"tests": ["def test_foo(): pass"]},
            actual_output=None,
        )

        with pytest.raises(ValueError, match="actual_output"):
            await metric.a_measure(tc)

    @pytest.mark.asyncio
    async def test_a_measure_missing_tests(self) -> None:
        """Test that missing tests raises error."""
        metric = UnitTestMetric()
        tc = TestCase(
            input="test",
            expected={"tests": []},  # Empty tests
            actual_output="def foo(): pass",
        )

        with pytest.raises(ValueError, match="No tests"):
            await metric.a_measure(tc)


class TestFuzzTestAsync:
    """Tests for FuzzTestMetric.a_measure()."""

    @pytest.mark.asyncio
    async def test_a_measure_robust_function(self) -> None:
        """Test async fuzz test with robust function."""
        config = FuzzTestConfig(
            num_fuzz_cases=10,  # Reduce number for faster tests
            edge_cases=False,  # Disable edge cases to avoid inf issues
            input_types=["int"],  # Only use ints to avoid repr() issues with inf
            type_confusion=False,  # Disable type confusion
            boundary_values=False,  # Disable boundary values
        )
        metric = FuzzTestMetric(config)
        tc = TestCase(
            input="Write a safe square function",
            expected={"function_name": "square"},
            actual_output="""def square(x):
    try:
        return x * x
    except:
        return 0
""",
        )

        result = await metric.a_measure(tc)

        assert result.name == "fuzz_test"
        # Should pass most tests due to error handling
        assert result.score >= 0.7  # Most random ints should work
        assert "total_cases" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_fragile_function(self) -> None:
        """Test async fuzz test with fragile function."""
        config = FuzzTestConfig(
            num_fuzz_cases=10,  # Reduce number for faster tests
            edge_cases=False,  # Disable edge cases
            input_types=["int"],  # Only use ints
        )
        metric = FuzzTestMetric(config)
        tc = TestCase(
            input="Write a division function",
            expected={"function_name": "divide"},
            actual_output="def divide(a, b):\n    return a / b",  # No error handling
        )

        result = await metric.a_measure(tc)

        assert result.name == "fuzz_test"
        # Should fail some tests due to division by zero
        assert "total_cases" in result.metadata

    @pytest.mark.asyncio
    async def test_a_measure_none_actual_output(self) -> None:
        """Test that None actual_output raises error."""
        metric = FuzzTestMetric()
        tc = TestCase(
            input="test",
            expected={"function_name": "foo"},
            actual_output=None,
        )

        with pytest.raises(ValueError, match="actual_output"):
            await metric.a_measure(tc)

    @pytest.mark.asyncio
    async def test_a_measure_missing_function_name(self) -> None:
        """Test that missing function_name raises error."""
        metric = FuzzTestMetric()
        tc = TestCase(
            input="test",
            expected={},  # No function_name
            actual_output="def foo(): pass",
        )

        with pytest.raises(ValueError, match="function_name"):
            await metric.a_measure(tc)
