"""Unit tests for core type definitions."""

import asyncio

import pytest
from pydantic import ValidationError

from evaris.types import BaseMetric, EvalResult, Golden, MetricResult, TestCase, TestResult


class TestTestCase:
    """Tests for TestCase model (updated to require actual_output)."""

    def test_create_with_required_fields(self) -> None:
        """Test creating TestCase with required fields (input + actual_output)."""
        tc = TestCase(input="test input", actual_output="test output")
        assert tc.input == "test input"
        assert tc.actual_output == "test output"
        assert tc.expected is None
        assert tc.metadata == {}

    def test_create_with_all_fields(self) -> None:
        """Test creating TestCase with all fields."""
        tc = TestCase(
            input="What is 2+2?",
            actual_output="4",
            expected="4",
            metadata={"difficulty": "easy", "category": "math"},
        )
        assert tc.input == "What is 2+2?"
        assert tc.actual_output == "4"
        assert tc.expected == "4"
        assert tc.metadata is not None
        assert tc.metadata["difficulty"] == "easy"

    def test_input_can_be_any_type(self) -> None:
        """Test that input field accepts any type."""
        # String input
        tc1 = TestCase(input="string", actual_output="output1")
        assert tc1.input == "string"

        # Dict input
        tc2 = TestCase(input={"query": "test", "context": "data"}, actual_output="output2")
        assert tc2.input["query"] == "test"

        # List input
        tc3 = TestCase(input=[1, 2, 3], actual_output="output3")
        assert tc3.input == [1, 2, 3]


class TestMetricResult:
    """Tests for MetricResult model."""

    def test_create_valid_metric_result(self) -> None:
        """Test creating MetricResult with valid score."""
        mr = MetricResult(name="accuracy", score=0.95, passed=True)
        assert mr.name == "accuracy"
        assert mr.score == 0.95
        assert mr.passed is True
        assert mr.metadata == {}

    def test_score_must_be_between_0_and_1(self) -> None:
        """Test that score is validated to be in [0, 1] range."""
        # Valid scores
        MetricResult(name="test", score=0.0, passed=False)
        MetricResult(name="test", score=1.0, passed=True)
        MetricResult(name="test", score=0.5, passed=True)

        # Invalid scores should raise ValidationError
        with pytest.raises(ValidationError):
            MetricResult(name="test", score=-0.1, passed=False)

        with pytest.raises(ValidationError):
            MetricResult(name="test", score=1.1, passed=True)

    def test_metadata_optional(self) -> None:
        """Test that metadata is optional and defaults to empty dict."""
        mr = MetricResult(name="test", score=0.5, passed=True)
        assert mr.metadata == {}

        mr_with_meta = MetricResult(
            name="test", score=0.5, passed=True, metadata={"reasoning": "Good match"}
        )
        assert mr_with_meta.metadata is not None
        assert mr_with_meta.metadata["reasoning"] == "Good match"


class TestTestResult:
    """Tests for TestResult model."""

    def test_create_test_result(self) -> None:
        """Test creating TestResult with all components."""
        tc = TestCase(input="test", actual_output="output", expected="output")
        mr = MetricResult(name="exact_match", score=1.0, passed=True)

        tr = TestResult(test_case=tc, output="output", metrics=[mr], latency_ms=123.45)

        assert tr.test_case.input == "test"
        assert tr.output == "output"
        assert len(tr.metrics) == 1
        assert tr.metrics[0].name == "exact_match"
        assert tr.latency_ms == 123.45
        assert tr.error is None

    def test_latency_must_be_non_negative(self) -> None:
        """Test that latency cannot be negative."""
        tc = TestCase(input="test", actual_output="out")
        mr = MetricResult(name="test", score=0.5, passed=True)

        # Valid latency
        TestResult(test_case=tc, output="out", metrics=[mr], latency_ms=0.0)
        TestResult(test_case=tc, output="out", metrics=[mr], latency_ms=100.5)

        # Invalid latency
        with pytest.raises(ValidationError):
            TestResult(test_case=tc, output="out", metrics=[mr], latency_ms=-1.0)

    def test_error_field_optional(self) -> None:
        """Test that error field is optional."""
        tc = TestCase(input="test", actual_output="out")
        mr = MetricResult(name="test", score=0.0, passed=False)

        # Without error
        tr1 = TestResult(test_case=tc, output="out", metrics=[mr], latency_ms=10.0)
        assert tr1.error is None

        # With error
        tr2 = TestResult(test_case=tc, output="", metrics=[], latency_ms=5.0, error="Agent failed")
        assert tr2.error == "Agent failed"


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_create_eval_result(self) -> None:
        """Test creating EvalResult with aggregated data."""
        tc = TestCase(input="test", actual_output="output", expected="output")
        mr = MetricResult(name="exact_match", score=1.0, passed=True)
        tr = TestResult(test_case=tc, output="output", metrics=[mr], latency_ms=50.0)

        er = EvalResult(
            name="test-eval",
            total=10,
            passed=8,
            failed=2,
            accuracy=0.8,
            avg_latency_ms=45.5,
            results=[tr],
        )

        assert er.name == "test-eval"
        assert er.total == 10
        assert er.passed == 8
        assert er.failed == 2
        assert er.accuracy == 0.8
        assert er.avg_latency_ms == 45.5
        assert len(er.results) == 1

    def test_accuracy_must_be_between_0_and_1(self) -> None:
        """Test accuracy validation."""
        # Valid accuracies
        EvalResult(
            name="test",
            total=1,
            passed=0,
            failed=1,
            accuracy=0.0,
            avg_latency_ms=10.0,
            results=[],
        )
        EvalResult(
            name="test",
            total=1,
            passed=1,
            failed=0,
            accuracy=1.0,
            avg_latency_ms=10.0,
            results=[],
        )

        # Invalid accuracies
        with pytest.raises(ValidationError):
            EvalResult(
                name="test",
                total=1,
                passed=1,
                failed=0,
                accuracy=1.5,
                avg_latency_ms=10.0,
                results=[],
            )

    def test_str_representation(self) -> None:
        """Test human-readable string representation."""
        er = EvalResult(
            name="my-eval",
            total=100,
            passed=95,
            failed=5,
            accuracy=0.95,
            avg_latency_ms=123.45,
            results=[],
        )

        str_repr = str(er)
        assert "my-eval" in str_repr
        assert "100" in str_repr
        assert "95" in str_repr
        assert "5" in str_repr
        assert "95.00%" in str_repr
        assert "123.45ms" in str_repr


class TestGolden:
    """Tests for Golden model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating Golden with only input field."""
        golden = Golden(input="What is 2+2?")
        assert golden.input == "What is 2+2?"
        assert golden.expected is None
        assert golden.metadata == {}

    def test_create_with_all_fields(self) -> None:
        """Test creating Golden with all fields."""
        golden = Golden(
            input="What is the capital of France?",
            expected="Paris",
            metadata={"category": "geography", "difficulty": "easy"},
        )
        assert golden.input == "What is the capital of France?"
        assert golden.expected == "Paris"
        assert golden.metadata["category"] == "geography"
        assert golden.metadata["difficulty"] == "easy"

    def test_input_can_be_any_type(self) -> None:
        """Test that input field accepts any type."""
        # String input
        g1 = Golden(input="string input")
        assert g1.input == "string input"

        # Dict input
        g2 = Golden(input={"query": "test", "context": ["doc1", "doc2"]})
        assert g2.input["query"] == "test"

        # List input
        g3 = Golden(input=["item1", "item2"])
        assert g3.input == ["item1", "item2"]

    def test_expected_can_be_any_type(self) -> None:
        """Test that expected field accepts any type."""
        # String expected
        g1 = Golden(input="q1", expected="answer1")
        assert g1.expected == "answer1"

        # Dict expected
        g2 = Golden(input="q2", expected={"answer": "42", "confidence": 0.95})
        assert isinstance(g2.expected, dict)
        assert g2.expected["answer"] == "42"

        # List expected
        g3 = Golden(input="q3", expected=["option1", "option2"])
        assert g3.expected == ["option1", "option2"]

    def test_no_actual_output_field(self) -> None:
        """Test that Golden does not have actual_output field."""
        golden = Golden(input="test", expected="output")
        assert not hasattr(golden, "actual_output")


class TestTestCaseFromGolden:
    """Tests for TestCase.from_golden() method."""

    def test_create_from_golden_basic(self) -> None:
        """Test creating TestCase from Golden with actual output."""
        golden = Golden(input="What is 2+2?", expected="4")
        test_case = TestCase.from_golden(golden, actual_output="4")

        assert test_case.input == "What is 2+2?"
        assert test_case.expected == "4"
        assert test_case.actual_output == "4"
        assert test_case.metadata == {}

    def test_create_from_golden_with_metadata(self) -> None:
        """Test that metadata is preserved when creating from Golden."""
        golden = Golden(
            input="Capital of France?",
            expected="Paris",
            metadata={"category": "geography", "source": "test_suite_1"},
        )
        test_case = TestCase.from_golden(golden, actual_output="Paris, France")

        assert test_case.input == "Capital of France?"
        assert test_case.expected == "Paris"
        assert test_case.actual_output == "Paris, France"
        assert test_case.metadata["category"] == "geography"
        assert test_case.metadata["source"] == "test_suite_1"

    def test_create_from_golden_without_expected(self) -> None:
        """Test creating TestCase from Golden that has no expected output."""
        golden = Golden(input="Open-ended question")
        test_case = TestCase.from_golden(golden, actual_output="Generated answer")

        assert test_case.input == "Open-ended question"
        assert test_case.expected is None
        assert test_case.actual_output == "Generated answer"

    def test_actual_output_can_be_any_type(self) -> None:
        """Test that actual_output can be any type."""
        golden = Golden(input="test")

        # String output
        tc1 = TestCase.from_golden(golden, actual_output="string")
        assert tc1.actual_output == "string"

        # Dict output
        tc2 = TestCase.from_golden(golden, actual_output={"result": "success"})
        assert tc2.actual_output["result"] == "success"

        # List output
        tc3 = TestCase.from_golden(golden, actual_output=[1, 2, 3])
        assert tc3.actual_output == [1, 2, 3]


class TestTestCaseWithActualOutput:
    """Tests for updated TestCase with actual_output field."""

    def test_create_with_actual_output(self) -> None:
        """Test creating TestCase with actual_output field."""
        tc = TestCase(input="What is 2+2?", expected="4", actual_output="4")
        assert tc.input == "What is 2+2?"
        assert tc.expected == "4"
        assert tc.actual_output == "4"

    def test_actual_output_required_for_direct_creation(self) -> None:
        """Test that actual_output is required when creating TestCase directly."""
        # This should work - actual_output provided
        tc1 = TestCase(input="test", actual_output="output")
        assert tc1.actual_output == "output"

        # This should raise ValidationError - missing actual_output
        with pytest.raises(ValidationError) as exc_info:
            TestCase(input="test")

        # Check that the error mentions actual_output
        assert "actual_output" in str(exc_info.value).lower()

    def test_all_three_values_present(self) -> None:
        """Test TestCase with input, expected, and actual_output."""
        tc = TestCase(
            input="Capital of France?",
            expected="Paris",
            actual_output="Paris is the capital",
        )
        assert tc.input == "Capital of France?"
        assert tc.expected == "Paris"
        assert tc.actual_output == "Paris is the capital"


class TestBaseMetric:
    """Tests for BaseMetric abstract class."""

    def test_cannot_instantiate_base_metric(self) -> None:
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric()  # type: ignore

    def test_must_implement_a_measure(self) -> None:
        """Test that concrete metrics must implement a_measure method."""

        # This should fail - missing a_measure
        with pytest.raises(TypeError):

            class IncompleteMetric(BaseMetric):
                pass

            IncompleteMetric()  # type: ignore

    def test_concrete_metric_with_a_measure(self) -> None:
        """Test creating a concrete metric that implements a_measure."""

        class ConcreteMetric(BaseMetric):
            async def a_measure(self, test_case: TestCase) -> MetricResult:
                return MetricResult(name="test_metric", score=0.8, passed=True, metadata={})

        metric = ConcreteMetric()
        assert metric is not None

        # Test that a_measure works
        test_case = TestCase(input="test", actual_output="output")
        result = asyncio.run(metric.a_measure(test_case))

        assert isinstance(result, MetricResult)
        assert result.name == "test_metric"
        assert result.score == 0.8
        assert result.passed is True

    def test_concrete_metric_measure_sync_wrapper(self) -> None:
        """Test that measure() method provides sync wrapper for a_measure()."""

        class SyncWrapperMetric(BaseMetric):
            async def a_measure(self, test_case: TestCase) -> MetricResult:
                return MetricResult(
                    name="async_metric", score=0.9, passed=True, metadata={"method": "async"}
                )

            def measure(self, test_case: TestCase) -> MetricResult:
                """Sync wrapper."""
                return asyncio.run(self.a_measure(test_case))

        metric = SyncWrapperMetric()
        test_case = TestCase(input="test", actual_output="output")

        # Test sync method
        result = metric.measure(test_case)
        assert isinstance(result, MetricResult)
        assert result.name == "async_metric"
        assert result.score == 0.9
