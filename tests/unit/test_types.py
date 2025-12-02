"""Unit tests for core type definitions."""

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from evaris.types import (
    BaseMetric,
    EvalResult,
    Golden,
    MetricResult,
    ReasoningStep,
    TestCase,
    TestResult,
)


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


class TestReasoningStep:
    """Tests for ReasoningStep model (new for reasoning framework)."""

    def test_create_valid_reasoning_step(self) -> None:
        """Test creating ReasoningStep with all required fields."""
        step = ReasoningStep(
            step_number=1,
            operation="text_normalization",
            description="Normalizing input text for comparison",
            inputs={"expected": "Paris", "actual": "paris"},
            outputs={"expected_norm": "paris", "actual_norm": "paris"},
        )
        assert step.step_number == 1
        assert step.operation == "text_normalization"
        assert step.description == "Normalizing input text for comparison"
        assert step.inputs["expected"] == "Paris"
        assert step.outputs["expected_norm"] == "paris"
        assert step.metadata == {}

    def test_reasoning_step_with_metadata(self) -> None:
        """Test ReasoningStep with optional metadata."""
        step = ReasoningStep(
            step_number=2,
            operation="similarity_calculation",
            description="Computing cosine similarity",
            inputs={"embedding1": [0.1, 0.2], "embedding2": [0.3, 0.4]},
            outputs={"similarity": 0.92},
            metadata={"model": "all-MiniLM-L6-v2", "dimension": 384},
        )
        assert step.metadata["model"] == "all-MiniLM-L6-v2"
        assert step.metadata["dimension"] == 384

    def test_reasoning_step_inputs_can_be_any_type(self) -> None:
        """Test that inputs field can contain various types."""
        step = ReasoningStep(
            step_number=1,
            operation="test",
            description="Test with various input types",
            inputs={
                "string": "test",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "bool": True,
            },
            outputs={},
        )
        assert step.inputs["string"] == "test"
        assert step.inputs["number"] == 42
        assert step.inputs["list"] == [1, 2, 3]
        assert step.inputs["dict"]["nested"] == "value"
        assert step.inputs["bool"] is True

    def test_reasoning_step_outputs_can_be_any_type(self) -> None:
        """Test that outputs field can contain various types."""
        step = ReasoningStep(
            step_number=1,
            operation="test",
            description="Test with various output types",
            inputs={},
            outputs={
                "result": "success",
                "score": 0.95,
                "details": {"passed": True, "errors": []},
            },
        )
        assert step.outputs["result"] == "success"
        assert step.outputs["score"] == 0.95
        assert step.outputs["details"]["passed"] is True

    def test_reasoning_step_metadata_optional(self) -> None:
        """Test that metadata defaults to empty dict."""
        step = ReasoningStep(
            step_number=1,
            operation="test",
            description="Test step",
            inputs={},
            outputs={},
        )
        assert step.metadata == {}
        assert isinstance(step.metadata, dict)


class TestMetricResultWithReasoning:
    """Tests for MetricResult with new reasoning fields."""

    def test_metric_result_with_reasoning_summary(self) -> None:
        """Test MetricResult with reasoning summary string."""
        mr = MetricResult(
            name="semantic_similarity",
            score=0.92,
            passed=True,
            reasoning="High semantic similarity (92%) exceeds threshold (80%)",
        )
        assert mr.reasoning == "High semantic similarity (92%) exceeds threshold (80%)"
        assert mr.reasoning_steps is None
        assert mr.reasoning_type is None

    def test_metric_result_with_reasoning_steps(self) -> None:
        """Test MetricResult with structured reasoning steps."""
        steps = [
            ReasoningStep(
                step_number=1,
                operation="normalize",
                description="Normalize text",
                inputs={"text": "Paris"},
                outputs={"normalized": "paris"},
            ),
            ReasoningStep(
                step_number=2,
                operation="compare",
                description="Compare normalized texts",
                inputs={"expected": "paris", "actual": "paris"},
                outputs={"match": True, "score": 1.0},
            ),
        ]
        mr = MetricResult(
            name="exact_match",
            score=1.0,
            passed=True,
            reasoning_steps=steps,
        )
        assert mr.reasoning_steps is not None
        assert len(mr.reasoning_steps) == 2
        assert mr.reasoning_steps[0].operation == "normalize"
        assert mr.reasoning_steps[1].operation == "compare"
        assert mr.reasoning_steps[1].outputs["match"] is True

    def test_metric_result_with_reasoning_type(self) -> None:
        """Test MetricResult with reasoning type indicator."""
        mr_logic = MetricResult(
            name="exact_match",
            score=1.0,
            passed=True,
            reasoning="Exact match after normalization",
            reasoning_type="logic",
        )
        assert mr_logic.reasoning_type == "logic"

        mr_llm = MetricResult(
            name="llm_judge",
            score=0.85,
            passed=True,
            reasoning="The output correctly answers the question...",
            reasoning_type="llm",
        )
        assert mr_llm.reasoning_type == "llm"

        mr_hybrid = MetricResult(
            name="semantic_similarity",
            score=0.9,
            passed=True,
            reasoning="Similarity score calculated, then explained by LLM",
            reasoning_type="hybrid",
        )
        assert mr_hybrid.reasoning_type == "hybrid"

    def test_metric_result_with_all_reasoning_fields(self) -> None:
        """Test MetricResult with all reasoning fields populated."""
        steps = [
            ReasoningStep(
                step_number=1,
                operation="embedding",
                description="Generate embeddings",
                inputs={"text": "Paris"},
                outputs={"embedding": [0.1, 0.2, 0.3]},
            )
        ]
        mr = MetricResult(
            name="semantic_similarity",
            score=0.92,
            passed=True,
            metadata={"threshold": 0.8, "model": "all-MiniLM-L6-v2"},
            reasoning="High similarity score exceeds threshold",
            reasoning_steps=steps,
            reasoning_type="logic",
        )
        assert mr.reasoning is not None
        assert mr.reasoning_steps is not None
        assert mr.reasoning_type == "logic"
        assert len(mr.reasoning_steps) == 1
        assert mr.metadata["threshold"] == 0.8

    def test_metric_result_reasoning_fields_optional(self) -> None:
        """Test that reasoning fields are optional and default to None."""
        mr = MetricResult(
            name="test",
            score=0.5,
            passed=True,
        )
        assert mr.reasoning is None
        assert mr.reasoning_steps is None
        assert mr.reasoning_type is None

    def test_metric_result_backward_compatible(self) -> None:
        """Test that existing code without reasoning fields still works."""
        mr = MetricResult(
            name="exact_match",
            score=1.0,
            passed=True,
            metadata={"case_sensitive": False},
        )
        assert mr.name == "exact_match"
        assert mr.score == 1.0
        assert mr.passed is True
        assert mr.metadata["case_sensitive"] is False
        assert mr.reasoning is None

    def test_metric_result_reasoning_type_validation(self) -> None:
        """Test that reasoning_type only accepts valid literals."""
        valid_types: list[str] = ["logic", "llm", "hybrid"]
        for rtype in valid_types:
            mr = MetricResult(
                name="test",
                score=0.8,
                passed=True,
                reasoning_type=rtype,  # type: ignore[arg-type]
            )
            assert mr.reasoning_type == rtype

        with pytest.raises(ValidationError):
            MetricResult(
                name="test",
                score=0.8,
                passed=True,
                reasoning_type="invalid_type",  # type: ignore[arg-type]
            )


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
        assert tc2.actual_output is not None
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
        """Test that actual_output is optional when creating TestCase directly."""
        # This should work - actual_output provided
        tc1 = TestCase(input="test", actual_output="output")
        assert tc1.actual_output == "output"

        # This should also work - actual_output omitted (it's optional now)
        tc2 = TestCase(input="test")
        assert tc2.actual_output is None

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
            async def a_measure(self, test_case: TestCase, actual_output: Any) -> MetricResult:
                return MetricResult(name="test_metric", score=0.8, passed=True, metadata={})

        metric = ConcreteMetric()
        assert metric is not None

        # Test that a_measure works
        test_case = TestCase(input="test", actual_output="output")
        result = asyncio.run(metric.a_measure(test_case, "output"))

        assert isinstance(result, MetricResult)
        assert result.name == "test_metric"
        assert result.score == 0.8
        assert result.passed is True

    def test_concrete_metric_measure_sync_wrapper(self) -> None:
        """Test that measure() method provides sync wrapper for a_measure()."""

        class SyncWrapperMetric(BaseMetric):
            async def a_measure(self, test_case: TestCase, actual_output: Any) -> MetricResult:
                return MetricResult(
                    name="async_metric", score=0.9, passed=True, metadata={"method": "async"}
                )

            def measure(self, test_case: TestCase, actual_output: Any) -> MetricResult:
                """Sync wrapper."""
                return asyncio.run(self.a_measure(test_case, actual_output))

        metric = SyncWrapperMetric()
        test_case = TestCase(input="test", actual_output="output")

        # Test sync method
        result = metric.measure(test_case, "output")
        assert isinstance(result, MetricResult)
        assert result.name == "async_metric"
        assert result.score == 0.9
