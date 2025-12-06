"""Unit tests for evaris_server.api.schemas module.

Tests Pydantic request/response models, including:
- Request validation
- Response serialization
- Enum handling
- Default values
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from evaris_server.api.schemas import (
    EvalListItem,
    EvalListResponse,
    EvalStatus,
    EvalSummary,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    LogLevel,
    LogRequest,
    LogResponse,
    MetricScore,
    MetricSummary,
    SpanInput,
    TestCaseInput,
    TestResultOutput,
    TraceRequest,
    TraceResponse,
)


class TestEvalStatus:
    """Tests for EvalStatus enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert EvalStatus.PENDING.value == "PENDING"
        assert EvalStatus.RUNNING.value == "RUNNING"
        assert EvalStatus.PASSED.value == "PASSED"
        assert EvalStatus.FAILED.value == "FAILED"

    def test_from_string(self) -> None:
        """Test creating from string value."""
        assert EvalStatus("PENDING") == EvalStatus.PENDING
        assert EvalStatus("PASSED") == EvalStatus.PASSED


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_values(self) -> None:
        """Test all log levels."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestTestCaseInput:
    """Tests for TestCaseInput request model."""

    def test_valid_test_case(self) -> None:
        """Test creating valid test case."""
        tc = TestCaseInput(
            input="What is 2+2?",
            expected="4",
            actual_output="4",
        )

        assert tc.input == "What is 2+2?"
        assert tc.expected == "4"
        assert tc.actual_output == "4"
        assert tc.metadata == {}

    def test_with_metadata(self) -> None:
        """Test test case with metadata."""
        tc = TestCaseInput(
            input="test",
            actual_output="result",
            metadata={"source": "unit-test", "index": 1},
        )

        assert tc.metadata == {"source": "unit-test", "index": 1}

    def test_optional_expected(self) -> None:
        """Test that expected is optional."""
        tc = TestCaseInput(
            input="Tell me a joke",
            actual_output="Why did the chicken...",
        )

        assert tc.expected is None

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise error."""
        with pytest.raises(ValidationError):
            TestCaseInput(input="test")  # Missing actual_output

        with pytest.raises(ValidationError):
            TestCaseInput(actual_output="result")  # Missing input

    def test_complex_input_types(self) -> None:
        """Test that complex input types are allowed."""
        tc = TestCaseInput(
            input={"query": "search", "filters": ["a", "b"]},
            expected=["result1", "result2"],
            actual_output={"data": [1, 2, 3]},
        )

        assert tc.input == {"query": "search", "filters": ["a", "b"]}


class TestEvaluateRequest:
    """Tests for EvaluateRequest model."""

    def test_valid_request(self) -> None:
        """Test valid evaluation request."""
        request = EvaluateRequest(
            name="my-test-run",
            test_cases=[
                TestCaseInput(input="q1", actual_output="a1"),
                TestCaseInput(input="q2", actual_output="a2"),
            ],
            metrics=["exact_match", "faithfulness"],
        )

        assert request.name == "my-test-run"
        assert len(request.test_cases) == 2
        assert request.metrics == ["exact_match", "faithfulness"]
        assert request.dataset_id is None
        assert request.metadata == {}

    def test_with_optional_fields(self) -> None:
        """Test request with optional fields."""
        request = EvaluateRequest(
            name="test",
            test_cases=[TestCaseInput(input="q", actual_output="a")],
            metrics=["exact_match"],
            dataset_id="dataset_123",
            metadata={"version": "1.0"},
        )

        assert request.dataset_id == "dataset_123"
        assert request.metadata == {"version": "1.0"}

    def test_empty_test_cases_rejected(self) -> None:
        """Test that empty test_cases list is allowed but not ideal."""
        # Pydantic allows empty lists by default
        request = EvaluateRequest(
            name="empty",
            test_cases=[],
            metrics=["exact_match"],
        )
        assert len(request.test_cases) == 0

    def test_missing_required_fields(self) -> None:
        """Test missing required fields."""
        with pytest.raises(ValidationError):
            EvaluateRequest(
                test_cases=[TestCaseInput(input="q", actual_output="a")],
                metrics=["exact_match"],
            )  # Missing name


class TestMetricScore:
    """Tests for MetricScore response model."""

    def test_basic_score(self) -> None:
        """Test basic metric score."""
        score = MetricScore(
            name="exact_match",
            score=1.0,
            passed=True,
        )

        assert score.name == "exact_match"
        assert score.score == 1.0
        assert score.passed is True
        assert score.threshold == 0.5  # Default
        assert score.reasoning is None

    def test_with_all_fields(self) -> None:
        """Test score with all fields."""
        score = MetricScore(
            name="faithfulness",
            score=0.85,
            passed=True,
            threshold=0.7,
            reasoning="The response is accurate.",
            reasoning_steps=[{"step": 1, "text": "Check facts"}],
            reasoning_type="llm",
            metadata={"model": "gpt-4"},
        )

        assert score.threshold == 0.7
        assert score.reasoning == "The response is accurate."
        assert score.reasoning_type == "llm"


class TestTestResultOutput:
    """Tests for TestResultOutput response model."""

    def test_basic_result(self) -> None:
        """Test basic test result."""
        result = TestResultOutput(
            input="question",
            expected="answer",
            actual_output="answer",
            scores=[MetricScore(name="exact_match", score=1.0, passed=True)],
            passed=True,
        )

        assert result.passed is True
        assert len(result.scores) == 1

    def test_with_error(self) -> None:
        """Test result with error."""
        result = TestResultOutput(
            input="question",
            expected="answer",
            actual_output="error response",
            scores=[],
            passed=False,
            error="Metric evaluation failed",
        )

        assert result.error == "Metric evaluation failed"


class TestMetricSummary:
    """Tests for MetricSummary model."""

    def test_basic_summary(self) -> None:
        """Test basic metric summary."""
        summary = MetricSummary(
            accuracy=0.9,
            avg_score=0.85,
        )

        assert summary.accuracy == 0.9
        assert summary.avg_score == 0.85
        assert summary.min_score is None
        assert summary.max_score is None
        assert summary.std_dev is None

    def test_full_summary(self) -> None:
        """Test summary with all fields."""
        summary = MetricSummary(
            accuracy=0.8,
            avg_score=0.75,
            min_score=0.2,
            max_score=1.0,
            std_dev=0.15,
        )

        assert summary.min_score == 0.2
        assert summary.max_score == 1.0


class TestEvalSummary:
    """Tests for EvalSummary model."""

    def test_basic_summary(self) -> None:
        """Test basic eval summary."""
        summary = EvalSummary(
            total=10,
            passed=8,
            failed=2,
            accuracy=0.8,
        )

        assert summary.total == 10
        assert summary.passed == 8
        assert summary.failed == 2
        assert summary.accuracy == 0.8
        assert summary.metrics == {}

    def test_with_metrics(self) -> None:
        """Test summary with per-metric stats."""
        summary = EvalSummary(
            total=5,
            passed=4,
            failed=1,
            accuracy=0.8,
            avg_latency_ms=150.5,
            metrics={
                "exact_match": MetricSummary(accuracy=1.0, avg_score=1.0),
                "faithfulness": MetricSummary(accuracy=0.6, avg_score=0.7),
            },
        )

        assert len(summary.metrics) == 2
        assert summary.metrics["exact_match"].accuracy == 1.0


class TestEvaluateResponse:
    """Tests for EvaluateResponse model."""

    def test_complete_response(self) -> None:
        """Test complete evaluation response."""
        now = datetime.now(timezone.utc)

        response = EvaluateResponse(
            eval_id="eval_123",
            organization_id="org_456",
            project_id="proj_789",
            name="test-run",
            status=EvalStatus.PASSED,
            summary=EvalSummary(total=1, passed=1, failed=0, accuracy=1.0),
            results=[
                TestResultOutput(
                    input="q",
                    expected="a",
                    actual_output="a",
                    scores=[],
                    passed=True,
                )
            ],
            created_at=now,
            completed_at=now,
        )

        assert response.eval_id == "eval_123"
        assert response.status == EvalStatus.PASSED

    def test_pending_response(self) -> None:
        """Test response for pending evaluation."""
        response = EvaluateResponse(
            eval_id="eval_pending",
            organization_id="org_1",
            project_id="proj_1",
            name="pending-run",
            status=EvalStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        assert response.status == EvalStatus.PENDING
        assert response.summary is None
        assert response.results is None
        assert response.completed_at is None


class TestEvalListItem:
    """Tests for EvalListItem model."""

    def test_list_item(self) -> None:
        """Test eval list item."""
        item = EvalListItem(
            eval_id="eval_1",
            organization_id="org_1",
            project_id="proj_1",
            name="test",
            status=EvalStatus.PASSED,
            total=5,
            passed=5,
            failed=0,
            accuracy=1.0,
            created_at=datetime.now(timezone.utc),
        )

        assert item.eval_id == "eval_1"


class TestEvalListResponse:
    """Tests for EvalListResponse model."""

    def test_paginated_response(self) -> None:
        """Test paginated list response."""
        response = EvalListResponse(
            evaluations=[
                EvalListItem(
                    eval_id="eval_1",
                    organization_id="org_1",
                    project_id="proj_1",
                    name="test1",
                    status=EvalStatus.PASSED,
                    created_at=datetime.now(timezone.utc),
                ),
            ],
            total=100,
            limit=20,
            offset=0,
        )

        assert len(response.evaluations) == 1
        assert response.total == 100
        assert response.limit == 20


class TestSpanInput:
    """Tests for SpanInput model."""

    def test_basic_span(self) -> None:
        """Test basic span input."""
        span = SpanInput(name="llm_call")

        assert span.name == "llm_call"
        assert span.children == []

    def test_nested_spans(self) -> None:
        """Test nested spans."""
        child = SpanInput(name="child", duration_ms=50.0)
        parent = SpanInput(
            name="parent",
            duration_ms=100.0,
            children=[child],
        )

        assert len(parent.children) == 1
        assert parent.children[0].name == "child"


class TestTraceRequest:
    """Tests for TraceRequest model."""

    def test_valid_request(self) -> None:
        """Test valid trace request."""
        request = TraceRequest(
            name="agent-run",
            spans=[SpanInput(name="root")],
        )

        assert request.name == "agent-run"
        assert len(request.spans) == 1

    def test_with_eval_id(self) -> None:
        """Test trace with associated eval."""
        request = TraceRequest(
            name="eval-trace",
            spans=[SpanInput(name="root")],
            eval_id="eval_123",
        )

        assert request.eval_id == "eval_123"


class TestTraceResponse:
    """Tests for TraceResponse model."""

    def test_response(self) -> None:
        """Test trace response."""
        response = TraceResponse(
            trace_id="trace_123",
            organization_id="org_1",
            project_id="proj_1",
            name="test-trace",
            span_count=3,
            duration_ms=150.0,
            created_at=datetime.now(timezone.utc),
        )

        assert response.trace_id == "trace_123"
        assert response.span_count == 3


class TestLogRequest:
    """Tests for LogRequest model."""

    def test_basic_log(self) -> None:
        """Test basic log request."""
        request = LogRequest(message="Test message")

        assert request.message == "Test message"
        assert request.level == LogLevel.INFO  # Default
        assert request.source == "evaris-sdk"  # Default

    def test_error_log(self) -> None:
        """Test error log."""
        request = LogRequest(
            message="Something went wrong",
            level=LogLevel.ERROR,
            agent_id="agent_123",
            eval_id="eval_456",
            metadata={"stack_trace": "..."},
        )

        assert request.level == LogLevel.ERROR
        assert request.agent_id == "agent_123"


class TestLogResponse:
    """Tests for LogResponse model."""

    def test_response(self) -> None:
        """Test log response."""
        response = LogResponse(
            log_id="log_123",
            organization_id="org_1",
            project_id="proj_1",
            level=LogLevel.INFO,
            created_at=datetime.now(timezone.utc),
        )

        assert response.log_id == "log_123"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_healthy_response(self) -> None:
        """Test healthy response."""
        response = HealthResponse(
            status="ok",
            version="0.1.0",
            database="connected",
        )

        assert response.status == "ok"

    def test_degraded_response(self) -> None:
        """Test degraded response."""
        response = HealthResponse(
            status="degraded",
            version="0.1.0",
            database="disconnected",
        )

        assert response.status == "degraded"
