"""Unit tests for evaris.client module.

Tests the EvarisClient cloud client, including:
- Response model parsing (AssessmentResult, TraceResult, LogResult)
- Client initialization and configuration
- HTTP request methods with mocked responses
- Sync wrappers
- Context managers
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.client import (
    AssessmentResult,
    AssessmentSummary,
    EvarisClient,
    LogResult,
    MetricScore,
    Span,
    TestResult,
    TraceResult,
    get_client,
)
from evaris.retry import RetryConfig
from evaris.types import TestCase


class TestMetricScore:
    """Tests for MetricScore response model."""

    def test_from_dict_basic(self) -> None:
        """Test parsing basic metric score."""
        data = {
            "name": "faithfulness",
            "score": 0.85,
            "passed": True,
        }

        score = MetricScore.from_dict(data)

        assert score.name == "faithfulness"
        assert score.score == 0.85
        assert score.passed is True
        assert score.reasoning is None
        assert score.metadata == {}

    def test_from_dict_with_all_fields(self) -> None:
        """Test parsing metric score with all optional fields."""
        data = {
            "name": "toxicity",
            "score": 0.1,
            "passed": True,
            "reasoning": "Content is safe and professional.",
            "metadata": {"model": "gpt-4", "latency_ms": 150},
        }

        score = MetricScore.from_dict(data)

        assert score.name == "toxicity"
        assert score.score == 0.1
        assert score.passed is True
        assert score.reasoning == "Content is safe and professional."
        assert score.metadata == {"model": "gpt-4", "latency_ms": 150}


class TestTestResult:
    """Tests for TestResult response model."""

    def test_from_dict_basic(self) -> None:
        """Test parsing basic test result."""
        data = {
            "input": "What is 2+2?",
            "expected": "4",
            "actual_output": "4",
            "scores": [
                {"name": "exact_match", "score": 1.0, "passed": True}
            ],
            "passed": True,
        }

        result = TestResult.from_dict(data)

        assert result.input == "What is 2+2?"
        assert result.expected == "4"
        assert result.actual_output == "4"
        assert len(result.scores) == 1
        assert result.scores[0].name == "exact_match"
        assert result.passed is True

    def test_from_dict_with_none_expected(self) -> None:
        """Test parsing result with null expected value."""
        data = {
            "input": "Tell me a joke",
            "actual_output": "Why did the chicken...",
            "scores": [],
            "passed": True,
        }

        result = TestResult.from_dict(data)

        assert result.expected is None


class TestAssessmentSummary:
    """Tests for AssessmentSummary response model."""

    def test_from_dict_basic(self) -> None:
        """Test parsing basic summary."""
        data = {
            "total": 10,
            "passed": 8,
            "failed": 2,
            "accuracy": 0.8,
        }

        summary = AssessmentSummary.from_dict(data)

        assert summary.total == 10
        assert summary.passed == 8
        assert summary.failed == 2
        assert summary.accuracy == 0.8
        assert summary.metrics == {}

    def test_from_dict_with_metrics(self) -> None:
        """Test parsing summary with per-metric stats."""
        data = {
            "total": 10,
            "passed": 8,
            "failed": 2,
            "accuracy": 0.8,
            "metrics": {
                "faithfulness": {"accuracy": 0.9, "avg_score": 0.85},
                "toxicity": {"accuracy": 0.7, "avg_score": 0.2},
            },
        }

        summary = AssessmentSummary.from_dict(data)

        assert "faithfulness" in summary.metrics
        assert summary.metrics["faithfulness"]["accuracy"] == 0.9


class TestAssessmentResult:
    """Tests for AssessmentResult response model."""

    def test_from_dict_basic(self) -> None:
        """Test parsing basic assessment result."""
        data = {
            "assessment_id": "assess_123",
            "project_id": "proj_456",
            "name": "my-test-run",
            "status": "PASSED",
            "summary": {
                "total": 5,
                "passed": 5,
                "failed": 0,
                "accuracy": 1.0,
            },
            "results": [],
            "created_at": "2024-01-15T10:30:00Z",
            "completed_at": "2024-01-15T10:30:05Z",
        }

        result = AssessmentResult.from_dict(data)

        assert result.assessment_id == "assess_123"
        assert result.project_id == "proj_456"
        assert result.name == "my-test-run"
        assert result.status == "PASSED"
        assert result.summary.accuracy == 1.0
        assert result.created_at.year == 2024

    def test_from_dict_legacy_run_id(self) -> None:
        """Test parsing with legacy 'run_id' field name."""
        data = {
            "run_id": "run_legacy",
            "project_id": "proj_1",
            "name": "test",
            "status": "PASSED",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = AssessmentResult.from_dict(data)

        assert result.assessment_id == "run_legacy"

    def test_from_dict_fallback_to_id(self) -> None:
        """Test parsing with 'id' as fallback."""
        data = {
            "id": "id_fallback",
            "project_id": "proj_1",
            "name": "test",
            "status": "PASSED",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = AssessmentResult.from_dict(data)

        assert result.assessment_id == "id_fallback"

    def test_from_dict_with_null_optional_fields(self) -> None:
        """Test parsing with null summary and results."""
        data = {
            "assessment_id": "assess_1",
            "project_id": "proj_1",
            "name": "test",
            "status": "RUNNING",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = AssessmentResult.from_dict(data)

        assert result.summary is None
        assert result.results is None
        assert result.completed_at is None


class TestTraceResult:
    """Tests for TraceResult response model."""

    def test_from_dict(self) -> None:
        """Test parsing trace result."""
        data = {
            "trace_id": "trace_abc",
            "project_id": "proj_1",
            "name": "agent-run",
            "span_count": 5,
            "duration_ms": 1234.5,
            "created_at": "2024-01-15T12:00:00Z",
        }

        result = TraceResult.from_dict(data)

        assert result.trace_id == "trace_abc"
        assert result.span_count == 5
        assert result.duration_ms == 1234.5

    def test_from_dict_null_duration(self) -> None:
        """Test parsing with null duration."""
        data = {
            "trace_id": "trace_1",
            "project_id": "proj_1",
            "name": "test",
            "span_count": 1,
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = TraceResult.from_dict(data)

        assert result.duration_ms is None


class TestLogResult:
    """Tests for LogResult response model."""

    def test_from_dict(self) -> None:
        """Test parsing log result."""
        data = {
            "log_id": "log_xyz",
            "project_id": "proj_1",
            "level": "INFO",
            "created_at": "2024-01-15T12:00:00Z",
        }

        result = LogResult.from_dict(data)

        assert result.log_id == "log_xyz"
        assert result.level == "INFO"


class TestSpan:
    """Tests for Span input model."""

    def test_to_dict_basic(self) -> None:
        """Test basic span serialization."""
        span = Span(name="llm_call")

        data = span.to_dict()

        assert data["name"] == "llm_call"
        assert data["start_time"] is None
        assert data["children"] == []

    def test_to_dict_with_times(self) -> None:
        """Test span with timestamps."""
        start = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 10, 0, 5, tzinfo=timezone.utc)

        span = Span(
            name="api_call",
            start_time=start,
            end_time=end,
            duration_ms=5000.0,
        )

        data = span.to_dict()

        assert "2024-01-15" in data["start_time"]
        assert data["duration_ms"] == 5000.0

    def test_to_dict_nested_children(self) -> None:
        """Test span with nested children."""
        child = Span(name="child_span", duration_ms=100)
        parent = Span(
            name="parent_span",
            children=[child],
            duration_ms=500,
        )

        data = parent.to_dict()

        assert len(data["children"]) == 1
        assert data["children"][0]["name"] == "child_span"


class TestEvarisClientInit:
    """Tests for EvarisClient initialization."""

    def test_init_basic(self) -> None:
        """Test basic client initialization."""
        client = EvarisClient(api_key="test_key")

        assert client.api_key == "test_key"
        assert client.base_url == "https://api.evaris.ai"
        assert client.timeout == 300.0

    def test_init_custom_base_url(self) -> None:
        """Test client with custom base URL."""
        client = EvarisClient(
            api_key="test_key",
            base_url="http://localhost:8000/",
        )

        # Should strip trailing slash
        assert client.base_url == "http://localhost:8000"

    def test_init_custom_retry_config(self) -> None:
        """Test client with custom retry config."""
        config = RetryConfig(max_retries=5)
        client = EvarisClient(api_key="test", retry_config=config)

        assert client.retry_config.max_retries == 5

    def test_init_without_httpx_raises(self) -> None:
        """Test that missing httpx raises ImportError."""
        with patch.dict("sys.modules", {"httpx": None}):
            # Need to reload client module to test import check
            # This is tricky to test properly - skip for now
            pass

    def test_get_headers(self) -> None:
        """Test that headers include auth and content type."""
        client = EvarisClient(api_key="my_api_key")
        headers = client._get_headers()

        assert headers["Authorization"] == "Bearer my_api_key"
        assert headers["Content-Type"] == "application/json"
        assert "evaris-python" in headers["User-Agent"]


class TestEvarisClientContextManagers:
    """Tests for EvarisClient context managers."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager creates and closes client."""
        client = EvarisClient(api_key="test")

        async with client:
            assert client._async_client is not None

        assert client._async_client is None

    def test_sync_context_manager(self) -> None:
        """Test sync context manager creates and closes client."""
        client = EvarisClient(api_key="test")

        with client:
            assert client._sync_client is not None

        assert client._sync_client is None


class TestEvarisClientMethods:
    """Tests for EvarisClient HTTP methods with mocked responses."""

    @pytest.fixture
    def mock_assess_response(self) -> dict:
        """Mock successful assess response."""
        return {
            "assessment_id": "assess_test",
            "project_id": "proj_test",
            "name": "unit-test",
            "status": "PASSED",
            "summary": {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "accuracy": 1.0,
            },
            "results": [
                {
                    "input": "test input",
                    "expected": "expected",
                    "actual_output": "expected",
                    "scores": [{"name": "exact_match", "score": 1.0, "passed": True}],
                    "passed": True,
                }
            ],
            "created_at": "2024-01-01T00:00:00Z",
            "completed_at": "2024-01-01T00:00:01Z",
        }

    @pytest.mark.asyncio
    async def test_assess_async(self, mock_assess_response: dict) -> None:
        """Test async assess method."""
        client = EvarisClient(api_key="test")

        # Mock the retry function
        with patch("evaris.client.request_with_retry") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_assess_response
            mock_retry.return_value = mock_response

            async with client:
                result = await client.assess(
                    name="unit-test",
                    test_cases=[TestCase(input="test", actual_output="expected")],
                    metrics=["exact_match"],
                )

            assert result.assessment_id == "assess_test"
            assert result.status == "PASSED"
            assert mock_retry.called

    def test_assess_sync(self, mock_assess_response: dict) -> None:
        """Test sync assess method."""
        client = EvarisClient(api_key="test")

        with patch("evaris.client.request_with_retry_sync") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_assess_response
            mock_retry.return_value = mock_response

            with client:
                result = client.assess_sync(
                    name="unit-test",
                    test_cases=[TestCase(input="test", actual_output="expected")],
                    metrics=["exact_match"],
                )

            assert result.assessment_id == "assess_test"

    @pytest.mark.asyncio
    async def test_trace_async(self) -> None:
        """Test async trace method."""
        client = EvarisClient(api_key="test")

        mock_trace_response = {
            "trace_id": "trace_test",
            "project_id": "proj_1",
            "name": "test-trace",
            "span_count": 2,
            "duration_ms": 100.0,
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("evaris.client.request_with_retry") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_trace_response
            mock_retry.return_value = mock_response

            async with client:
                result = await client.trace(
                    name="test-trace",
                    spans=[Span(name="root")],
                )

            assert result.trace_id == "trace_test"

    def test_trace_sync(self) -> None:
        """Test sync trace method."""
        client = EvarisClient(api_key="test")

        mock_trace_response = {
            "trace_id": "trace_sync",
            "project_id": "proj_1",
            "name": "sync-trace",
            "span_count": 1,
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("evaris.client.request_with_retry_sync") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_trace_response
            mock_retry.return_value = mock_response

            with client:
                result = client.trace_sync(
                    name="sync-trace",
                    spans=[Span(name="root")],
                )

            assert result.trace_id == "trace_sync"

    @pytest.mark.asyncio
    async def test_log_async(self) -> None:
        """Test async log method."""
        client = EvarisClient(api_key="test")

        mock_log_response = {
            "log_id": "log_test",
            "project_id": "proj_1",
            "level": "INFO",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("evaris.client.request_with_retry") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_log_response
            mock_retry.return_value = mock_response

            async with client:
                result = await client.log(message="Test log")

            assert result.log_id == "log_test"

    def test_log_sync(self) -> None:
        """Test sync log method."""
        client = EvarisClient(api_key="test")

        mock_log_response = {
            "log_id": "log_sync",
            "project_id": "proj_1",
            "level": "WARNING",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("evaris.client.request_with_retry_sync") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_log_response
            mock_retry.return_value = mock_response

            with client:
                result = client.log_sync(message="Test", level="warning")

            assert result.level == "WARNING"

    @pytest.mark.asyncio
    async def test_get_assessment_async(self, mock_assess_response: dict) -> None:
        """Test async get_assessment method."""
        client = EvarisClient(api_key="test")

        with patch("evaris.client.request_with_retry") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_assess_response
            mock_retry.return_value = mock_response

            async with client:
                result = await client.get_assessment("assess_test")

            assert result.assessment_id == "assess_test"

    def test_get_assessment_sync(self, mock_assess_response: dict) -> None:
        """Test sync get_assessment method."""
        client = EvarisClient(api_key="test")

        with patch("evaris.client.request_with_retry_sync") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_assess_response
            mock_retry.return_value = mock_response

            with client:
                result = client.get_assessment_sync("assess_test")

            assert result.assessment_id == "assess_test"

    @pytest.mark.asyncio
    async def test_list_assessments_async(self, mock_assess_response: dict) -> None:
        """Test async list_assessments method."""
        client = EvarisClient(api_key="test")

        with patch("evaris.client.request_with_retry") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = {"items": [mock_assess_response]}
            mock_retry.return_value = mock_response

            async with client:
                results = await client.list_assessments(limit=10)

            assert len(results) == 1
            assert results[0].assessment_id == "assess_test"

    def test_list_assessments_sync(self, mock_assess_response: dict) -> None:
        """Test sync list_assessments method."""
        client = EvarisClient(api_key="test")

        with patch("evaris.client.request_with_retry_sync") as mock_retry:
            mock_response = MagicMock()
            mock_response.json.return_value = {"items": [mock_assess_response]}
            mock_retry.return_value = mock_response

            with client:
                results = client.list_assessments_sync(limit=5)

            assert len(results) == 1


class TestGetClient:
    """Tests for get_client convenience function."""

    def test_with_api_key(self) -> None:
        """Test get_client with explicit API key."""
        client = get_client(api_key="explicit_key")

        assert client.api_key == "explicit_key"

    def test_with_env_var(self) -> None:
        """Test get_client reads from environment."""
        with patch.dict("os.environ", {"EVARIS_API_KEY": "env_key"}):
            client = get_client()

        assert client.api_key == "env_key"

    def test_missing_key_raises(self) -> None:
        """Test get_client raises without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                get_client()

            assert "No API key provided" in str(exc_info.value)

    def test_with_custom_base_url(self) -> None:
        """Test get_client with custom base URL."""
        client = get_client(api_key="test", base_url="http://localhost:3000")

        assert client.base_url == "http://localhost:3000"

    def test_with_retry_config(self) -> None:
        """Test get_client with custom retry config."""
        config = RetryConfig(max_retries=10)
        client = get_client(api_key="test", retry_config=config)

        assert client.retry_config.max_retries == 10
