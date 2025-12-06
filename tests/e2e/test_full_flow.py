"""End-to-end tests for the full Evaris flow.

Tests the complete integration:
- Python SDK (EvarisClient)
- evaris-web (API gateway with auth) - optional
- evaris-server (evaluation runner)
- PostgreSQL (data persistence)

Run with: pytest tests/e2e -v --e2e

Requirements:
- evaris-server running on EVARIS_API_URL (default: http://localhost:8080)
- PostgreSQL database connected
- Valid test API key in EVARIS_TEST_API_KEY
"""

import time
from datetime import datetime, timezone
from typing import Any

import pytest

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


class TestAssessmentFlow:
    """E2E tests for the assessment (evaluation) flow."""

    def test_assess_sync_exact_match(
        self,
        evaris_client,
        sample_test_cases,
        cleanup_assessments: list[str],
    ) -> None:
        """Test synchronous assessment with exact_match metric."""
        from evaris import TestCase

        # Create test cases with known outcomes
        test_cases = [
            TestCase(
                input="What is 2+2?",
                expected="4",
                actual_output="4",  # Exact match
            ),
            TestCase(
                input="What is 3+3?",
                expected="6",
                actual_output="The answer is 6",  # Partial match
            ),
        ]

        result = evaris_client.assess_sync(
            name="e2e-test-exact-match",
            test_cases=test_cases,
            metrics=["exact_match"],
            metadata={"test_type": "e2e", "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        # Track for cleanup
        cleanup_assessments.append(result.assessment_id)

        # Verify response structure
        assert result.assessment_id is not None
        assert result.name == "e2e-test-exact-match"
        assert result.status in ("PASSED", "FAILED", "COMPLETED")

        # Verify summary
        assert result.summary is not None
        assert result.summary.total == 2
        assert result.summary.passed >= 0
        assert result.summary.failed >= 0
        assert result.summary.passed + result.summary.failed == result.summary.total

        # Verify individual results
        assert result.results is not None
        assert len(result.results) == 2

        for test_result in result.results:
            assert len(test_result.scores) > 0
            for score in test_result.scores:
                assert score.name == "exact_match"
                assert 0.0 <= score.score <= 1.0
                assert isinstance(score.passed, bool)

    @pytest.mark.asyncio
    async def test_assess_async_multiple_metrics(
        self,
        async_evaris_client,
        cleanup_assessments: list[str],
    ) -> None:
        """Test async assessment with multiple metrics."""
        from evaris import TestCase

        test_cases = [
            TestCase(
                input="Explain photosynthesis",
                expected="Plants convert sunlight into energy",
                actual_output="Photosynthesis is the process by which plants use sunlight to produce glucose from carbon dioxide and water.",
            ),
        ]

        result = await async_evaris_client.assess(
            name="e2e-test-multi-metric",
            test_cases=test_cases,
            metrics=["answer_relevance", "semantic_similarity"],
        )

        cleanup_assessments.append(result.assessment_id)

        assert result.assessment_id is not None
        assert result.summary is not None
        assert result.results is not None

        # Should have scores for multiple metrics
        if result.results:
            metric_names = {s.name for s in result.results[0].scores}
            assert len(metric_names) >= 1  # At least one metric evaluated

    def test_assess_empty_test_cases(self, evaris_client) -> None:
        """Test that empty test cases are handled properly."""
        # The server should either reject or handle empty test cases
        result = evaris_client.assess_sync(
            name="e2e-test-empty",
            test_cases=[],
            metrics=["exact_match"],
        )

        assert result.summary.total == 0

    def test_get_assessment_by_id(
        self,
        evaris_client,
        cleanup_assessments: list[str],
    ) -> None:
        """Test retrieving an assessment by ID."""
        from evaris import TestCase

        # First create an assessment
        test_cases = [
            TestCase(
                input="Test",
                expected="Test",
                actual_output="Test",
            ),
        ]

        created = evaris_client.assess_sync(
            name="e2e-test-retrieve",
            test_cases=test_cases,
            metrics=["exact_match"],
        )
        cleanup_assessments.append(created.assessment_id)

        # Now retrieve it
        retrieved = evaris_client.get_assessment_sync(created.assessment_id)

        assert retrieved.assessment_id == created.assessment_id
        assert retrieved.name == created.name
        assert retrieved.status == created.status

    def test_list_assessments(
        self,
        evaris_client,
        cleanup_assessments: list[str],
    ) -> None:
        """Test listing recent assessments."""
        from evaris import TestCase

        # Create a few assessments
        for i in range(3):
            result = evaris_client.assess_sync(
                name=f"e2e-test-list-{i}",
                test_cases=[TestCase(input=f"test-{i}", expected="ok", actual_output="ok")],
                metrics=["exact_match"],
            )
            cleanup_assessments.append(result.assessment_id)

        # List assessments
        assessments = evaris_client.list_assessments_sync(limit=10)

        # Should have at least the 3 we created
        assert len(assessments) >= 3

        # Verify structure
        for assessment in assessments:
            assert assessment.assessment_id is not None
            assert assessment.name is not None


class TestTraceFlow:
    """E2E tests for the tracing flow."""

    def test_trace_sync_single_span(self, evaris_client, sample_spans) -> None:
        """Test synchronous trace creation with a single span."""
        result = evaris_client.trace_sync(
            name="e2e-test-trace-single",
            spans=sample_spans[:1],  # Just first span
            duration_ms=150.5,
            metadata={"test": "e2e"},
        )

        assert result.trace_id is not None
        assert result.name == "e2e-test-trace-single"
        assert result.span_count >= 1

    @pytest.mark.asyncio
    async def test_trace_async_nested_spans(self, async_evaris_client) -> None:
        """Test async trace creation with nested spans."""
        from evaris import Span

        now = datetime.now(timezone.utc)

        nested_spans = [
            Span(
                name="agent_run",
                start_time=now,
                duration_ms=500.0,
                metadata={"agent": "test"},
                children=[
                    Span(
                        name="llm_call",
                        duration_ms=200.0,
                        input={"prompt": "Hello"},
                        output={"response": "Hi there!"},
                    ),
                    Span(
                        name="tool_call",
                        duration_ms=100.0,
                        input={"tool": "search"},
                        output={"results": []},
                    ),
                ],
            ),
        ]

        result = await async_evaris_client.trace(
            name="e2e-test-trace-nested",
            spans=nested_spans,
        )

        assert result.trace_id is not None
        assert result.span_count >= 1  # At least the parent span


class TestLogFlow:
    """E2E tests for the logging flow."""

    def test_log_sync_info(self, evaris_client) -> None:
        """Test synchronous log creation at INFO level."""
        result = evaris_client.log_sync(
            message="E2E test log message",
            level="info",
            metadata={"test_run": "e2e", "timestamp": time.time()},
        )

        assert result.log_id is not None
        assert result.level.lower() == "info"

    @pytest.mark.asyncio
    async def test_log_async_error(self, async_evaris_client) -> None:
        """Test async log creation at ERROR level."""
        result = await async_evaris_client.log(
            message="E2E test error log",
            level="error",
            metadata={"error_code": "E2E_TEST"},
        )

        assert result.log_id is not None
        assert result.level.lower() == "error"

    def test_log_with_custom_timestamp(self, evaris_client) -> None:
        """Test log with custom timestamp."""
        custom_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        result = evaris_client.log_sync(
            message="E2E test with custom timestamp",
            level="debug",
            timestamp=custom_time,
        )

        assert result.log_id is not None


class TestErrorHandling:
    """E2E tests for error handling and edge cases."""

    def test_invalid_metric_name(self, evaris_client) -> None:
        """Test behavior with invalid metric name."""
        from evaris import TestCase

        import httpx

        test_cases = [
            TestCase(
                input="test",
                expected="test",
                actual_output="test",
            ),
        ]

        # Server should either return error or ignore invalid metric
        try:
            result = evaris_client.assess_sync(
                name="e2e-test-invalid-metric",
                test_cases=test_cases,
                metrics=["nonexistent_metric_xyz"],
            )
            # If it succeeds, check results
            assert result.assessment_id is not None
        except httpx.HTTPStatusError as e:
            # 400 Bad Request is expected for invalid metrics
            assert e.response.status_code in (400, 422)

    def test_get_nonexistent_assessment(self, evaris_client) -> None:
        """Test retrieving a non-existent assessment."""
        import httpx

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            evaris_client.get_assessment_sync("nonexistent_assessment_id_12345")

        assert exc_info.value.response.status_code == 404

    def test_large_payload(
        self,
        evaris_client,
        cleanup_assessments: list[str],
    ) -> None:
        """Test handling of larger payloads."""
        from evaris import TestCase

        # Create 50 test cases
        test_cases = [
            TestCase(
                input=f"Question {i}: What is {i} * {i}?",
                expected=str(i * i),
                actual_output=str(i * i),
            )
            for i in range(50)
        ]

        result = evaris_client.assess_sync(
            name="e2e-test-large-payload",
            test_cases=test_cases,
            metrics=["exact_match"],
        )

        cleanup_assessments.append(result.assessment_id)

        assert result.summary.total == 50


class TestRetryBehavior:
    """E2E tests for retry behavior (if server simulates failures)."""

    def test_successful_request_no_retry(self, evaris_client) -> None:
        """Test that successful requests don't trigger retries."""
        from evaris import TestCase

        retry_count = 0

        def on_retry(attempt: int, delay: float, error: Exception | None) -> None:
            nonlocal retry_count
            retry_count += 1

        # Reconfigure client with callback
        from evaris import EvarisClient, RetryConfig

        config = RetryConfig(max_retries=3, base_delay_ms=100, jitter=False)

        with EvarisClient(
            api_key=evaris_client.api_key,
            base_url=evaris_client.base_url,
            retry_config=config,
            on_retry=on_retry,
        ) as client:
            result = client.assess_sync(
                name="e2e-test-no-retry",
                test_cases=[TestCase(input="test", expected="test", actual_output="test")],
                metrics=["exact_match"],
            )

        assert result.assessment_id is not None
        assert retry_count == 0  # No retries needed


class TestDataPersistence:
    """E2E tests verifying data is persisted correctly."""

    def test_assessment_persisted_and_retrievable(
        self,
        evaris_client,
        cleanup_assessments: list[str],
    ) -> None:
        """Test that assessment data is persisted and can be retrieved."""
        from evaris import TestCase

        unique_name = f"e2e-persist-test-{int(time.time())}"

        test_cases = [
            TestCase(
                input="What color is the sky?",
                expected="Blue",
                actual_output="The sky is blue.",
            ),
        ]

        # Create assessment
        created = evaris_client.assess_sync(
            name=unique_name,
            test_cases=test_cases,
            metrics=["exact_match"],
            metadata={"persist_test": True},
        )
        cleanup_assessments.append(created.assessment_id)

        # Retrieve it
        retrieved = evaris_client.get_assessment_sync(created.assessment_id)

        # Verify data integrity
        assert retrieved.name == unique_name
        assert retrieved.summary.total == 1
        assert retrieved.results is not None
        assert len(retrieved.results) == 1
        assert retrieved.results[0].input == "What color is the sky?"

    def test_assessment_appears_in_list(
        self,
        evaris_client,
        cleanup_assessments: list[str],
    ) -> None:
        """Test that created assessment appears in the list."""
        from evaris import TestCase

        unique_name = f"e2e-list-test-{int(time.time())}"

        created = evaris_client.assess_sync(
            name=unique_name,
            test_cases=[TestCase(input="test", expected="test", actual_output="test")],
            metrics=["exact_match"],
        )
        cleanup_assessments.append(created.assessment_id)

        # List recent assessments
        assessments = evaris_client.list_assessments_sync(limit=50)

        # Find our assessment
        found = next(
            (a for a in assessments if a.assessment_id == created.assessment_id),
            None,
        )

        assert found is not None
        assert found.name == unique_name
