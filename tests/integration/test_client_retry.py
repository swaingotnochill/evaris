"""Integration tests for EvarisClient retry behavior.

Tests the client's retry logic against simulated server responses
using respx to mock httpx at the transport level.
"""

import pytest
import respx
from httpx import Response

from evaris import EvarisClient, RetryConfig, RetryExhaustedError, TestCase


@pytest.fixture
def api_key() -> str:
    """Test API key."""
    return "test_api_key_for_integration"


@pytest.fixture
def base_url() -> str:
    """Test base URL."""
    return "https://api.evaris.test"


@pytest.fixture
def mock_assess_success() -> dict:
    """Successful assessment response."""
    return {
        "assessment_id": "assess_integration_123",
        "project_id": "proj_test",
        "name": "integration-test",
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
                "expected": "expected output",
                "actual_output": "expected output",
                "scores": [{"name": "exact_match", "score": 1.0, "passed": True}],
                "passed": True,
            }
        ],
        "created_at": "2024-01-15T10:00:00Z",
        "completed_at": "2024-01-15T10:00:05Z",
    }


class TestRetryOn503:
    """Tests for retry behavior on 503 Service Unavailable."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_success_after_503(
        self,
        api_key: str,
        base_url: str,
        mock_assess_success: dict,
    ) -> None:
        """Test successful retry after initial 503."""
        # First call returns 503, second succeeds
        route = respx.post(f"{base_url}/internal/evaluate").mock(
            side_effect=[
                Response(503, text="Service Unavailable"),
                Response(201, json=mock_assess_success),
            ]
        )

        config = RetryConfig(
            max_retries=3,
            base_delay_ms=1,  # Fast for testing
            jitter=False,
        )

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            result = await client.assess(
                name="integration-test",
                test_cases=[TestCase(input="test", actual_output="output")],
                metrics=["exact_match"],
            )

        assert result.assessment_id == "assess_integration_123"
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_exhausted_after_max_503s(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test RetryExhaustedError after max retries on 503."""
        route = respx.post(f"{base_url}/internal/evaluate").mock(
            return_value=Response(503, text="Service Unavailable")
        )

        config = RetryConfig(
            max_retries=2,
            base_delay_ms=1,
            jitter=False,
        )

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            with pytest.raises(RetryExhaustedError) as exc_info:
                await client.assess(
                    name="failing-test",
                    test_cases=[TestCase(input="test", actual_output="output")],
                    metrics=["exact_match"],
                )

        assert exc_info.value.attempts == 3  # Initial + 2 retries
        assert exc_info.value.status_code == 503
        assert route.call_count == 3


class TestRetryOn429:
    """Tests for retry behavior on 429 Rate Limit."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_rate_limit(
        self,
        api_key: str,
        base_url: str,
        mock_assess_success: dict,
    ) -> None:
        """Test retry on rate limit (429)."""
        route = respx.post(f"{base_url}/internal/evaluate").mock(
            side_effect=[
                Response(429, text="Rate limit exceeded"),
                Response(429, text="Rate limit exceeded"),
                Response(201, json=mock_assess_success),
            ]
        )

        config = RetryConfig(
            max_retries=5,
            base_delay_ms=1,
            jitter=False,
        )

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            result = await client.assess(
                name="rate-limited-test",
                test_cases=[TestCase(input="test", actual_output="output")],
                metrics=["exact_match"],
            )

        assert result.status == "PASSED"
        assert route.call_count == 3


class TestNoRetryOn4xx:
    """Tests that client errors (4xx except 429) don't retry."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_retry_on_400(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test that 400 Bad Request doesn't retry."""
        import httpx

        route = respx.post(f"{base_url}/internal/evaluate").mock(
            return_value=Response(400, json={"detail": "Invalid request"})
        )

        async with EvarisClient(api_key=api_key, base_url=base_url) as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.assess(
                    name="bad-request-test",
                    test_cases=[TestCase(input="test", actual_output="output")],
                    metrics=["exact_match"],
                )

        assert exc_info.value.response.status_code == 400
        assert route.call_count == 1  # No retry

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_retry_on_401(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test that 401 Unauthorized doesn't retry."""
        import httpx

        route = respx.post(f"{base_url}/internal/evaluate").mock(
            return_value=Response(401, json={"detail": "Invalid API key"})
        )

        async with EvarisClient(api_key=api_key, base_url=base_url) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.assess(
                    name="unauthorized-test",
                    test_cases=[TestCase(input="test", actual_output="output")],
                    metrics=["exact_match"],
                )

        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_retry_on_404(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test that 404 Not Found doesn't retry."""
        import httpx

        route = respx.get(f"{base_url}/internal/evaluatements/nonexistent").mock(
            return_value=Response(404, json={"detail": "Not found"})
        )

        async with EvarisClient(api_key=api_key, base_url=base_url) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_assessment("nonexistent")

        assert route.call_count == 1


class TestOnRetryCallback:
    """Tests for on_retry callback."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_callback_invoked_on_retry(
        self,
        api_key: str,
        base_url: str,
        mock_assess_success: dict,
    ) -> None:
        """Test that on_retry callback is called with correct args."""
        respx.post(f"{base_url}/internal/evaluate").mock(
            side_effect=[
                Response(503),
                Response(503),
                Response(201, json=mock_assess_success),
            ]
        )

        callback_invocations = []

        def on_retry(attempt: int, delay: float, error: Exception | None) -> None:
            callback_invocations.append({
                "attempt": attempt,
                "delay": delay,
                "error_type": type(error).__name__ if error else None,
            })

        config = RetryConfig(
            max_retries=3,
            base_delay_ms=10,
            jitter=False,
        )

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
            on_retry=on_retry,
        ) as client:
            await client.assess(
                name="callback-test",
                test_cases=[TestCase(input="test", actual_output="output")],
                metrics=["exact_match"],
            )

        # Should have 2 retry callbacks (for attempts 0 and 1)
        assert len(callback_invocations) == 2
        assert callback_invocations[0]["attempt"] == 0
        assert callback_invocations[1]["attempt"] == 1


class TestSyncRetry:
    """Tests for sync client retry behavior."""

    @respx.mock
    def test_sync_retry_on_503(
        self,
        api_key: str,
        base_url: str,
        mock_assess_success: dict,
    ) -> None:
        """Test sync client retries on 503."""
        route = respx.post(f"{base_url}/internal/evaluate").mock(
            side_effect=[
                Response(503),
                Response(201, json=mock_assess_success),
            ]
        )

        config = RetryConfig(max_retries=2, base_delay_ms=1, jitter=False)

        with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            result = client.assess_sync(
                name="sync-retry-test",
                test_cases=[TestCase(input="test", actual_output="output")],
                metrics=["exact_match"],
            )

        assert result.assessment_id == "assess_integration_123"
        assert route.call_count == 2

    @respx.mock
    def test_sync_retry_exhausted(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test sync client raises after max retries."""
        route = respx.post(f"{base_url}/internal/evaluate").mock(
            return_value=Response(503)
        )

        config = RetryConfig(max_retries=1, base_delay_ms=1, jitter=False)

        with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            with pytest.raises(RetryExhaustedError) as exc_info:
                client.assess_sync(
                    name="sync-fail-test",
                    test_cases=[TestCase(input="test", actual_output="output")],
                    metrics=["exact_match"],
                )

        assert exc_info.value.attempts == 2
        assert route.call_count == 2


class TestOtherMethods:
    """Tests for retry on other client methods."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_trace_retry(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test trace method retries on 503."""
        from evaris import Span

        mock_response = {
            "trace_id": "trace_123",
            "project_id": "proj_1",
            "name": "test-trace",
            "span_count": 1,
            "duration_ms": 100.0,
            "created_at": "2024-01-01T00:00:00Z",
        }

        route = respx.post(f"{base_url}/internal/trace").mock(
            side_effect=[
                Response(503),
                Response(201, json=mock_response),
            ]
        )

        config = RetryConfig(max_retries=2, base_delay_ms=1, jitter=False)

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            result = await client.trace(
                name="test-trace",
                spans=[Span(name="root")],
            )

        assert result.trace_id == "trace_123"
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_log_retry(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        """Test log method retries on 503."""
        mock_response = {
            "log_id": "log_123",
            "project_id": "proj_1",
            "level": "INFO",
            "created_at": "2024-01-01T00:00:00Z",
        }

        route = respx.post(f"{base_url}/internal/log").mock(
            side_effect=[
                Response(503),
                Response(201, json=mock_response),
            ]
        )

        config = RetryConfig(max_retries=2, base_delay_ms=1, jitter=False)

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            result = await client.log(message="Test log")

        assert result.log_id == "log_123"
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_assessments_retry(
        self,
        api_key: str,
        base_url: str,
        mock_assess_success: dict,
    ) -> None:
        """Test list_assessments retries on 503."""
        route = respx.get(f"{base_url}/internal/evaluatements").mock(
            side_effect=[
                Response(503),
                Response(200, json={"items": [mock_assess_success]}),
            ]
        )

        config = RetryConfig(max_retries=2, base_delay_ms=1, jitter=False)

        async with EvarisClient(
            api_key=api_key,
            base_url=base_url,
            retry_config=config,
        ) as client:
            results = await client.list_assessments()

        assert len(results) == 1
        assert route.call_count == 2
