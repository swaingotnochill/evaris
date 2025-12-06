"""Unit tests for evaris.retry module.

Tests the retry logic with exponential backoff, including:
- RetryConfig defaults and custom values
- Backoff delay calculation with jitter
- Status code and exception filtering
- Retry loop behavior with mocked HTTP client
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.retry import (
    DEFAULT_RETRY_STATUS_CODES,
    RetryConfig,
    RetryExhaustedError,
    aggressive_retry_config,
    calculate_backoff_delay,
    default_retry_config,
    no_retry_config,
    request_with_retry,
    request_with_retry_sync,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay_ms == 100.0
        assert config.max_delay_ms == 30000.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on_status_codes == DEFAULT_RETRY_STATUS_CODES

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay_ms=200.0,
            max_delay_ms=60000.0,
            exponential_base=3.0,
            jitter=False,
            retry_on_status_codes={500, 502},
        )

        assert config.max_retries == 5
        assert config.base_delay_ms == 200.0
        assert config.max_delay_ms == 60000.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retry_on_status_codes == {500, 502}

    def test_should_retry_status_retryable(self) -> None:
        """Test that retryable status codes are identified."""
        config = RetryConfig()

        # All default retryable codes
        assert config.should_retry_status(429) is True  # Rate limit
        assert config.should_retry_status(500) is True  # Internal error
        assert config.should_retry_status(502) is True  # Bad gateway
        assert config.should_retry_status(503) is True  # Service unavailable
        assert config.should_retry_status(504) is True  # Gateway timeout

    def test_should_retry_status_non_retryable(self) -> None:
        """Test that non-retryable status codes are rejected."""
        config = RetryConfig()

        # Client errors should not retry
        assert config.should_retry_status(400) is False  # Bad request
        assert config.should_retry_status(401) is False  # Unauthorized
        assert config.should_retry_status(403) is False  # Forbidden
        assert config.should_retry_status(404) is False  # Not found

        # Success should not retry
        assert config.should_retry_status(200) is False
        assert config.should_retry_status(201) is False

    def test_should_retry_exception_retryable(self) -> None:
        """Test that retryable exceptions are identified."""
        config = RetryConfig()

        # Mock exceptions with retryable names
        timeout_exc = type("TimeoutException", (Exception,), {})()
        connect_exc = type("ConnectError", (Exception,), {})()
        protocol_exc = type("RemoteProtocolError", (Exception,), {})()

        assert config.should_retry_exception(timeout_exc) is True
        assert config.should_retry_exception(connect_exc) is True
        assert config.should_retry_exception(protocol_exc) is True

    def test_should_retry_exception_non_retryable(self) -> None:
        """Test that non-retryable exceptions are rejected."""
        config = RetryConfig()

        # Standard exceptions should not retry
        assert config.should_retry_exception(ValueError("test")) is False
        assert config.should_retry_exception(RuntimeError("test")) is False
        assert config.should_retry_exception(KeyError("test")) is False


class TestCalculateBackoffDelay:
    """Tests for calculate_backoff_delay function."""

    def test_exponential_growth_no_jitter(self) -> None:
        """Test exponential backoff without jitter."""
        config = RetryConfig(
            base_delay_ms=100.0,
            exponential_base=2.0,
            jitter=False,
        )

        # delay = base * (base ^ attempt)
        assert calculate_backoff_delay(0, config) == 0.1    # 100ms
        assert calculate_backoff_delay(1, config) == 0.2    # 200ms
        assert calculate_backoff_delay(2, config) == 0.4    # 400ms
        assert calculate_backoff_delay(3, config) == 0.8    # 800ms

    def test_max_delay_cap(self) -> None:
        """Test that delay is capped at max_delay_ms."""
        config = RetryConfig(
            base_delay_ms=1000.0,
            max_delay_ms=2000.0,
            exponential_base=2.0,
            jitter=False,
        )

        # Attempt 0: 1000ms
        assert calculate_backoff_delay(0, config) == 1.0
        # Attempt 1: 2000ms (at cap)
        assert calculate_backoff_delay(1, config) == 2.0
        # Attempt 2: would be 4000ms, but capped at 2000ms
        assert calculate_backoff_delay(2, config) == 2.0
        # Attempt 10: still capped
        assert calculate_backoff_delay(10, config) == 2.0

    def test_jitter_reduces_delay(self) -> None:
        """Test that jitter reduces delay (between 0.5x and 1.0x)."""
        config = RetryConfig(
            base_delay_ms=1000.0,
            jitter=True,
        )

        # Run multiple times to verify jitter is applied
        delays = [calculate_backoff_delay(0, config) for _ in range(100)]

        # All delays should be between 0.5s and 1.0s
        assert all(0.5 <= d <= 1.0 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 1

    def test_custom_exponential_base(self) -> None:
        """Test with custom exponential base."""
        config = RetryConfig(
            base_delay_ms=100.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert calculate_backoff_delay(0, config) == 0.1    # 100ms
        assert calculate_backoff_delay(1, config) == 0.3    # 300ms
        assert calculate_backoff_delay(2, config) == 0.9    # 900ms


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError exception."""

    def test_basic_creation(self) -> None:
        """Test creating error with basic attributes."""
        error = RetryExhaustedError(
            message="Request failed",
            attempts=3,
        )

        assert str(error) == "Request failed"
        assert error.attempts == 3
        assert error.last_error is None
        assert error.status_code is None

    def test_with_all_attributes(self) -> None:
        """Test creating error with all attributes."""
        original_error = ValueError("Connection refused")

        error = RetryExhaustedError(
            message="Request failed after 4 attempts",
            attempts=4,
            last_error=original_error,
            status_code=503,
        )

        assert error.attempts == 4
        assert error.last_error is original_error
        assert error.status_code == 503


class TestFactoryFunctions:
    """Tests for retry config factory functions."""

    def test_default_retry_config(self) -> None:
        """Test default_retry_config factory."""
        config = default_retry_config()

        assert config.max_retries == 3
        assert config.jitter is True

    def test_aggressive_retry_config(self) -> None:
        """Test aggressive_retry_config factory."""
        config = aggressive_retry_config()

        assert config.max_retries == 5
        assert config.base_delay_ms == 200.0
        assert config.max_delay_ms == 60000.0

    def test_no_retry_config(self) -> None:
        """Test no_retry_config factory."""
        config = no_retry_config()

        assert config.max_retries == 0
        assert config.retry_on_status_codes == set()


class TestRequestWithRetry:
    """Tests for request_with_retry async function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Test successful request on first attempt."""
        # Mock httpx client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        config = RetryConfig()

        response = await request_with_retry(
            client=mock_client,
            method="GET",
            url="/test",
            config=config,
        )

        assert response == mock_response
        assert mock_client.request.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_503_then_success(self) -> None:
        """Test retry on 503, then success."""
        # First response: 503
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.request = MagicMock()

        # Second response: 200
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=[mock_response_503, mock_response_200]
        )

        config = RetryConfig(jitter=False, base_delay_ms=1)  # Fast for testing

        response = await request_with_retry(
            client=mock_client,
            method="GET",
            url="/test",
            config=config,
        )

        assert response == mock_response_200
        assert mock_client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        """Test RetryExhaustedError when all retries fail."""
        # All responses: 503
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.request = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        config = RetryConfig(max_retries=2, jitter=False, base_delay_ms=1)

        with pytest.raises(RetryExhaustedError) as exc_info:
            await request_with_retry(
                client=mock_client,
                method="GET",
                url="/test",
                config=config,
            )

        assert exc_info.value.attempts == 3  # Initial + 2 retries
        assert exc_info.value.status_code == 503
        assert mock_client.request.call_count == 3

    @pytest.mark.asyncio
    async def test_on_retry_callback_called(self) -> None:
        """Test that on_retry callback is invoked."""
        # First response: 503, second: 200
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.request = MagicMock()

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=[mock_response_503, mock_response_200]
        )

        callback_calls = []
        def on_retry(attempt, delay, error):
            callback_calls.append((attempt, delay, error))

        config = RetryConfig(jitter=False, base_delay_ms=1)

        await request_with_retry(
            client=mock_client,
            method="GET",
            url="/test",
            config=config,
            on_retry=on_retry,
        )

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 0  # First retry attempt

    @pytest.mark.asyncio
    async def test_non_retryable_status_raises_immediately(self) -> None:
        """Test that 400 errors raise immediately without retry."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.request = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=mock_response.request,
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        config = RetryConfig()

        with pytest.raises(httpx.HTTPStatusError):
            await request_with_retry(
                client=mock_client,
                method="POST",
                url="/test",
                config=config,
            )

        # Should only be called once (no retries)
        assert mock_client.request.call_count == 1


class TestRequestWithRetrySync:
    """Tests for request_with_retry_sync function."""

    def test_success_on_first_attempt(self) -> None:
        """Test successful sync request on first attempt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request = MagicMock(return_value=mock_response)

        config = RetryConfig()

        response = request_with_retry_sync(
            client=mock_client,
            method="GET",
            url="/test",
            config=config,
        )

        assert response == mock_response
        assert mock_client.request.call_count == 1

    def test_retry_on_503_then_success(self) -> None:
        """Test sync retry on 503, then success."""
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.request = MagicMock()

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request = MagicMock(
            side_effect=[mock_response_503, mock_response_200]
        )

        config = RetryConfig(jitter=False, base_delay_ms=1)

        response = request_with_retry_sync(
            client=mock_client,
            method="GET",
            url="/test",
            config=config,
        )

        assert response == mock_response_200
        assert mock_client.request.call_count == 2

    def test_retry_exhausted(self) -> None:
        """Test sync RetryExhaustedError when all retries fail."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.request = MagicMock()

        mock_client = MagicMock()
        mock_client.request = MagicMock(return_value=mock_response)

        config = RetryConfig(max_retries=1, jitter=False, base_delay_ms=1)

        with pytest.raises(RetryExhaustedError) as exc_info:
            request_with_retry_sync(
                client=mock_client,
                method="GET",
                url="/test",
                config=config,
            )

        assert exc_info.value.attempts == 2
        assert exc_info.value.status_code == 503
