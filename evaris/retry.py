"""Retry logic with exponential backoff for HTTP requests.

This module provides configurable retry behavior for the Evaris cloud client,
including exponential backoff, jitter, and intelligent error classification.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Set, Tuple, Type

if TYPE_CHECKING:
    import httpx


# Default retryable status codes (transient server errors and rate limiting)
DEFAULT_RETRY_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay_ms: Initial delay in milliseconds (default: 100)
        max_delay_ms: Maximum delay cap in milliseconds (default: 30000)
        exponential_base: Multiplier for exponential growth (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
        retry_on_status_codes: HTTP status codes that trigger retry
        retry_on_exceptions: Exception types that trigger retry

    Example:
        >>> config = RetryConfig(max_retries=5, base_delay_ms=200)
        >>> client = EvarisClient(api_key="...", retry_config=config)
    """

    max_retries: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 30000.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_status_codes: Set[int] = field(
        default_factory=lambda: DEFAULT_RETRY_STATUS_CODES.copy()
    )
    # Exception types are defined as strings to avoid import issues
    # Actual types are resolved at runtime
    retry_on_exception_names: Tuple[str, ...] = (
        "TimeoutException",
        "ConnectError",
        "RemoteProtocolError",
    )

    def should_retry_status(self, status_code: int) -> bool:
        """Check if a status code should trigger a retry."""
        return status_code in self.retry_on_status_codes

    def should_retry_exception(self, exc: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        exc_name = type(exc).__name__
        return exc_name in self.retry_on_exception_names


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and optional jitter.

    The delay follows the formula:
        delay = min(base_delay * (exponential_base ^ attempt), max_delay)

    With jitter enabled, the final delay is multiplied by a random
    factor between 0.5 and 1.0 to prevent thundering herd effects.

    Args:
        attempt: Zero-indexed retry attempt number
        config: Retry configuration

    Returns:
        Delay in seconds (not milliseconds)

    Example:
        >>> config = RetryConfig(base_delay_ms=100, exponential_base=2)
        >>> calculate_backoff_delay(0, config)  # ~0.1s
        >>> calculate_backoff_delay(1, config)  # ~0.2s
        >>> calculate_backoff_delay(2, config)  # ~0.4s
    """
    delay_ms = min(
        config.base_delay_ms * (config.exponential_base**attempt),
        config.max_delay_ms,
    )

    if config.jitter:
        # Add jitter: multiply by random factor between 0.5 and 1.0
        # This helps prevent synchronized retry storms
        jitter_factor = 0.5 + (random.random() * 0.5)
        delay_ms *= jitter_factor

    return delay_ms / 1000.0  # Convert to seconds


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        attempts: Number of attempts made
        last_error: The last error that occurred
        status_code: HTTP status code if available
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.status_code = status_code


async def request_with_retry(
    client: "httpx.AsyncClient",
    method: str,
    url: str,
    config: RetryConfig,
    on_retry: Callable[[int, float, Exception | None], None] | None = None,
    **kwargs,
) -> "httpx.Response":
    """Make HTTP request with exponential backoff retry.

    This function wraps httpx requests with intelligent retry logic,
    automatically retrying on transient failures with exponential backoff.

    Args:
        client: The httpx AsyncClient to use
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        config: Retry configuration
        on_retry: Optional callback called before each retry with
                  (attempt, delay, error) arguments
        **kwargs: Additional arguments passed to client.request()

    Returns:
        The successful httpx.Response

    Raises:
        RetryExhaustedError: When all retries are exhausted
        httpx.HTTPStatusError: For non-retryable HTTP errors (4xx except 429)

    Example:
        >>> async with httpx.AsyncClient() as client:
        ...     response = await request_with_retry(
        ...         client, "POST", "/api/assess",
        ...         config=RetryConfig(),
        ...         json={"test_cases": [...]}
        ...     )
    """
    import httpx

    attempt = 0
    last_exception: Exception | None = None
    last_status_code: int | None = None

    while attempt <= config.max_retries:
        try:
            response = await client.request(method, url, **kwargs)

            # Check if status code is retryable
            if config.should_retry_status(response.status_code):
                last_status_code = response.status_code
                last_exception = httpx.HTTPStatusError(
                    f"Server returned {response.status_code}",
                    request=response.request,
                    response=response,
                )
                # Fall through to retry logic
            else:
                # Success or non-retryable error
                response.raise_for_status()
                return response

        except httpx.HTTPStatusError as e:
            # Non-retryable HTTP error (already raised above for retryable ones)
            if not config.should_retry_status(e.response.status_code):
                raise
            last_exception = e
            last_status_code = e.response.status_code

        except Exception as e:
            # Check if exception type is retryable
            if not config.should_retry_exception(e):
                raise
            last_exception = e

        # Calculate delay and retry if attempts remain
        if attempt < config.max_retries:
            delay = calculate_backoff_delay(attempt, config)

            # Call optional retry callback
            if on_retry:
                on_retry(attempt, delay, last_exception)

            await asyncio.sleep(delay)

        attempt += 1

    # All retries exhausted
    raise RetryExhaustedError(
        message=f"Request failed after {config.max_retries + 1} attempts",
        attempts=config.max_retries + 1,
        last_error=last_exception,
        status_code=last_status_code,
    )


def request_with_retry_sync(
    client: "httpx.Client",
    method: str,
    url: str,
    config: RetryConfig,
    on_retry: Callable[[int, float, Exception | None], None] | None = None,
    **kwargs,
) -> "httpx.Response":
    """Synchronous version of request_with_retry.

    Same behavior as the async version but for synchronous httpx.Client.

    Args:
        client: The httpx Client to use (not AsyncClient)
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        config: Retry configuration
        on_retry: Optional callback called before each retry
        **kwargs: Additional arguments passed to client.request()

    Returns:
        The successful httpx.Response

    Raises:
        RetryExhaustedError: When all retries are exhausted
        httpx.HTTPStatusError: For non-retryable HTTP errors
    """
    import time

    import httpx

    attempt = 0
    last_exception: Exception | None = None
    last_status_code: int | None = None

    while attempt <= config.max_retries:
        try:
            response = client.request(method, url, **kwargs)

            if config.should_retry_status(response.status_code):
                last_status_code = response.status_code
                last_exception = httpx.HTTPStatusError(
                    f"Server returned {response.status_code}",
                    request=response.request,
                    response=response,
                )
            else:
                response.raise_for_status()
                return response

        except httpx.HTTPStatusError as e:
            if not config.should_retry_status(e.response.status_code):
                raise
            last_exception = e
            last_status_code = e.response.status_code

        except Exception as e:
            if not config.should_retry_exception(e):
                raise
            last_exception = e

        if attempt < config.max_retries:
            delay = calculate_backoff_delay(attempt, config)
            if on_retry:
                on_retry(attempt, delay, last_exception)
            time.sleep(delay)

        attempt += 1

    raise RetryExhaustedError(
        message=f"Request failed after {config.max_retries + 1} attempts",
        attempts=config.max_retries + 1,
        last_error=last_exception,
        status_code=last_status_code,
    )


# Convenience factory functions for common configurations
def default_retry_config() -> RetryConfig:
    """Create a default retry configuration suitable for most use cases."""
    return RetryConfig()


def aggressive_retry_config() -> RetryConfig:
    """Create a more aggressive retry configuration for critical requests.

    Uses more retries with longer delays for important operations.
    """
    return RetryConfig(
        max_retries=5,
        base_delay_ms=200.0,
        max_delay_ms=60000.0,
        exponential_base=2.0,
        jitter=True,
    )


def no_retry_config() -> RetryConfig:
    """Create a configuration that disables retries.

    Useful when you want explicit control over retry behavior.
    """
    return RetryConfig(
        max_retries=0,
        retry_on_status_codes=set(),
    )
