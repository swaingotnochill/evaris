"""Evaris Cloud Client - SDK for sending assessments to Evaris Cloud.

This module provides a thin HTTP client for interacting with the Evaris
cloud service. It handles:
- Authentication via API key
- Sending test cases for assessment
- Retrieving assessment results
- Sending traces and logs
- Automatic retry with exponential backoff
- Both async and sync interfaces

Usage:
    from evaris import EvarisClient

    # Async usage (recommended)
    async with EvarisClient(api_key="your-api-key") as client:
        result = await client.assess(
            name="my-assessment-run",
            test_cases=[
                TestCase(
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                )
            ],
            metrics=["faithfulness", "toxicity"],
        )

    # Sync usage
    client = EvarisClient(api_key="your-api-key")
    result = client.assess_sync(
        name="my-assessment-run",
        test_cases=[...],
        metrics=["faithfulness"],
    )

    print(result.summary.accuracy)
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import datetime
from typing import Any, Callable, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from evaris.retry import (
    RetryConfig,
    RetryExhaustedError,
    default_retry_config,
    request_with_retry,
    request_with_retry_sync,
)
from evaris.types import TestCase


# ==============================================================================
# Response Models
# ==============================================================================


class MetricScore:
    """Score for a single metric on a single test case."""

    def __init__(
        self,
        name: str,
        score: float,
        passed: bool,
        reasoning: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.score = score
        self.passed = passed
        self.reasoning = reasoning
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricScore:
        return cls(
            name=data["name"],
            score=data["score"],
            passed=data["passed"],
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
        )


class TestResult:
    """Result for a single test case."""

    def __init__(
        self,
        input: Any,
        expected: Any,
        actual_output: Any,
        scores: list[MetricScore],
        passed: bool,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.input = input
        self.expected = expected
        self.actual_output = actual_output
        self.scores = scores
        self.passed = passed
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        return cls(
            input=data["input"],
            expected=data.get("expected"),
            actual_output=data["actual_output"],
            scores=[MetricScore.from_dict(s) for s in data["scores"]],
            passed=data["passed"],
            metadata=data.get("metadata", {}),
        )


class AssessmentSummary:
    """Summary statistics for an assessment run."""

    def __init__(
        self,
        total: int,
        passed: int,
        failed: int,
        accuracy: float,
        metrics: Optional[dict[str, dict[str, float]]] = None,
    ):
        self.total = total
        self.passed = passed
        self.failed = failed
        self.accuracy = accuracy
        self.metrics = metrics or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentSummary:
        return cls(
            total=data["total"],
            passed=data["passed"],
            failed=data["failed"],
            accuracy=data["accuracy"],
            metrics=data.get("metrics", {}),
        )


class AssessmentResult:
    """Result of an assessment run."""

    def __init__(
        self,
        assessment_id: str,
        project_id: str,
        name: str,
        status: str,
        summary: Optional[AssessmentSummary],
        results: Optional[list[TestResult]],
        created_at: datetime,
        completed_at: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.assessment_id = assessment_id
        self.project_id = project_id
        self.name = name
        self.status = status
        self.summary = summary
        self.results = results
        self.created_at = created_at
        self.completed_at = completed_at
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentResult:
        # Handle various ID field names from server response
        # Priority: assessment_id > eval_id > run_id > id
        assessment_id = (
            data.get("assessment_id")
            or data.get("eval_id")
            or data.get("run_id")
            or data.get("id", "")
        )

        # Handle summary: either nested object or flat fields (from list response)
        summary = None
        if data.get("summary"):
            summary = AssessmentSummary.from_dict(data["summary"])
        elif "total" in data or "passed" in data or "failed" in data:
            # List response has flat fields
            summary = AssessmentSummary(
                total=data.get("total") or 0,
                passed=data.get("passed") or 0,
                failed=data.get("failed") or 0,
                accuracy=data.get("accuracy") or 0.0,
                metrics={},
            )

        # Parse datetime - handle both string and datetime objects
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif created_at is None:
            created_at = datetime.now()

        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

        return cls(
            assessment_id=assessment_id,
            project_id=data.get("project_id", ""),
            name=data.get("name", ""),
            status=data.get("status", "PENDING"),
            summary=summary,
            results=[TestResult.from_dict(r) for r in data["results"]] if data.get("results") else None,
            created_at=created_at,
            completed_at=completed_at,
            metadata=data.get("metadata", {}),
        )


class TraceResult:
    """Result of storing a trace."""

    def __init__(
        self,
        trace_id: str,
        project_id: str,
        name: str,
        span_count: int,
        duration_ms: Optional[float],
        created_at: datetime,
    ):
        self.trace_id = trace_id
        self.project_id = project_id
        self.name = name
        self.span_count = span_count
        self.duration_ms = duration_ms
        self.created_at = created_at

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceResult:
        return cls(
            trace_id=data["trace_id"],
            project_id=data["project_id"],
            name=data["name"],
            span_count=data["span_count"],
            duration_ms=data.get("duration_ms"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        )


class LogResult:
    """Result of storing a log entry."""

    def __init__(
        self,
        log_id: str,
        project_id: str,
        level: str,
        created_at: datetime,
    ):
        self.log_id = log_id
        self.project_id = project_id
        self.level = level
        self.created_at = created_at

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogResult:
        return cls(
            log_id=data["log_id"],
            project_id=data["project_id"],
            level=data["level"],
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        )


# ==============================================================================
# Span Input
# ==============================================================================


class Span:
    """A span within a trace."""

    def __init__(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_ms: Optional[float] = None,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        children: Optional[list[Span]] = None,
    ):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.duration_ms = duration_ms
        self.input = input
        self.output = output
        self.metadata = metadata or {}
        self.children = children or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


# ==============================================================================
# Client
# ==============================================================================


class EvarisClient:
    """Client for interacting with Evaris Cloud.

    This is a thin HTTP client that sends assessment requests to the
    Evaris cloud service. All heavy computation (LLM judge calls, etc.)
    happens server-side.

    Features:
    - Automatic retry with exponential backoff for transient failures
    - Both async and sync interfaces
    - Connection pooling via httpx
    - Configurable timeouts and retry policies

    Args:
        api_key: Your Evaris API key (from evaris.ai dashboard)
        base_url: Evaris API URL (default: https://api.evaris.ai)
        timeout: Request timeout in seconds (default: 300 for long assessments)
        retry_config: Optional retry configuration. If None, uses default
                      retry policy with 3 retries and exponential backoff.
        on_retry: Optional callback called before each retry attempt.
                  Receives (attempt, delay, error) arguments.

    Example:
        # Async usage with context manager (recommended)
        async with EvarisClient(api_key="ev_xxx") as client:
            result = await client.assess(
                name="my-assessment",
                test_cases=[...],
                metrics=["faithfulness"],
            )

        # Sync usage
        client = EvarisClient(api_key="ev_xxx")
        result = client.assess_sync(
            name="my-assessment",
            test_cases=[...],
            metrics=["faithfulness"],
        )

        # Custom retry configuration
        from evaris.retry import RetryConfig
        client = EvarisClient(
            api_key="ev_xxx",
            retry_config=RetryConfig(max_retries=5, base_delay_ms=200),
        )
    """

    DEFAULT_BASE_URL = "https://api.evaris.ai"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        retry_config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, float, Exception | None], None]] = None,
        internal_token: Optional[str] = None,
    ):
        """Initialize EvarisClient.

        Args:
            api_key: Your Evaris API key (from evaris.ai dashboard)
            base_url: Evaris API URL (default: https://api.evaris.ai)
            timeout: Request timeout in seconds (default: 300)
            retry_config: Optional retry configuration
            on_retry: Optional callback called before each retry attempt
            internal_token: Optional JWT token for internal/E2E testing.
                When provided, uses X-Context-Token header for direct
                server authentication, bypassing the API gateway.
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for EvarisClient. "
                "Install it with: pip install httpx"
            )

        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.retry_config = retry_config or default_retry_config()
        self.on_retry = on_retry
        self.internal_token = internal_token
        self._async_client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests.

        Uses X-Context-Token for internal JWT auth when internal_token is set,
        otherwise uses standard Authorization Bearer for API key auth.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "evaris-python/0.1.0",
        }

        if self.internal_token:
            # Internal JWT auth for E2E testing / direct server calls
            headers["X-Context-Token"] = self.internal_token
        else:
            # Standard API key auth via API gateway
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def __aenter__(self) -> EvarisClient:
        """Enter async context manager."""
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> EvarisClient:
        """Enter sync context manager."""
        self._sync_client = httpx.Client(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit sync context manager."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get the async HTTP client, creating one if needed."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        """Get the sync HTTP client, creating one if needed."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._sync_client

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close_sync(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    def _run_sync(self, coro):
        """Safely run async coroutine from sync context.

        This helper method properly handles running async code from
        synchronous contexts, avoiding common pitfalls with event loops.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            RuntimeError: If called from within an async context
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync methods from async context. "
                "Use the async version instead (e.g., assess() instead of assess_sync())."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise

        # Safe to create a new event loop
        return asyncio.run(coro)

    # ==========================================================================
    # Assess (run metrics)
    # ==========================================================================

    async def assess(
        self,
        name: str,
        test_cases: list[TestCase],
        metrics: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> AssessmentResult:
        """Run an assessment on the Evaris cloud.

        Sends test cases to Evaris for assessment. The LLM judge runs
        server-side using Evaris's API keys.

        Args:
            name: Name for this assessment run
            test_cases: List of test cases to assess
            metrics: List of metric names (e.g., ["faithfulness", "toxicity"])
            metadata: Optional metadata to attach to the run

        Returns:
            AssessmentResult with scores, summary, and detailed results

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_async_client()

        # Convert test cases to dicts
        test_case_dicts = []
        for tc in test_cases:
            test_case_dicts.append({
                "input": tc.input,
                "expected": tc.expected,
                "actual_output": tc.actual_output,
                "metadata": tc.metadata or {},
            })

        payload = {
            "name": name,
            "test_cases": test_case_dicts,
            "metrics": metrics,
            "metadata": metadata or {},
        }

        response = await request_with_retry(
            client=client,
            method="POST",
            url="/internal/evaluate",
            config=self.retry_config,
            on_retry=self.on_retry,
            json=payload,
        )

        return AssessmentResult.from_dict(response.json())

    def assess_sync(
        self,
        name: str,
        test_cases: list[TestCase],
        metrics: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> AssessmentResult:
        """Synchronous version of assess().

        Runs an assessment using a synchronous HTTP client with retry support.

        Args:
            name: Name for this assessment run
            test_cases: List of test cases to assess
            metrics: List of metric names (e.g., ["faithfulness", "toxicity"])
            metadata: Optional metadata to attach to the run

        Returns:
            AssessmentResult with scores, summary, and detailed results

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_sync_client()

        # Convert test cases to dicts
        test_case_dicts = []
        for tc in test_cases:
            test_case_dicts.append({
                "input": tc.input,
                "expected": tc.expected,
                "actual_output": tc.actual_output,
                "metadata": tc.metadata or {},
            })

        payload = {
            "name": name,
            "test_cases": test_case_dicts,
            "metrics": metrics,
            "metadata": metadata or {},
        }

        response = request_with_retry_sync(
            client=client,
            method="POST",
            url="/internal/evaluate",
            config=self.retry_config,
            on_retry=self.on_retry,
            json=payload,
        )

        return AssessmentResult.from_dict(response.json())

    # ==========================================================================
    # Trace
    # ==========================================================================

    async def trace(
        self,
        name: str,
        spans: list[Span],
        duration_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceResult:
        """Store a trace with spans.

        Args:
            name: Name for this trace
            spans: List of spans in the trace
            duration_ms: Total trace duration (calculated from spans if not provided)
            metadata: Optional metadata

        Returns:
            TraceResult with trace ID

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_async_client()

        payload = {
            "name": name,
            "spans": [s.to_dict() for s in spans],
            "duration_ms": duration_ms,
            "metadata": metadata or {},
        }

        response = await request_with_retry(
            client=client,
            method="POST",
            url="/internal/trace",
            config=self.retry_config,
            on_retry=self.on_retry,
            json=payload,
        )

        return TraceResult.from_dict(response.json())

    def trace_sync(
        self,
        name: str,
        spans: list[Span],
        duration_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceResult:
        """Synchronous version of trace().

        Args:
            name: Name for this trace
            spans: List of spans in the trace
            duration_ms: Total trace duration (calculated from spans if not provided)
            metadata: Optional metadata

        Returns:
            TraceResult with trace ID

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_sync_client()

        payload = {
            "name": name,
            "spans": [s.to_dict() for s in spans],
            "duration_ms": duration_ms,
            "metadata": metadata or {},
        }

        response = request_with_retry_sync(
            client=client,
            method="POST",
            url="/internal/trace",
            config=self.retry_config,
            on_retry=self.on_retry,
            json=payload,
        )

        return TraceResult.from_dict(response.json())

    # ==========================================================================
    # Log
    # ==========================================================================

    async def log(
        self,
        message: str,
        level: str = "info",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> LogResult:
        """Store a log entry.

        Args:
            message: Log message
            level: Log level (debug, info, warning, error)
            metadata: Optional metadata
            timestamp: Log timestamp (defaults to now)

        Returns:
            LogResult with log ID

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_async_client()

        payload = {
            "message": message,
            "level": level,
            "metadata": metadata or {},
            "timestamp": timestamp.isoformat() if timestamp else None,
        }

        response = await request_with_retry(
            client=client,
            method="POST",
            url="/internal/log",
            config=self.retry_config,
            on_retry=self.on_retry,
            json=payload,
        )

        return LogResult.from_dict(response.json())

    def log_sync(
        self,
        message: str,
        level: str = "info",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> LogResult:
        """Synchronous version of log().

        Args:
            message: Log message
            level: Log level (debug, info, warning, error)
            metadata: Optional metadata
            timestamp: Log timestamp (defaults to now)

        Returns:
            LogResult with log ID

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_sync_client()

        payload = {
            "message": message,
            "level": level,
            "metadata": metadata or {},
            "timestamp": timestamp.isoformat() if timestamp else None,
        }

        response = request_with_retry_sync(
            client=client,
            method="POST",
            url="/internal/log",
            config=self.retry_config,
            on_retry=self.on_retry,
            json=payload,
        )

        return LogResult.from_dict(response.json())

    # ==========================================================================
    # Get Results
    # ==========================================================================

    async def get_assessment(self, assessment_id: str) -> AssessmentResult:
        """Get an assessment by ID.

        Args:
            assessment_id: The assessment ID

        Returns:
            AssessmentResult with full details

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors (including 404)
        """
        client = self._get_async_client()

        response = await request_with_retry(
            client=client,
            method="GET",
            url=f"/internal/evaluations/{assessment_id}",
            config=self.retry_config,
            on_retry=self.on_retry,
        )

        return AssessmentResult.from_dict(response.json())

    def get_assessment_sync(self, assessment_id: str) -> AssessmentResult:
        """Synchronous version of get_assessment().

        Args:
            assessment_id: The assessment ID

        Returns:
            AssessmentResult with full details

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors (including 404)
        """
        client = self._get_sync_client()

        response = request_with_retry_sync(
            client=client,
            method="GET",
            url=f"/internal/evaluations/{assessment_id}",
            config=self.retry_config,
            on_retry=self.on_retry,
        )

        return AssessmentResult.from_dict(response.json())

    async def list_assessments(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AssessmentResult]:
        """List recent assessments.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of AssessmentResult objects

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_async_client()

        response = await request_with_retry(
            client=client,
            method="GET",
            url="/internal/evaluations",
            config=self.retry_config,
            on_retry=self.on_retry,
            params={"limit": limit, "offset": offset},
        )

        data = response.json()
        # Server returns "evaluations" field, not "items"
        return [AssessmentResult.from_dict(item) for item in data.get("evaluations", [])]

    def list_assessments_sync(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AssessmentResult]:
        """Synchronous version of list_assessments().

        Args:
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of AssessmentResult objects

        Raises:
            RetryExhaustedError: If all retry attempts fail
            httpx.HTTPStatusError: For non-retryable HTTP errors
        """
        client = self._get_sync_client()

        response = request_with_retry_sync(
            client=client,
            method="GET",
            url="/internal/evaluations",
            config=self.retry_config,
            on_retry=self.on_retry,
            params={"limit": limit, "offset": offset},
        )

        data = response.json()
        # Server returns "evaluations" field, not "items"
        return [AssessmentResult.from_dict(item) for item in data.get("evaluations", [])]


# ==============================================================================
# Convenience function
# ==============================================================================


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
) -> EvarisClient:
    """Get an Evaris client instance.

    If api_key is not provided, it will be read from the EVARIS_API_KEY
    environment variable.

    Args:
        api_key: Your Evaris API key
        base_url: Optional custom API URL
        retry_config: Optional retry configuration

    Returns:
        EvarisClient instance

    Raises:
        ValueError: If no API key is provided or found in environment
    """
    import os

    key = api_key or os.environ.get("EVARIS_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Either pass api_key or set EVARIS_API_KEY "
            "environment variable."
        )

    return EvarisClient(api_key=key, base_url=base_url, retry_config=retry_config)
