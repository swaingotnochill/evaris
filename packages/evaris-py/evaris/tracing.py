"""
OpenTelemetry tracing integration for Evaris.

This module provides distributed tracing capabilities using OpenTelemetry,
allowing you to monitor evaluation performance, debug issues, and export
traces to observability platforms like Jaeger, Zipkin, or cloud providers.
"""

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

# Type stubs for when OpenTelemetry is not installed
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.trace import Span, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Dummy types for when OpenTelemetry is not installed
    Span = Any  # type: ignore
    TracerProvider = Any  # type: ignore

logger = logging.getLogger(__name__)


class SpanType:
    """Span types for distributed tracing.

    Defines semantic span types for different operations in evaris evaluation.
    """

    # Phase 1 span types
    EVALUATION = "evaluation"
    TEST_CASE = "test_case"
    METRIC = "metric"

    # Phase 3 span types (agent execution)
    AGENT_EXECUTION = "agent_execution"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_STEP = "agent_step"
    METRIC_REASONING = "metric_reasoning"
    EMBEDDING = "embedding"

    # Additional utility span types
    CACHE_LOOKUP = "cache_lookup"
    CACHE_STORE = "cache_store"


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available or tracing is disabled."""

    @contextmanager
    def start_span(self, name: str, attributes: Optional[dict[str, Any]] = None) -> Iterator[None]:
        """No-op context manager that yields None."""
        yield None

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """No-op event recording."""
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op attribute setting."""
        pass

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """No-op status setting."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op exception recording."""
        pass


class EvarisTracer:
    """
    OpenTelemetry tracer for Evaris evaluations.

    Provides distributed tracing with performance metrics, error tracking,
    and integration with observability platforms.

    Args:
        service_name: Name of the service (default: "evaris")
        exporter_type: Type of exporter ("otlp", "console", "none", or custom exporter)
        otlp_endpoint: OTLP endpoint URL (default: "http://localhost:4317")
        enabled: Enable/disable tracing (default: True, can be overridden by EVARIS_TRACING env var)

    Example:
        >>> tracer = EvarisTracer(service_name="my-app", exporter_type="console")
        >>> with tracer.start_span("evaluation") as span:
        ...     # Your evaluation code here
        ...     tracer.set_attribute("test_count", 10)
    """

    def __init__(
        self,
        service_name: str = "evaris",
        exporter_type: str = "otlp",
        otlp_endpoint: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        # Check if tracing is enabled via environment variable or parameter
        env_enabled = os.getenv("EVARIS_TRACING", "true").lower() in ("true", "1", "yes")
        self.enabled = enabled if enabled is not None else env_enabled

        if not self.enabled:
            logger.debug("Tracing is disabled")
            self._tracer: Any = NoOpTracer()
            self._current_span: Optional[Span] = None
            return

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry is not installed. Install with: pip install evaris[tracing]"
            )
            self._tracer = NoOpTracer()
            self._current_span = None
            return

        # Create resource with service name
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter
        exporter: Optional[SpanExporter] = None
        if exporter_type == "otlp":
            endpoint = otlp_endpoint or os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )
            try:
                exporter = OTLPSpanExporter(endpoint=endpoint)
                logger.info(f"OTLP exporter configured with endpoint: {endpoint}")
            except Exception as e:
                logger.warning(f"Failed to configure OTLP exporter: {e}. Falling back to console.")
                exporter = ConsoleSpanExporter()
        elif exporter_type == "console":
            exporter = ConsoleSpanExporter()
            logger.info("Console exporter configured")
        elif exporter_type == "none":
            exporter = None
            logger.info("No exporter configured (tracing spans will not be exported)")
        elif isinstance(exporter_type, SpanExporter):
            exporter = exporter_type
            logger.info(f"Custom exporter configured: {type(exporter).__name__}")
        else:
            logger.warning(f"Unknown exporter type: {exporter_type}. Using console exporter.")
            exporter = ConsoleSpanExporter()

        # Add span processor
        if exporter:
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer instance
        self._tracer = trace.get_tracer(__name__)

        logger.info(f"EvarisTracer initialized with service_name='{service_name}'")

    @contextmanager
    def start_span(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ) -> Iterator[Optional[Span]]:
        """
        Start a new span with optional attributes.

        Args:
            name: Name of the span
            attributes: Dictionary of attributes to attach to the span

        Yields:
            Span object or None if tracing is disabled

        Example:
            >>> with tracer.start_span("metric_execution", {"metric": "exact_match"}):
            ...     # Your metric code here
            ...     pass
        """
        if not self.enabled or isinstance(self._tracer, NoOpTracer):
            yield None
            return

        span = self._tracer.start_as_current_span(name)
        span_context = span.__enter__()

        try:
            # Set attributes
            if attributes:
                for key, value in attributes.items():
                    span_context.set_attribute(key, self._serialize_attribute(value))

            # Record start time as event
            span_context.add_event("span.start")

            self._current_span = span_context
            yield span_context

        except Exception as e:
            # Record exception
            if hasattr(span_context, "record_exception"):
                span_context.record_exception(e)
            if hasattr(span_context, "set_status"):
                span_context.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # Record end time as event
            if hasattr(span_context, "add_event"):
                span_context.add_event("span.end")
            span.__exit__(None, None, None)
            self._current_span = None

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value (will be serialized to string if needed)
        """
        if self._current_span and hasattr(self._current_span, "set_attribute"):
            self._current_span.set_attribute(key, self._serialize_attribute(value))

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """
        Add an event to the current span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        if self._current_span and hasattr(self._current_span, "add_event"):
            if attributes:
                serialized = {k: self._serialize_attribute(v) for k, v in attributes.items()}
                self._current_span.add_event(name, attributes=serialized)
            else:
                self._current_span.add_event(name)

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """
        Set the status of the current span.

        Args:
            status: Status ("ok", "error", "unset")
            description: Optional status description
        """
        if not self._current_span or not hasattr(self._current_span, "set_status"):
            return

        status_code = StatusCode.UNSET
        if status.lower() == "ok":
            status_code = StatusCode.OK
        elif status.lower() == "error":
            status_code = StatusCode.ERROR

        self._current_span.set_status(Status(status_code, description or ""))

    def record_exception(self, exception: Exception) -> None:
        """
        Record an exception in the current span.

        Args:
            exception: Exception to record
        """
        if self._current_span and hasattr(self._current_span, "record_exception"):
            self._current_span.record_exception(exception)

    @staticmethod
    def _serialize_attribute(value: Any) -> Any:
        """
        Serialize attribute values to OpenTelemetry-compatible types.

        OpenTelemetry attributes must be: str, bool, int, float, or sequences thereof.
        """
        if isinstance(value, (str, bool, int, float)):
            return value
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to strings if they contain non-primitive types
            if all(isinstance(v, (str, bool, int, float)) for v in value):
                return value
            return str(value)
        else:
            # Convert everything else to string
            return str(value)

    @contextmanager
    def measure_time(self, metric_name: str) -> Iterator[None]:
        """
        Context manager to measure execution time and add as span attribute.

        Args:
            metric_name: Name of the timing metric (will be suffixed with _ms)

        Example:
            >>> with tracer.measure_time("database_query"):
            ...     # Your code here
            ...     pass
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.set_attribute(f"{metric_name}_ms", round(elapsed_ms, 2))
            self.add_event(f"{metric_name}.completed", {"duration_ms": round(elapsed_ms, 2)})

    # Phase 3 helper methods for common attributes

    def set_reasoning(self, explanation: str, confidence: Optional[float] = None) -> None:
        """Set reasoning for metric evaluation.

        Args:
            explanation: Explanation of the score/decision
            confidence: Optional confidence score (0.0-1.0)
        """
        self.set_attribute("reasoning", explanation)
        if confidence is not None:
            self.set_attribute("reasoning_confidence", confidence)

    def set_cost(self, cost_usd: float) -> None:
        """Set cost for LLM call or operation.

        Args:
            cost_usd: Cost in USD
        """
        self.set_attribute("cost_usd", round(cost_usd, 6))

    def set_tokens(
        self,
        tokens: int,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Set token count for LLM call.

        Args:
            tokens: Total token count
            input_tokens: Optional input/prompt tokens
            output_tokens: Optional output/completion tokens
        """
        self.set_attribute("tokens_total", tokens)
        if input_tokens is not None:
            self.set_attribute("tokens_input", input_tokens)
        if output_tokens is not None:
            self.set_attribute("tokens_output", output_tokens)

    def set_cached(self, cached: bool, cache_key: Optional[str] = None) -> None:
        """Mark operation as cached.

        Args:
            cached: Whether this was a cache hit
            cache_key: Optional cache key for debugging
        """
        self.set_attribute("cached", cached)
        if cache_key:
            self.set_attribute("cache_key", cache_key[:100])  # Truncate long keys

    def set_latency(self, latency_ms: float) -> None:
        """Set operation latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.set_attribute("latency_ms", round(latency_ms, 2))

    def set_model(self, model: str, provider: Optional[str] = None) -> None:
        """Set model information for LLM calls.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus")
            provider: Optional provider (e.g., "openai", "anthropic")
        """
        self.set_attribute("model", model)
        if provider:
            self.set_attribute("provider", provider)


class DebugLogger:
    """
    Debug logger for detailed evaluation logging.

    Provides structured logging for debugging metric execution, LLM prompts,
    and intermediate values. Only logs when debug mode is enabled.

    Args:
        enabled: Enable/disable debug logging (default: False, can be
            overridden by EVARIS_DEBUG env var)
        logger_name: Logger name (default: "evaris.debug")

    Example:
        >>> debug = DebugLogger(enabled=True)
        >>> debug.log_prompt("llm_judge", prompt="What is 2+2?", model="gpt-4")
    """

    def __init__(self, enabled: Optional[bool] = None, logger_name: str = "evaris.debug"):
        # Check if debug mode is enabled via environment variable or parameter
        env_enabled = os.getenv("EVARIS_DEBUG", "false").lower() in ("true", "1", "yes")
        self.enabled = enabled if enabled is not None else env_enabled
        self.logger = logging.getLogger(logger_name)

        # Configure debug logger
        if self.enabled and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def log_prompt(self, metric_name: str, prompt: str, **kwargs: Any) -> None:
        """
        Log an LLM prompt with metadata.

        Args:
            metric_name: Name of the metric
            prompt: The prompt text
            **kwargs: Additional metadata (model, temperature, etc.)
        """
        if not self.enabled:
            return

        metadata = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"[{metric_name}] Prompt ({metadata}):\n{prompt}")

    def log_response(self, metric_name: str, response: str, **kwargs: Any) -> None:
        """
        Log an LLM response with metadata.

        Args:
            metric_name: Name of the metric
            response: The response text
            **kwargs: Additional metadata (tokens, cost, etc.)
        """
        if not self.enabled:
            return

        metadata = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"[{metric_name}] Response ({metadata}):\n{response}")

    def log_reasoning(self, metric_name: str, reasoning: str, score: float, **kwargs: Any) -> None:
        """
        Log reasoning from a metric evaluation.

        Args:
            metric_name: Name of the metric
            reasoning: The reasoning text
            score: The score assigned
            **kwargs: Additional metadata
        """
        if not self.enabled:
            return

        metadata = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"[{metric_name}] Reasoning (score={score}, {metadata}):\n{reasoning}")

    def log_intermediate(self, metric_name: str, step: str, **values: Any) -> None:
        """
        Log intermediate values during metric execution.

        Args:
            metric_name: Name of the metric
            step: Description of the step
            **values: Key-value pairs of intermediate values
        """
        if not self.enabled:
            return

        formatted_values = "\n".join(f"  {k}: {v}" for k, v in values.items())
        self.logger.debug(f"[{metric_name}] {step}:\n{formatted_values}")

    def log_error(self, metric_name: str, error: Exception, **context: Any) -> None:
        """
        Log an error with context.

        Args:
            metric_name: Name of the metric
            error: The exception
            **context: Additional context information
        """
        if not self.enabled:
            return

        context_str = "\n".join(f"  {k}: {v}" for k, v in context.items())
        self.logger.error(
            f"[{metric_name}] Error: {type(error).__name__}: {str(error)}\nContext:\n{context_str}",
            exc_info=True,
        )


# Global tracer and debug logger instances
_global_tracer: Optional[EvarisTracer] = None
_global_debug_logger: Optional[DebugLogger] = None


def get_tracer() -> EvarisTracer:
    """
    Get the global EvarisTracer instance.

    If not initialized, creates a default tracer.

    Returns:
        Global EvarisTracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = EvarisTracer()
    return _global_tracer


def get_debug_logger() -> DebugLogger:
    """
    Get the global DebugLogger instance.

    If not initialized, creates a default debug logger.

    Returns:
        Global DebugLogger instance
    """
    global _global_debug_logger
    if _global_debug_logger is None:
        _global_debug_logger = DebugLogger()
    return _global_debug_logger


def configure_tracing(
    service_name: str = "evaris",
    exporter_type: str = "otlp",
    otlp_endpoint: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> EvarisTracer:
    """
    Configure the global EvarisTracer instance.

    Args:
        service_name: Name of the service
        exporter_type: Type of exporter ("otlp", "console", "none")
        otlp_endpoint: OTLP endpoint URL
        enabled: Enable/disable tracing

    Returns:
        Configured EvarisTracer instance

    Example:
        >>> from evaris.tracing import configure_tracing
        >>> tracer = configure_tracing(service_name="my-app", exporter_type="console")
    """
    global _global_tracer
    _global_tracer = EvarisTracer(
        service_name=service_name,
        exporter_type=exporter_type,
        otlp_endpoint=otlp_endpoint,
        enabled=enabled,
    )
    return _global_tracer


def configure_debug_logging(enabled: Optional[bool] = None) -> DebugLogger:
    """
    Configure the global DebugLogger instance.

    Args:
        enabled: Enable/disable debug logging

    Returns:
        Configured DebugLogger instance

    Example:
        >>> from evaris.tracing import configure_debug_logging
        >>> debug = configure_debug_logging(enabled=True)
    """
    global _global_debug_logger
    _global_debug_logger = DebugLogger(enabled=enabled)
    return _global_debug_logger
