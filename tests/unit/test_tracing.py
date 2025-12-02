"""Unit tests for tracing module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from evaris.tracing import (
    DebugLogger,
    EvarisTracer,
    NoOpTracer,
    configure_debug_logging,
    configure_tracing,
    get_debug_logger,
    get_tracer,
)


class TestNoOpTracer:
    """Tests for NoOpTracer."""

    def setup_method(self) -> None:
        """Setup for each test."""
        self.tracer = NoOpTracer()

    def test_start_span_yields_none(self) -> None:
        """Test that start_span yields None."""
        with self.tracer.start_span("test_span") as span:
            assert span is None

    def test_start_span_with_attributes(self) -> None:
        """Test start_span with attributes (should be no-op)."""
        with self.tracer.start_span("test_span", attributes={"key": "value"}) as span:
            assert span is None

    def test_add_event_no_op(self) -> None:
        """Test add_event is a no-op."""
        self.tracer.add_event("test_event")  # Should not raise
        self.tracer.add_event("test_event", {"key": "value"})  # Should not raise

    def test_set_attribute_no_op(self) -> None:
        """Test set_attribute is a no-op."""
        self.tracer.set_attribute("key", "value")  # Should not raise

    def test_set_status_no_op(self) -> None:
        """Test set_status is a no-op."""
        self.tracer.set_status("ok")  # Should not raise
        self.tracer.set_status("error", "description")  # Should not raise

    def test_record_exception_no_op(self) -> None:
        """Test record_exception is a no-op."""
        exc = ValueError("test error")
        self.tracer.record_exception(exc)  # Should not raise


class TestEvarisTracer:
    """Tests for EvarisTracer."""

    def test_init_disabled_via_parameter(self) -> None:
        """Test tracer disabled via parameter."""
        tracer = EvarisTracer(enabled=False)
        assert not tracer.enabled
        assert isinstance(tracer._tracer, NoOpTracer)

    def test_init_disabled_via_env_var(self) -> None:
        """Test tracer disabled via environment variable."""
        with patch.dict(os.environ, {"EVARIS_TRACING": "false"}):
            tracer = EvarisTracer()
            assert not tracer.enabled
            assert isinstance(tracer._tracer, NoOpTracer)

    def test_init_without_opentelemetry(self) -> None:
        """Test initialization when OpenTelemetry not available."""
        with patch("evaris.tracing.OTEL_AVAILABLE", False):
            tracer = EvarisTracer()
            assert isinstance(tracer._tracer, NoOpTracer)

    def test_start_span_when_disabled(self) -> None:
        """Test start_span when tracing is disabled."""
        tracer = EvarisTracer(enabled=False)
        with tracer.start_span("test_span") as span:
            assert span is None

    def test_start_span_with_attributes_when_disabled(self) -> None:
        """Test start_span with attributes when disabled."""
        tracer = EvarisTracer(enabled=False)
        with tracer.start_span("test_span", attributes={"key": "value"}) as span:
            assert span is None

    def test_set_attribute_when_disabled(self) -> None:
        """Test set_attribute when tracing is disabled."""
        tracer = EvarisTracer(enabled=False)
        # Should not raise even without current span
        tracer.set_attribute("key", "value")

    def test_add_event_when_disabled(self) -> None:
        """Test add_event when tracing is disabled."""
        tracer = EvarisTracer(enabled=False)
        # Should not raise
        tracer.add_event("test_event", {"key": "value"})

    def test_set_status_when_disabled(self) -> None:
        """Test set_status when tracing is disabled."""
        tracer = EvarisTracer(enabled=False)
        # Should not raise
        tracer.set_status("ok", "description")

    def test_record_exception_when_disabled(self) -> None:
        """Test record_exception when tracing is disabled."""
        tracer = EvarisTracer(enabled=False)
        # Should not raise
        tracer.record_exception(ValueError("test"))

    def test_measure_time_when_disabled(self) -> None:
        """Test measure_time when tracing is disabled."""
        tracer = EvarisTracer(enabled=False)
        with tracer.measure_time("operation"):
            pass  # Should not raise

    def test_serialize_attribute_string(self) -> None:
        """Test attribute serialization for strings."""
        result = EvarisTracer._serialize_attribute("test")
        assert result == "test"

    def test_serialize_attribute_int(self) -> None:
        """Test attribute serialization for integers."""
        result = EvarisTracer._serialize_attribute(42)
        assert result == 42

    def test_serialize_attribute_float(self) -> None:
        """Test attribute serialization for floats."""
        result = EvarisTracer._serialize_attribute(3.14)
        assert result == 3.14

    def test_serialize_attribute_bool(self) -> None:
        """Test attribute serialization for booleans."""
        result = EvarisTracer._serialize_attribute(True)
        assert result is True

    def test_serialize_attribute_list_of_primitives(self) -> None:
        """Test attribute serialization for lists of primitives."""
        result = EvarisTracer._serialize_attribute([1, 2, 3])
        assert result == [1, 2, 3]

    def test_serialize_attribute_list_of_strings(self) -> None:
        """Test attribute serialization for lists of strings."""
        result = EvarisTracer._serialize_attribute(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_serialize_attribute_complex_object(self) -> None:
        """Test attribute serialization for complex objects."""
        result = EvarisTracer._serialize_attribute({"key": "value"})
        assert isinstance(result, str)
        assert "key" in result

    def test_serialize_attribute_list_with_complex(self) -> None:
        """Test attribute serialization for lists with complex objects."""
        result = EvarisTracer._serialize_attribute([1, {"key": "value"}, 3])
        assert isinstance(result, str)


class TestDebugLogger:
    """Tests for DebugLogger."""

    def test_init_disabled_by_default(self) -> None:
        """Test debug logger disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            logger = DebugLogger()
            assert not logger.enabled

    def test_init_enabled_via_parameter(self) -> None:
        """Test debug logger enabled via parameter."""
        logger = DebugLogger(enabled=True)
        assert logger.enabled

    def test_init_enabled_via_env_var(self) -> None:
        """Test debug logger enabled via environment variable."""
        with patch.dict(os.environ, {"EVARIS_DEBUG": "true"}):
            logger = DebugLogger()
            assert logger.enabled

    def test_init_env_var_variants(self) -> None:
        """Test environment variable accepts various true values."""
        for value in ["true", "True", "1", "yes", "YES"]:
            with patch.dict(os.environ, {"EVARIS_DEBUG": value}):
                logger = DebugLogger()
                assert logger.enabled, f"Failed for value: {value}"

    def test_log_prompt_when_disabled(self) -> None:
        """Test log_prompt does nothing when disabled."""
        logger = DebugLogger(enabled=False)
        # Should not raise
        logger.log_prompt("test_metric", "Test prompt", model="gpt-4")

    def test_log_prompt_when_enabled(self) -> None:
        """Test log_prompt logs when enabled."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = DebugLogger(enabled=True)
            logger.log_prompt("test_metric", "Test prompt", model="gpt-4", temperature=0.7)

            # Verify logger.debug was called
            mock_logger_instance.debug.assert_called_once()
            call_args = mock_logger_instance.debug.call_args[0][0]
            assert "test_metric" in call_args
            assert "Test prompt" in call_args
            assert "model=gpt-4" in call_args

    def test_log_response_when_disabled(self) -> None:
        """Test log_response does nothing when disabled."""
        logger = DebugLogger(enabled=False)
        # Should not raise
        logger.log_response("test_metric", "Test response", tokens=100)

    def test_log_response_when_enabled(self) -> None:
        """Test log_response logs when enabled."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = DebugLogger(enabled=True)
            logger.log_response("test_metric", "Test response", tokens=100)

            # Verify logger.debug was called
            mock_logger_instance.debug.assert_called_once()
            call_args = mock_logger_instance.debug.call_args[0][0]
            assert "test_metric" in call_args
            assert "Test response" in call_args

    def test_log_reasoning_when_disabled(self) -> None:
        """Test log_reasoning does nothing when disabled."""
        logger = DebugLogger(enabled=False)
        # Should not raise
        logger.log_reasoning("test_metric", "Reasoning text", score=0.85)

    def test_log_reasoning_when_enabled(self) -> None:
        """Test log_reasoning logs when enabled."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = DebugLogger(enabled=True)
            logger.log_reasoning("test_metric", "Reasoning text", score=0.85, confidence="high")

            # Verify logger.debug was called
            mock_logger_instance.debug.assert_called_once()
            call_args = mock_logger_instance.debug.call_args[0][0]
            assert "test_metric" in call_args
            assert "Reasoning text" in call_args
            assert "score=0.85" in call_args

    def test_log_intermediate_when_disabled(self) -> None:
        """Test log_intermediate does nothing when disabled."""
        logger = DebugLogger(enabled=False)
        # Should not raise
        logger.log_intermediate("test_metric", "Step 1", value1=42)

    def test_log_intermediate_when_enabled(self) -> None:
        """Test log_intermediate logs when enabled."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = DebugLogger(enabled=True)
            logger.log_intermediate("test_metric", "Step 1", value1=42, value2="test")

            # Verify logger.debug was called
            mock_logger_instance.debug.assert_called_once()
            call_args = mock_logger_instance.debug.call_args[0][0]
            assert "test_metric" in call_args
            assert "Step 1" in call_args
            assert "value1: 42" in call_args
            assert "value2: test" in call_args

    def test_log_error_when_disabled(self) -> None:
        """Test log_error does nothing when disabled."""
        logger = DebugLogger(enabled=False)
        error = ValueError("test error")
        # Should not raise
        logger.log_error("test_metric", error, context_key="context_value")

    def test_log_error_when_enabled(self) -> None:
        """Test log_error logs when enabled."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = DebugLogger(enabled=True)
            error = ValueError("test error")
            logger.log_error("test_metric", error, context_key="context_value")

            # Verify logger.error was called
            mock_logger_instance.error.assert_called_once()
            call_args = mock_logger_instance.error.call_args[0][0]
            assert "test_metric" in call_args
            assert "ValueError" in call_args
            assert "test error" in call_args


class TestGlobalTracerAndLogger:
    """Tests for global tracer and logger functions."""

    def test_get_tracer_creates_instance(self) -> None:
        """Test get_tracer creates a global instance."""
        # Reset global state
        import evaris.tracing

        evaris.tracing._global_tracer = None

        tracer1 = get_tracer()
        tracer2 = get_tracer()

        # Should return the same instance
        assert tracer1 is tracer2

    def test_get_debug_logger_creates_instance(self) -> None:
        """Test get_debug_logger creates a global instance."""
        # Reset global state
        import evaris.tracing

        evaris.tracing._global_debug_logger = None

        logger1 = get_debug_logger()
        logger2 = get_debug_logger()

        # Should return the same instance
        assert logger1 is logger2

    def test_configure_tracing(self) -> None:
        """Test configure_tracing sets global tracer."""
        tracer = configure_tracing(service_name="test-service", exporter_type="none", enabled=False)

        assert isinstance(tracer, EvarisTracer)
        # Subsequent get_tracer() should return the same instance
        assert get_tracer() is tracer

    def test_configure_debug_logging(self) -> None:
        """Test configure_debug_logging sets global logger."""
        logger = configure_debug_logging(enabled=True)

        assert isinstance(logger, DebugLogger)
        assert logger.enabled is True
        # Subsequent get_debug_logger() should return the same instance
        assert get_debug_logger() is logger

    def test_configure_tracing_service_name(self) -> None:
        """Test configure_tracing accepts service name."""
        tracer = configure_tracing(service_name="custom-service", enabled=False)
        assert isinstance(tracer, EvarisTracer)

    def test_configure_tracing_exporter_type(self) -> None:
        """Test configure_tracing accepts exporter type."""
        for exporter_type in ["otlp", "console", "none"]:
            tracer = configure_tracing(exporter_type=exporter_type, enabled=False)
            assert isinstance(tracer, EvarisTracer)

    def test_configure_tracing_enabled_false(self) -> None:
        """Test configure_tracing with enabled=False."""
        tracer = configure_tracing(enabled=False)
        assert not tracer.enabled
        assert isinstance(tracer._tracer, NoOpTracer)

    def test_configure_tracing_enabled_true_without_otel(self) -> None:
        """Test configure_tracing with enabled=True but no OpenTelemetry."""
        with patch("evaris.tracing.OTEL_AVAILABLE", False):
            tracer = configure_tracing(enabled=True)
            # Should fall back to NoOpTracer
            assert isinstance(tracer._tracer, NoOpTracer)


class TestTracingWithMockedOTel:
    """Tests for tracing with mocked OpenTelemetry."""

    def test_evaris_tracer_init_with_otel(self) -> None:
        """Test EvarisTracer initialization when OpenTelemetry is available."""
        # Import the actual OpenTelemetry types for mocking
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_tracer_instance = MagicMock()
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(service_name="test", exporter_type="none")

                        assert tracer.enabled
                        assert tracer._tracer == mock_tracer_instance
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_evaris_tracer_console_exporter(self) -> None:
        """Test EvarisTracer with console exporter."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.TracerProvider") as mock_provider_cls:
                    with patch("evaris.tracing.ConsoleSpanExporter") as mock_console:
                        with patch("evaris.tracing.BatchSpanProcessor") as mock_processor:
                            with patch("evaris.tracing.trace.set_tracer_provider"):
                                with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                                    mock_provider = MagicMock()
                                    mock_provider_cls.return_value = mock_provider
                                    mock_get_tracer.return_value = MagicMock()

                                    EvarisTracer(exporter_type="console")

                                    # Verify console exporter was created
                                    mock_console.assert_called_once()
                                    # Verify processor was created
                                    mock_processor.assert_called_once()
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_evaris_tracer_otlp_exporter(self) -> None:
        """Test EvarisTracer with OTLP exporter."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.TracerProvider") as mock_provider_cls:
                    with patch("evaris.tracing.OTLPSpanExporter") as mock_otlp:
                        with patch("evaris.tracing.BatchSpanProcessor"):
                            with patch("evaris.tracing.trace.set_tracer_provider"):
                                with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                                    mock_provider = MagicMock()
                                    mock_provider_cls.return_value = mock_provider
                                    mock_get_tracer.return_value = MagicMock()

                                    EvarisTracer(
                                        exporter_type="otlp", otlp_endpoint="http://test:4317"
                                    )

                                    # Verify OTLP exporter was created with endpoint
                                    mock_otlp.assert_called_once_with(endpoint="http://test:4317")
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_evaris_tracer_otlp_fallback_to_console(self) -> None:
        """Test OTLP exporter falls back to console on error."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.TracerProvider") as mock_provider_cls:
                    with patch("evaris.tracing.OTLPSpanExporter") as mock_otlp:
                        with patch("evaris.tracing.ConsoleSpanExporter") as mock_console:
                            with patch("evaris.tracing.BatchSpanProcessor"):
                                with patch("evaris.tracing.trace.set_tracer_provider"):
                                    with patch(
                                        "evaris.tracing.trace.get_tracer"
                                    ) as mock_get_tracer:
                                        # Make OTLP exporter raise an exception
                                        mock_otlp.side_effect = Exception("OTLP connection failed")
                                        mock_provider = MagicMock()
                                        mock_provider_cls.return_value = mock_provider
                                        mock_get_tracer.return_value = MagicMock()

                                        EvarisTracer(exporter_type="otlp")

                                        # Should fall back to console exporter
                                        mock_console.assert_called_once()
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_start_span_with_otel(self) -> None:
        """Test start_span when OpenTelemetry is available."""
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_span = MagicMock()
                        mock_span.__enter__ = MagicMock(return_value=mock_span)
                        mock_span.__exit__ = MagicMock(return_value=None)

                        mock_tracer_instance = MagicMock()
                        mock_tracer_instance.start_as_current_span.return_value = mock_span
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(exporter_type="none")

                        with tracer.start_span("test", attributes={"key": "value"}):
                            pass

                        # Verify span was started
                        mock_tracer_instance.start_as_current_span.assert_called_once_with("test")
                        # Verify attributes were set
                        mock_span.set_attribute.assert_called()
                        # Verify events were added
                        assert mock_span.add_event.call_count >= 2  # start and end events
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_set_attribute_with_current_span(self) -> None:
        """Test set_attribute when there's a current span."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_span = MagicMock()
                        mock_span.__enter__ = MagicMock(return_value=mock_span)
                        mock_span.__exit__ = MagicMock(return_value=None)

                        mock_tracer_instance = MagicMock()
                        mock_tracer_instance.start_as_current_span.return_value = mock_span
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(exporter_type="none")

                        with tracer.start_span("test"):
                            tracer.set_attribute("test_key", "test_value")

                        # Verify attribute was set on current span
                        calls = [
                            call
                            for call in mock_span.set_attribute.call_args_list
                            if len(call[0]) == 2 and call[0][0] == "test_key"
                        ]
                        assert len(calls) >= 1
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_add_event_with_current_span(self) -> None:
        """Test add_event when there's a current span."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_span = MagicMock()
                        mock_span.__enter__ = MagicMock(return_value=mock_span)
                        mock_span.__exit__ = MagicMock(return_value=None)

                        mock_tracer_instance = MagicMock()
                        mock_tracer_instance.start_as_current_span.return_value = mock_span
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(exporter_type="none")

                        with tracer.start_span("test"):
                            tracer.add_event("custom_event", {"detail": "info"})

                        # Find custom event calls
                        calls = [
                            call
                            for call in mock_span.add_event.call_args_list
                            if call[0][0] == "custom_event"
                        ]
                        assert len(calls) == 1
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_set_status_with_current_span(self) -> None:
        """Test set_status when there's a current span."""
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        with patch("evaris.tracing.Status"):
                            with patch("evaris.tracing.StatusCode"):
                                mock_span = MagicMock()
                                mock_span.__enter__ = MagicMock(return_value=mock_span)
                                mock_span.__exit__ = MagicMock(return_value=None)

                                mock_tracer_instance = MagicMock()
                                mock_tracer_instance.start_as_current_span.return_value = mock_span
                                mock_get_tracer.return_value = mock_tracer_instance

                                tracer = EvarisTracer(exporter_type="none")

                                with tracer.start_span("test"):
                                    tracer.set_status("ok", "Success")
                                    tracer.set_status("error", "Failed")

                                # Verify set_status was called
                                assert mock_span.set_status.call_count >= 2
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_record_exception_with_current_span(self) -> None:
        """Test record_exception when there's a current span."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_span = MagicMock()
                        mock_span.__enter__ = MagicMock(return_value=mock_span)
                        mock_span.__exit__ = MagicMock(return_value=None)

                        mock_tracer_instance = MagicMock()
                        mock_tracer_instance.start_as_current_span.return_value = mock_span
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(exporter_type="none")

                        test_exception = ValueError("test error")

                        with tracer.start_span("test"):
                            tracer.record_exception(test_exception)

                        # Verify exception was recorded
                        # Note: record_exception is called twice - once manually,
                        # once by exception handler
                        assert mock_span.record_exception.call_count >= 1
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_measure_time_with_otel(self) -> None:
        """Test measure_time with OpenTelemetry."""
        try:
            import time

            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_span = MagicMock()
                        mock_span.__enter__ = MagicMock(return_value=mock_span)
                        mock_span.__exit__ = MagicMock(return_value=None)

                        mock_tracer_instance = MagicMock()
                        mock_tracer_instance.start_as_current_span.return_value = mock_span
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(exporter_type="none")

                        with tracer.start_span("test"):
                            with tracer.measure_time("operation"):
                                time.sleep(0.01)  # Small delay

                        # Verify timing attribute was set
                        calls = [
                            call
                            for call in mock_span.set_attribute.call_args_list
                            if len(call[0]) == 2 and call[0][0] == "operation_ms"
                        ]
                        assert len(calls) >= 1
                        # Verify timing event was added
                        event_calls = [
                            call
                            for call in mock_span.add_event.call_args_list
                            if len(call[0]) > 0 and "completed" in call[0][0]
                        ]
                        assert len(event_calls) >= 1
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_evaris_tracer_custom_exporter(self) -> None:
        """Test EvarisTracer with custom exporter object."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace.export import SpanExporter

            # Create a custom exporter mock
            custom_exporter = MagicMock(spec=SpanExporter)

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.TracerProvider") as mock_provider_cls:
                    with patch("evaris.tracing.BatchSpanProcessor") as mock_processor:
                        with patch("evaris.tracing.trace.set_tracer_provider"):
                            with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                                mock_provider = MagicMock()
                                mock_provider_cls.return_value = mock_provider
                                mock_get_tracer.return_value = MagicMock()

                                # Pass custom exporter object
                                EvarisTracer(exporter_type=custom_exporter)

                                # Verify processor was created with custom exporter
                                mock_processor.assert_called_once()
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_evaris_tracer_unknown_exporter_type(self) -> None:
        """Test EvarisTracer with unknown exporter type falls back to console."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.TracerProvider") as mock_provider_cls:
                    with patch("evaris.tracing.ConsoleSpanExporter") as mock_console:
                        with patch("evaris.tracing.BatchSpanProcessor"):
                            with patch("evaris.tracing.trace.set_tracer_provider"):
                                with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                                    mock_provider = MagicMock()
                                    mock_provider_cls.return_value = mock_provider
                                    mock_get_tracer.return_value = MagicMock()

                                    # Pass unknown exporter type
                                    EvarisTracer(exporter_type="unknown_type")

                                    # Should fall back to console exporter
                                    mock_console.assert_called_once()
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_add_event_without_attributes(self) -> None:
        """Test add_event without attributes."""
        try:
            from opentelemetry import trace

            with patch("evaris.tracing.OTEL_AVAILABLE", True):
                with patch("evaris.tracing.trace.set_tracer_provider"):
                    with patch("evaris.tracing.trace.get_tracer") as mock_get_tracer:
                        mock_span = MagicMock()
                        mock_span.__enter__ = MagicMock(return_value=mock_span)
                        mock_span.__exit__ = MagicMock(return_value=None)

                        mock_tracer_instance = MagicMock()
                        mock_tracer_instance.start_as_current_span.return_value = mock_span
                        mock_get_tracer.return_value = mock_tracer_instance

                        tracer = EvarisTracer(exporter_type="none")

                        with tracer.start_span("test"):
                            # Add event without attributes
                            tracer.add_event("simple_event")

                        # Find the simple event call (without attributes parameter)
                        calls = [
                            call
                            for call in mock_span.add_event.call_args_list
                            if call[0][0] == "simple_event"
                        ]
                        assert len(calls) == 1
        except ImportError:
            pytest.skip("OpenTelemetry not installed")
