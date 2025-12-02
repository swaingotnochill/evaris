"""Tests for Phase 3 tracing enhancements (SpanType and helper methods)."""

from evaris.tracing import EvarisTracer, SpanType


class TestSpanType:
    """Tests for SpanType constants."""

    def test_has_phase1_span_types(self) -> None:
        """Test that Phase 1 span types are defined."""
        assert hasattr(SpanType, "EVALUATION")
        assert hasattr(SpanType, "TEST_CASE")
        assert hasattr(SpanType, "METRIC")

        assert SpanType.EVALUATION == "evaluation"
        assert SpanType.TEST_CASE == "test_case"
        assert SpanType.METRIC == "metric"

    def test_has_phase3_span_types(self) -> None:
        """Test that Phase 3 span types are defined."""
        assert hasattr(SpanType, "AGENT_EXECUTION")
        assert hasattr(SpanType, "LLM_CALL")
        assert hasattr(SpanType, "TOOL_CALL")
        assert hasattr(SpanType, "AGENT_STEP")
        assert hasattr(SpanType, "METRIC_REASONING")
        assert hasattr(SpanType, "EMBEDDING")

        assert SpanType.AGENT_EXECUTION == "agent_execution"
        assert SpanType.LLM_CALL == "llm_call"
        assert SpanType.TOOL_CALL == "tool_call"
        assert SpanType.AGENT_STEP == "agent_step"
        assert SpanType.METRIC_REASONING == "metric_reasoning"
        assert SpanType.EMBEDDING == "embedding"

    def test_has_utility_span_types(self) -> None:
        """Test that utility span types are defined."""
        assert hasattr(SpanType, "CACHE_LOOKUP")
        assert hasattr(SpanType, "CACHE_STORE")

        assert SpanType.CACHE_LOOKUP == "cache_lookup"
        assert SpanType.CACHE_STORE == "cache_store"


class TestTracerEnhancements:
    """Tests for EvarisTracer Phase 3 helper methods."""

    def test_set_reasoning(self) -> None:
        """Test set_reasoning helper method."""
        tracer = EvarisTracer(enabled=False)  # Disable for testing

        with tracer.start_span("test", attributes={"type": "test"}):
            tracer.set_reasoning("This is why the score was given")
            # Should not raise

    def test_set_reasoning_with_confidence(self) -> None:
        """Test set_reasoning with confidence parameter."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_reasoning("Explanation", confidence=0.95)
            # Should not raise

    def test_set_cost(self) -> None:
        """Test set_cost helper method."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_cost(0.05)
            # Should not raise

    def test_set_cost_rounds_to_6_decimals(self) -> None:
        """Test that cost is rounded to 6 decimal places."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_cost(0.123456789)
            # Should round to 0.123457

    def test_set_tokens_total_only(self) -> None:
        """Test set_tokens with total only."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_tokens(1000)
            # Should not raise

    def test_set_tokens_with_input_output(self) -> None:
        """Test set_tokens with input and output tokens."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_tokens(1000, input_tokens=600, output_tokens=400)
            # Should not raise

    def test_set_tokens_with_only_input(self) -> None:
        """Test set_tokens with only input tokens."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_tokens(1000, input_tokens=600)
            # Should not raise

    def test_set_tokens_with_only_output(self) -> None:
        """Test set_tokens with only output tokens."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_tokens(1000, output_tokens=400)
            # Should not raise

    def test_set_cached_true(self) -> None:
        """Test set_cached with True value."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_cached(True)
            # Should not raise

    def test_set_cached_false(self) -> None:
        """Test set_cached with False value."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_cached(False)
            # Should not raise

    def test_set_cached_with_cache_key(self) -> None:
        """Test set_cached with cache key."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_cached(True, cache_key="llm:abc123xyz")
            # Should not raise

    def test_set_cached_truncates_long_keys(self) -> None:
        """Test that very long cache keys are truncated."""
        tracer = EvarisTracer(enabled=False)

        long_key = "x" * 200
        with tracer.start_span("test"):
            tracer.set_cached(True, cache_key=long_key)
            # Should truncate to 100 chars

    def test_set_latency(self) -> None:
        """Test set_latency helper method."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_latency(123.45)
            # Should not raise

    def test_set_latency_rounds_to_2_decimals(self) -> None:
        """Test that latency is rounded to 2 decimal places."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_latency(123.456789)
            # Should round to 123.46

    def test_set_model_name_only(self) -> None:
        """Test set_model with model name only."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_model("gpt-4")
            # Should not raise

    def test_set_model_with_provider(self) -> None:
        """Test set_model with model name and provider."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            tracer.set_model("gpt-4", provider="openai")
            # Should not raise

    def test_all_helpers_work_together(self) -> None:
        """Test that all helper methods can be used together."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("llm_call", attributes={"type": SpanType.LLM_CALL}):
            tracer.set_model("gpt-4", provider="openai")
            tracer.set_tokens(500, input_tokens=300, output_tokens=200)
            tracer.set_cost(0.015)
            tracer.set_latency(1234.56)
            tracer.set_cached(False)
            tracer.set_reasoning("Generated response successfully", confidence=0.95)
            # Should not raise

    def test_helpers_work_without_active_span(self) -> None:
        """Test that helpers don't crash when called without active span."""
        tracer = EvarisTracer(enabled=False)

        # These should not raise even without an active span
        tracer.set_reasoning("test")
        tracer.set_cost(0.01)
        tracer.set_tokens(100)
        tracer.set_cached(True)
        tracer.set_latency(50.0)
        tracer.set_model("gpt-4")

    def test_span_type_in_attributes(self) -> None:
        """Test using SpanType constants in span attributes."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("agent_exec", attributes={"span_type": SpanType.AGENT_EXECUTION}):
            pass

        with tracer.start_span("llm", attributes={"span_type": SpanType.LLM_CALL}):
            pass

        with tracer.start_span("tool", attributes={"span_type": SpanType.TOOL_CALL}):
            pass

    def test_nested_spans_with_different_types(self) -> None:
        """Test nested spans with different span types."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("eval", attributes={"type": SpanType.EVALUATION}):
            with tracer.start_span("test_case", attributes={"type": SpanType.TEST_CASE}):
                with tracer.start_span("agent", attributes={"type": SpanType.AGENT_EXECUTION}):
                    with tracer.start_span("llm", attributes={"type": SpanType.LLM_CALL}):
                        tracer.set_model("gpt-4")
                        tracer.set_tokens(100)
                        tracer.set_cost(0.002)

                with tracer.start_span("metric", attributes={"type": SpanType.METRIC}):
                    with tracer.start_span(
                        "reasoning", attributes={"type": SpanType.METRIC_REASONING}
                    ):
                        tracer.set_reasoning("Score explanation", confidence=0.9)

    def test_measure_time_compatibility(self) -> None:
        """Test that measure_time still works with new helpers."""
        tracer = EvarisTracer(enabled=False)

        with tracer.start_span("test"):
            with tracer.measure_time("operation"):
                # Simulate some work
                import time

                time.sleep(0.01)

            # Also use new helpers
            tracer.set_model("gpt-4")
            tracer.set_cost(0.001)
