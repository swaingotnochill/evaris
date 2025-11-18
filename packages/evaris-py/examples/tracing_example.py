"""Using Evaris with OpenTelemetry Tracing and Debug Logging.

This example demonstrates how to:
1. Enable tracing to monitor evaluation performance
2. Use debug logging to inspect LLM prompts and responses
3. Export traces to console or OTLP collector
4. Measure and analyze evaluation metrics

Requirements:
    pip install evaris[tracing]
    # Optional: Run Jaeger for trace visualization
    # docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
"""

import time

from evaris import evaluate
from evaris.tracing import configure_debug_logging, configure_tracing


def chatbot(query: str) -> str:
    """A simple chatbot that answers questions.

    In a real application, this would call an LLM API.
    For this example, we simulate with a simple response.
    """
    # Simulate some processing time
    time.sleep(0.1)

    # Simple rule-based responses for demonstration
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2+2?": "4",
        "Who wrote Romeo and Juliet?": "William Shakespeare",
        "What is the largest planet?": "Jupiter",
        "What is H2O?": "Water",
    }

    return responses.get(query, "I don't know the answer to that question.")


def main() -> None:
    """Run the evaluation example with tracing enabled."""

    print("=" * 80)
    print("Evaris Tracing Example")
    print("=" * 80)
    print()

    # =========================================================================
    # Step 1: Configure Tracing
    # =========================================================================
    print("Step 1: Configuring tracing...")

    # Option A: Console exporter (prints spans to console)
    # configure_tracing(
    #     service_name="chatbot-eval",
    #     exporter_type="console",  # Change to "otlp" for Jaeger/OTLP collector
    #     enabled=True
    # )

    # Option B: OTLP exporter (send to Jaeger, Zipkin, or cloud provider)
    # Uncomment to use OTLP (requires running collector):
    configure_tracing(
        service_name="chatbot-eval",
        exporter_type="otlp",
        otlp_endpoint="http://localhost:4317",
        enabled=True,
    )

    # Option C: Disable tracing
    # configure_tracing(enabled=False)

    print("Tracing configured (console exporter)")
    print()

    # =========================================================================
    # Step 2: Configure Debug Logging
    # =========================================================================
    print("Step 2: Configuring debug logging...")

    # Enable detailed debug logs for prompts, responses, and reasoning
    configure_debug_logging(enabled=True)

    # To disable debug logging:
    # configure_debug_logging(enabled=False)

    print("Debug logging enabled")
    print()

    # =========================================================================
    # Step 3: Prepare Test Data
    # =========================================================================
    print("Step 3: Preparing test data...")

    test_data = [
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "Who wrote Romeo and Juliet?", "expected": "William Shakespeare"},
        {"input": "What is the largest planet?", "expected": "Jupiter"},
        {"input": "What is H2O?", "expected": "Water"},
    ]

    print(f"Prepared {len(test_data)} test cases")
    print()

    # =========================================================================
    # Step 4: Run Evaluation with Tracing
    # =========================================================================
    print("Step 4: Running evaluation...")
    print()

    result = evaluate(
        name="chatbot-qa-evaluation",
        task=chatbot,
        data=test_data,
        metrics=["exact_match", "latency"],
        enable_tracing=True,  # Enable tracing for this evaluation
        enable_debug=True,  # Enable debug logging for this evaluation
    )

    # =========================================================================
    # Step 5: Analyze Results
    # =========================================================================
    print()
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print()
    print(f"Name: {result.name}")
    print(f"Total Tests: {result.total}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Avg Latency: {result.avg_latency_ms:.2f}ms")
    print()

    print("Individual Test Results:")
    print("-" * 80)
    for i, test_result in enumerate(result.results, 1):
        status = "PASS" if all(m.passed for m in test_result.metrics) else "FAIL"
        input_str = str(test_result.test_case.input)[:40]
        output_str = str(test_result.output)[:30]

        print(f"{status} Test {i}: {input_str}")
        print(f"  Output: {output_str}")
        print(f"  Latency: {test_result.latency_ms:.2f}ms")

        for metric in test_result.metrics:
            print(
                f"    - {metric.name}: {metric.score:.2f} ({'PASS' if metric.passed else 'FAIL'})"
            )

        if test_result.error:
            print(f"  Error: {test_result.error}")
        print()

    # =========================================================================
    # Step 6: View Traces
    # =========================================================================
    print("=" * 80)
    print("Viewing Traces")
    print("=" * 80)
    print()
    print("Traces have been exported using the configured exporter.")
    print()
    print("Console Exporter:")
    print("  - Traces are printed above in JSON format")
    print("  - Look for 'span' objects with timing and metadata")
    print()
    print("OTLP Exporter (if configured):")
    print("  - Open Jaeger UI: http://localhost:16686")
    print("  - Search for service: 'chatbot-eval'")
    print("  - Explore span hierarchy and timing breakdown")
    print()

    # =========================================================================
    # What to Look For in Traces
    # =========================================================================
    print("=" * 80)
    print("What to Look For in Traces")
    print("=" * 80)
    print()
    print("Span Hierarchy:")
    print("  evaluation")
    print("    ├── dataset_normalization")
    print("    ├── generate_test_cases")
    print("    ├── metric_resolution")
    print("    └── test_execution")
    print("          ├── test_case_execution (x5)")
    print("          │     ├── agent_execution")
    print("          │     └── metric.ExactMatchMetric")
    print("          │     └── metric.LatencyMetric")
    print()
    print("Key Attributes:")
    print("  - eval.name: Evaluation name")
    print("  - eval.total: Total test cases")
    print("  - eval.accuracy: Final accuracy")
    print("  - eval.avg_latency_ms: Average latency")
    print("  - test.passed: Whether individual test passed")
    print("  - metric.score: Individual metric scores")
    print()

    # =========================================================================
    # Environment Variables (Alternative Configuration)
    # =========================================================================
    print("=" * 80)
    print("Environment Variable Configuration")
    print("=" * 80)
    print()
    print("Instead of configure_tracing() and configure_debug_logging(),")
    print("you can use environment variables:")
    print()
    print("  export EVARIS_TRACING=true")
    print("  export EVARIS_DEBUG=true")
    print("  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317")
    print()
    print("Then tracing and debug logging will be automatically enabled.")
    print()


if __name__ == "__main__":
    main()
