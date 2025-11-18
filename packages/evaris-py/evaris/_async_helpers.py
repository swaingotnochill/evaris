"""Async evaluation helper functions."""

import asyncio
import inspect
import time
from typing import Any, Union

from evaris.agent_interface import is_async_agent as is_async_agent_interface
from evaris.tracing import get_debug_logger, get_tracer
from evaris.types import (
    AgentFunction,
    AsyncAgentFunction,
    MetricResult,
    TestCase,
    TestResult,
)


def _is_async_agent(task: Any) -> bool:
    """Detect if agent function or AgentInterface object is async.

    Args:
        task: Agent function or AgentInterface object to check

    Returns:
        True if agent is async, False otherwise
    """
    # Check if it's an AgentInterface object with a_execute
    if is_async_agent_interface(task):
        return True

    # Check if it's an async callable
    return (
        inspect.iscoroutinefunction(task)
        or inspect.isasyncgenfunction(task)
        or (hasattr(task, "__call__") and inspect.iscoroutinefunction(task.__call__))
    )


def _has_async_metrics(metrics: list[Union[str, Any]]) -> bool:
    """Check if any metrics support async execution.

    Args:
        metrics: List of metric names or instances

    Returns:
        True if any metric has a_measure() method
    """
    for metric in metrics:
        if isinstance(metric, str):
            # Built-in metrics will be resolved later
            continue
        if hasattr(metric, "a_measure"):
            return True
    return False


async def _execute_agent_async(
    task: Union[AgentFunction, AsyncAgentFunction], input_data: Any
) -> Any:
    """Execute agent function (sync or async).

    Handles both sync and async agent functions, running sync functions
    in a thread pool to avoid blocking the event loop.

    Args:
        task: Agent function (sync or async)
        input_data: Input to pass to the agent

    Returns:
        Agent output

    Raises:
        Any exception raised by the agent
    """
    tracer = get_tracer()
    debug = get_debug_logger()

    with tracer.start_span("agent_execution_async") as _:
        try:
            if inspect.iscoroutinefunction(task):
                # Async agent - await directly
                debug.log_intermediate("agent", "Executing async agent")
                output = await task(input_data)
            else:
                # Sync agent - run in thread pool to avoid blocking
                debug.log_intermediate("agent", "Executing sync agent in thread pool")
                output = await asyncio.to_thread(task, input_data)

            return output

        except Exception as e:
            debug.log_error("agent", e, input=input_data)
            raise


async def _run_metric_async(
    metric: Any, test_case: TestCase, actual_output: Any, latency_ms: float
) -> MetricResult:
    """Run a single metric asynchronously.

    If metric implements a_measure(), uses that. Otherwise, falls back to
    running score() in a thread pool.

    Args:
        metric: Metric instance
        test_case: Test case to evaluate
        actual_output: Actual output from agent
        latency_ms: Measured latency

    Returns:
        MetricResult

    Raises:
        Any exception raised by the metric
    """
    tracer = get_tracer()
    debug = get_debug_logger()

    metric_name = type(metric).__name__

    with tracer.start_span(f"metric.{metric_name}_async") as _:
        try:
            # Check if metric implements async interface
            metric_result: MetricResult
            if hasattr(metric, "a_measure"):
                debug.log_intermediate(metric_name, "Using async a_measure()")
                metric_result = await metric.a_measure(test_case)
            else:
                # Fallback to sync metric in thread pool
                debug.log_intermediate(metric_name, "Using sync score() in thread pool")

                # Special handling for LatencyMetric which needs latency_ms
                if metric_name == "LatencyMetric":
                    metric_result = await asyncio.to_thread(
                        metric.score, test_case, actual_output, latency_ms
                    )
                else:
                    metric_result = await asyncio.to_thread(metric.score, test_case, actual_output)

            tracer.set_attribute("metric.score", metric_result.score)
            tracer.set_attribute("metric.passed", metric_result.passed)

            return metric_result

        except Exception as e:
            debug.log_error(metric_name, e, test_case=str(test_case))
            # Return failed metric result instead of raising
            return MetricResult(
                name=metric_name,
                score=0.0,
                passed=False,
                metadata={"error": str(e), "error_type": type(e).__name__},
            )


async def _run_single_test_async(
    task: Union[AgentFunction, AsyncAgentFunction],
    test_case: TestCase,
    metrics: list[Any],
) -> TestResult:
    """Run a single test case asynchronously with parallel metric execution.

    Args:
        task: Agent function (sync or async)
        test_case: Test case to execute
        metrics: List of metric instances

    Returns:
        TestResult with all metric evaluations
    """
    tracer = get_tracer()
    debug = get_debug_logger()

    with tracer.start_span("test_case_execution_async") as _:
        # Use existing actual_output if available, otherwise run the agent
        start_time = time.perf_counter()

        try:
            if test_case.actual_output is not None:
                actual_output = test_case.actual_output
                debug.log_intermediate("test", "Using existing actual_output")
                # Check if latency was already measured during golden->testcase conversion
                if "_generated_latency_ms" in test_case.metadata:
                    latency_ms = test_case.metadata["_generated_latency_ms"]
                else:
                    latency_ms = (time.perf_counter() - start_time) * 1000
            else:
                actual_output = await _execute_agent_async(task, test_case.input)
                latency_ms = (time.perf_counter() - start_time) * 1000

        except Exception as agent_error:
            # Agent failed - record error
            latency_ms = (time.perf_counter() - start_time) * 1000
            tracer.record_exception(agent_error)
            tracer.set_status("error", str(agent_error))

            return TestResult(
                test_case=test_case,
                output=None,
                metrics=[],
                latency_ms=latency_ms,
                error=str(agent_error),
            )

        tracer.set_attribute("latency_ms", latency_ms)

        # Run all metrics in parallel for this test
        debug.log_intermediate("test", f"Running {len(metrics)} metrics in parallel")

        metric_tasks = [
            _run_metric_async(metric, test_case, actual_output, latency_ms) for metric in metrics
        ]

        # Execute metrics in parallel, collecting exceptions
        metric_results = await asyncio.gather(*metric_tasks, return_exceptions=True)

        # Filter out exceptions (errors are already converted to failed MetricResults)
        final_results: list[MetricResult] = []
        for i, metric_result in enumerate(metric_results):
            if isinstance(metric_result, Exception):
                # Shouldn't happen since _run_metric_async catches exceptions,
                # but handle just in case
                final_results.append(
                    MetricResult(
                        name=type(metrics[i]).__name__,
                        score=0.0,
                        passed=False,
                        metadata={
                            "error": str(metric_result),
                            "error_type": type(metric_result).__name__,
                        },
                    )
                )
            elif isinstance(metric_result, MetricResult):
                final_results.append(metric_result)

        # Determine if test passed (all metrics passed)
        test_passed = all(m.passed for m in final_results)
        tracer.set_attribute("test.passed", test_passed)

        debug.log_intermediate(
            "test",
            f"Test {'passed' if test_passed else 'failed'}",
            passed_metrics=sum(1 for m in final_results if m.passed),
            total_metrics=len(final_results),
        )

        return TestResult(
            test_case=test_case,
            output=actual_output,
            metrics=final_results,
            latency_ms=latency_ms,
            error=None,
        )
