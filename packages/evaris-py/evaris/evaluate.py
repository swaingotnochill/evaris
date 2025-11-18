"""Core evaluation functionality."""

import asyncio
import time
import warnings
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional, Union, cast

from evaris.agent_interface import (
    is_async_agent,
    is_sync_agent,
    validate_agent,
)
from evaris.baselines import BaselineConfig, BaselineManager
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.tracing import get_debug_logger, get_tracer
from evaris.types import (
    AgentFunction,
    AsyncAgentFunction,
    BaseMetric,
    DatasetInput,
    EvalResult,
    Golden,
    MetricResult,
    MultiModalInput,
    MultiModalOutput,
    TestCase,
    TestResult,
)

# Registry of built-in metrics
BUILTIN_METRICS: dict[str, Callable[[], Any]] = {
    "exact_match": lambda: ExactMatchMetric(),
    "latency": lambda: LatencyMetric(),
}


class LatencyMetric(BaseMetric):
    """Metric that measures execution latency.

    This metric always passes and simply records the execution time.
    """

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously measure latency.

        Args:
            test_case: The test case containing latency in metadata

        Returns:
            MetricResult with the latency information
        """
        # Latency is stored in metadata during test execution
        latency_ms = test_case.metadata.get("_generated_latency_ms", 0.0)
        return MetricResult(
            name="latency",
            score=1.0,  # Latency metric always "passes"
            passed=True,
            metadata={"latency_ms": latency_ms},
        )

    def score(  # type: ignore[override]
        self, test_case: TestCase, actual_output: Any, latency_ms: float
    ) -> MetricResult:
        """Score based on latency.

        Note: This metric has a special score() signature that includes latency_ms.
        This is intentionally incompatible with BaseMetric.score().

        Args:
            test_case: The test case (unused for latency)
            actual_output: The actual output (unused for latency)
            latency_ms: The measured latency in milliseconds

        Returns:
            MetricResult with the latency information
        """
        return MetricResult(
            name="latency",
            score=1.0,  # Latency metric always "passes"
            passed=True,
            metadata={"latency_ms": latency_ms},
        )


def _normalize_data(data: Union[dict[str, Any], Golden, TestCase]) -> Union[Golden, TestCase]:
    """Convert input data to Golden or TestCase object.

    Args:
        data: Dict, Golden, or TestCase

    Returns:
        Golden if data has no actual_output, TestCase if it does
    """
    if isinstance(data, (Golden, TestCase)):
        return data

    # Check if this is a complete TestCase (has actual_output)
    if "actual_output" in data:
        return TestCase(
            input=data.get("input"),
            actual_output=data["actual_output"],
            expected=data.get("expected"),
            metadata=data.get("metadata", {}),
        )

    # Otherwise, treat as Golden (static test data)
    return Golden(
        input=data.get("input"),
        expected=data.get("expected"),
        metadata=data.get("metadata", {}),
    )


def _resolve_metrics(metrics_input: list[Union[str, Any]]) -> list[Any]:
    """Resolve metric names or instances to metric instances.

    Args:
        metrics_input: List of metric names (str) or metric instances

    Returns:
        List of metric instances

    Raises:
        ValueError: If a metric name is not recognized
    """
    metrics = []
    for item in metrics_input:
        if isinstance(item, str):
            # String name - resolve to built-in metric
            if item not in BUILTIN_METRICS:
                available = ", ".join(BUILTIN_METRICS.keys())
                raise ValueError(
                    f"Unknown metric '{
                        item}'. Available metrics: {available}"
                )
            metrics.append(BUILTIN_METRICS[item]())
        else:
            # Already a metric instance - use directly
            metrics.append(item)

    return metrics


def _wrap_agent(
    task: Union[AgentFunction, AsyncAgentFunction, Any],
) -> Union[AgentFunction, AsyncAgentFunction]:
    """Wrap an AgentInterface object into a callable function.

    If the task is already a callable function, returns it unchanged.
    If the task implements AgentInterface, wraps it to create a callable.

    Args:
        task: Either a callable function or an AgentInterface object

    Returns:
        A callable function (sync or async)

    Raises:
        TypeError: If task is neither callable nor implements AgentInterface
    """
    # If already callable, return as-is
    if callable(task):
        return task

    # Check if it's an AgentInterface object
    if is_sync_agent(task) or is_async_agent(task):
        validate_agent(task)

        # Prefer async if available, otherwise use sync
        if is_async_agent(task):
            # Return async wrapper
            async def async_wrapper(input: MultiModalInput) -> MultiModalOutput:
                result = await task.a_execute(input)
                return cast(MultiModalOutput, result)

            return async_wrapper
        else:
            # Return sync wrapper
            def sync_wrapper(input: MultiModalInput) -> MultiModalOutput:
                result = task.execute(input)
                return cast(MultiModalOutput, result)

            return sync_wrapper

    # Not a callable and not an AgentInterface
    raise TypeError(
        f"task must be either a callable function or implement AgentInterface. "
        f"Got {type(task).__name__}"
    )


def _run_single_test(
    task: AgentFunction,
    test_case: TestCase,
    metrics: list[Any],
) -> TestResult:
    """Run a single test case and evaluate it.

    Args:
        task: The agent function to test
        test_case: The test case to run
        metrics: List of metric instances to evaluate

    Returns:
        TestResult with the evaluation results
    """
    tracer = get_tracer()
    debug = get_debug_logger()

    with tracer.start_span("test_case_execution"):
        # Use existing actual_output if available, otherwise run the agent
        error = None
        output = test_case.actual_output  # Use the actual_output from TestCase

        # Only run task if actual_output is None (not just empty string)
        if output is None:
            start_time = time.perf_counter()
            with tracer.start_span("agent_execution"):
                try:
                    output = task(test_case.input)
                    tracer.set_attribute("agent.success", True)
                except Exception as e:
                    error = str(e)
                    output = ""
                    tracer.set_attribute("agent.success", False)
                    tracer.set_attribute("agent.error", str(e))
                    tracer.record_exception(e)
                    debug.log_error("agent_execution", e, input=test_case.input)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        else:
            # Check if latency was already measured during golden->testcase conversion
            if "_generated_latency_ms" in test_case.metadata:
                latency_ms = test_case.metadata["_generated_latency_ms"]
            else:
                # Use existing output, set latency to 0 (agent already ran elsewhere)
                latency_ms = 0.0

        tracer.set_attribute("latency_ms", round(latency_ms, 2))

        # Validate test case has expected value (unless using validation-free metrics)
        if test_case.expected is None and metrics:
            # Check if any metrics require expected values
            requires_expected = any(not isinstance(m, LatencyMetric) for m in metrics)
            if requires_expected:
                import logging

                logging.warning(
                    f"Test case has no expected value, metrics may produce misleading results. "
                    f"Input: {str(test_case.input)[:100]}"
                )

        # Evaluate with each metric (in parallel for better performance)
        metric_results = []

        if error is None:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def run_metric(metric: Any) -> MetricResult:
                """Run a single metric and handle errors."""
                metric_name = getattr(metric, "__class__", type(metric)).__name__

                with tracer.start_span(f"metric.{metric_name}") as metric_span:
                    try:
                        # Latency metric needs the latency value
                        if isinstance(metric, LatencyMetric):
                            result = metric.score(test_case, output, latency_ms)
                        else:
                            result = metric.score(test_case, output)

                        # Record metric result in span
                        if metric_span:
                            metric_span.set_attribute("metric.score", result.score)
                            metric_span.set_attribute("metric.passed", result.passed)
                            tracer.set_status("ok")

                        return result
                    except Exception as e:
                        # If metric evaluation fails, record it
                        tracer.set_attribute("metric.error", str(e))
                        tracer.record_exception(e)
                        tracer.set_status("error", str(e))
                        debug.log_error(metric_name, e, test_case=str(test_case)[:100])

                        return MetricResult(
                            name=metric_name,
                            score=0.0,
                            passed=False,
                            metadata={"error": str(e), "error_type": type(e).__name__},
                        )

            # Run all metrics in parallel
            with ThreadPoolExecutor(max_workers=len(metrics)) as executor:
                futures = {executor.submit(run_metric, metric): metric for metric in metrics}
                for future in as_completed(futures):
                    result = future.result()
                    metric_results.append(result)

        # Record overall test result
        all_passed = error is None and all(m.passed for m in metric_results)
        tracer.set_attribute("test.passed", all_passed)
        tracer.set_attribute("test.metrics_count", len(metric_results))

        return TestResult(
            test_case=test_case,
            output=output,
            metrics=metric_results,
            latency_ms=latency_ms,
            error=error,
        )


def evaluate_sync(
    name: str,
    task: Union[AgentFunction, Any],
    data: DatasetInput,
    metrics: list[Union[str, Any]],
    baselines: Union[bool, BaselineConfig] = True,
    compliance_config: Optional[Any] = None,
    enable_tracing: Optional[bool] = None,
    enable_debug: Optional[bool] = None,
) -> EvalResult:
    """Synchronously evaluate an agent on a dataset using specified metrics.

    This is the synchronous version of evaluate() that runs test cases sequentially.
    For better performance with I/O-bound tasks, consider using evaluate_async().

    Args:
        name: Name of the evaluation (for identification)
        task: The agent to evaluate. Can be:
              - A callable function that accepts input and returns output
              - An object implementing AgentInterface (with execute/a_execute methods)
              - A wrapped agent (LangChainAgentWrapper, etc.)
        data: Dataset to evaluate on. Can be:
              - List of dicts with 'input' and optional 'expected' keys
              - List of TestCase objects
              - Path to a dataset file (not yet implemented)
        metrics: List of metrics to apply. Can be:
                 - Metric names (str): "exact_match", "latency"
                 - Metric instances: LLMJudgeMetric(), SemanticSimilarityMetric()
                 - Mix of both
        baselines: Enable baseline comparison (default: True).Can be:
                   - True: Use default baseline configuration
                   - False: Disable baseline comparison
                   - BaselineConfig: Custom baseline configuration
                   Baselines use the same metrics as the agent for fair comparison (ABC R.12, R.13).
        compliance_config: Optional ABC compliance configuration. If provided,
                          runs compliance checks and reports warnings.
        enable_tracing: Enable OpenTelemetry tracing (overrides EVARIS_TRACING env var)
        enable_debug: Enable debug logging (overrides EVARIS_DEBUG env var)

    Returns:
        EvalResult containing aggregated results and individual test results

    Raises:
        ValueError: If data is empty or metrics are invalid
        TypeError: If required arguments are missing
        ABCViolationError: If compliance_config.strict_mode=True and critical violations found

    Example:
        >>> def my_agent(query: str) -> str:
        ...     return f"Hello {query}"
        ...
        >>> result = evaluate_sync(
        ...     name="greeting-test",
        ...     task=my_agent,
        ...     data=[{"input": "World", "expected": "Hello World"}],
        ...     metrics=["exact_match", "latency"]
        ... )
        >>> print(result.accuracy)
        1.0

        >>> # With ABC compliance checking
        >>> from evaris.compliance import ABCComplianceConfig
        >>> config = ABCComplianceConfig(enabled=True, strict_mode=False)
        >>> result = evaluate_sync(
        ...     name="abc-eval",
        ...     task=my_agent,
        ...     data=test_cases,
        ...     metrics=["exact_match"],
        ...     compliance_config=config
        ... )

        >>> # With tracing and debug logging
        >>> result = evaluate_sync(
        ...     name="traced-eval",
        ...     task=my_agent,
        ...     data=test_cases,
        ...     metrics=["exact_match"],
        ...     enable_tracing=True,
        ...     enable_debug=True
        ... )
    """
    # Configure tracing and debug logging if specified
    from evaris.tracing import configure_debug_logging, configure_tracing

    if enable_tracing is not None:
        configure_tracing(service_name="evaris", enabled=enable_tracing)
    if enable_debug is not None:
        configure_debug_logging(enabled=enable_debug)

    tracer = get_tracer()
    debug = get_debug_logger()

    # Wrap agent if needed (converts AgentInterface to callable)
    task = _wrap_agent(task)

    with tracer.start_span("evaluation", attributes={"eval.name": name}) as eval_span:
        # Validate inputs
        if not data:
            raise ValueError("Data must contain at least one test case")

        tracer.set_attribute("eval.data_count", len(data))
        tracer.set_attribute("eval.metrics_count", len(metrics))

        # Normalize data to Golden or TestCase objects
        with tracer.start_span("dataset_normalization"):
            normalized_data = [_normalize_data(d) for d in data]

            # Separate Goldens from TestCases
            goldens: list[Golden] = [item for item in normalized_data if isinstance(item, Golden)]
            test_cases: list[TestCase] = [
                item for item in normalized_data if isinstance(item, TestCase)
            ]

            tracer.set_attribute("dataset.goldens_count", len(goldens))
            tracer.set_attribute("dataset.test_cases_count", len(test_cases))

            debug.log_intermediate(
                "dataset_normalization",
                "Dataset split",
                goldens=len(goldens),
                existing_test_cases=len(test_cases),
            )

        # Generate TestCases from Goldens by running the agent
        if goldens:
            with tracer.start_span("generate_test_cases") as gen_span:
                for i, golden in enumerate(goldens):
                    # Measure latency when generating test cases from goldens
                    start_time = time.perf_counter()
                    try:
                        actual_output = task(golden.input)
                        end_time = time.perf_counter()
                        latency_ms = (end_time - start_time) * 1000
                    except Exception as e:
                        # If agent fails, set actual_output to None so _run_single_test
                        # will try running it again and properly catch/record the error
                        actual_output = None
                        end_time = time.perf_counter()
                        latency_ms = (end_time - start_time) * 1000
                        tracer.record_exception(e)
                        debug.log_error("generate_test_case", e, golden_index=i)

                    test_case = TestCase.from_golden(golden, actual_output)
                    # Copy metadata before modifying to avoid polluting Golden
                    test_case.metadata = test_case.metadata.copy()
                    test_case.metadata["_generated_latency_ms"] = latency_ms
                    test_cases.append(test_case)

                if gen_span:
                    gen_span.set_attribute("generated_count", len(goldens))

        tracer.set_attribute("eval.total_test_cases", len(test_cases))

        # ABC Compliance checking (if enabled)
        if compliance_config is not None:
            with tracer.start_span("compliance_check"):
                from evaris.compliance import ABCComplianceChecker, ABCViolationError

                checker = ABCComplianceChecker(compliance_config)

                # Check evaluation configuration
                # Baselines are now integrated into evaluate()
                # TODO: Detect if statistics are used (requires architectural change to pass
                #       statistics info to evaluate() or integrate StatisticalAnalyzer)
                report = checker.check_evaluation_config(
                    includes_baselines=(baselines is not False),
                    includes_statistics=False,
                    sample_size=len(test_cases),
                )

                tracer.set_attribute("compliance.has_warnings", report.has_warnings())
                tracer.set_attribute("compliance.has_critical", report.has_critical())

                # In strict mode, raise exception if critical violations
                if compliance_config.strict_mode and report.has_critical():
                    raise ABCViolationError(report)

                # Otherwise, warn if any issues
                if report.has_warnings():
                    warnings.warn(
                        f"\n{report.format_warnings()}\n"
                        "See docs/ABC_COMPLIANCE.md for guidance on addressing these issues.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Resolve metrics
        with tracer.start_span("metric_resolution"):
            metric_instances = _resolve_metrics(metrics)
            tracer.set_attribute("metrics.resolved_count", len(metric_instances))
            metric_names = [type(m).__name__ for m in metric_instances]
            debug.log_intermediate("metric_resolution", "Resolved metrics", metrics=metric_names)

        # Run evaluation on each test case
        results = []
        with tracer.start_span("test_execution") as test_exec_span:
            for i, test_case in enumerate(test_cases):
                result = _run_single_test(task, test_case, metric_instances)
                results.append(result)

                # Add progress event
                if (i + 1) % 10 == 0 or (i + 1) == len(test_cases):
                    tracer.add_event(
                        "progress",
                        {"completed": i + 1, "total": len(test_cases)},
                    )

            if test_exec_span:
                test_exec_span.set_attribute("tests_executed", len(results))

        # Aggregate results
        with tracer.start_span("result_aggregation"):
            total = len(results)
            passed = sum(1 for r in results if r.error is None and all(m.passed for m in r.metrics))
            failed = total - passed
            accuracy = passed / total if total > 0 else 0.0
            avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0.0

            tracer.set_attribute("eval.total", total)
            tracer.set_attribute("eval.passed", passed)
            tracer.set_attribute("eval.failed", failed)
            tracer.set_attribute("eval.accuracy", round(accuracy, 4))
            tracer.set_attribute("eval.avg_latency_ms", round(avg_latency, 2))

            debug.log_intermediate(
                "result_aggregation",
                "Final results",
                total=total,
                passed=passed,
                failed=failed,
                accuracy=f"{accuracy:.2%}",
                avg_latency_ms=f"{avg_latency:.2f}",
            )

        # Create initial evaluation result
        eval_result = EvalResult(
            name=name,
            total=total,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            results=results,
            baseline_results=None,
            baseline_comparison=None,
        )

        # Run baseline comparison if enabled (ABC R.12, R.13)
        if baselines is not False:
            with tracer.start_span("baseline_comparison"):
                # Create baseline configuration
                baseline_config = (
                    baselines if isinstance(baselines, BaselineConfig) else BaselineConfig()
                )
                baseline_manager = BaselineManager(baseline_config)

                # Compare with baselines using same metrics
                baseline_comparison = baseline_manager.compare_with_baselines(
                    eval_result, metric_instances, agent_name=name
                )

                # Update eval result with baseline comparison
                eval_result.baseline_comparison = baseline_comparison

                debug.log_intermediate(
                    "baseline_comparison",
                    "Baseline comparison complete",
                    best_baseline=baseline_comparison.best_baseline,
                    improvement=baseline_comparison.improvement_over_best,
                    passes_all=baseline_comparison.passes_all_baselines,
                )

        # Mark evaluation as successful
        if eval_span:
            tracer.set_status("ok")

        return eval_result


async def evaluate_async(
    name: str,
    task: Union[AgentFunction, AsyncAgentFunction, Any],
    data: DatasetInput,
    metrics: list[Union[str, Any]],
    max_concurrency: int = 10,
    baselines: Union[bool, BaselineConfig] = True,
    compliance_config: Optional[Any] = None,
    enable_tracing: Optional[bool] = None,
    enable_debug: Optional[bool] = None,
) -> EvalResult:
    """Asynchronously evaluate an agent on a dataset using specified metrics.

    This is the async version of evaluate() that runs test cases in parallel
    for improved performance. It supports both sync and async agent functions,
    automatically running sync agents in a thread pool.

    Args:
        name: Name of the evaluation (for identification)
        task: The agent to evaluate. Can be:
              - A callable function (sync or async) that accepts input and returns output
              - An object implementing AgentInterface (with execute/a_execute methods)
              - A wrapped agent (LangChainAgentWrapper, etc.)
        data: Dataset to evaluate on. Can be:
              - List of dicts with 'input' and optional 'expected' keys
              - List of TestCase objects
              - Path to a dataset file (not yet implemented)
        metrics: List of metrics to apply. Can be:
                 - Metric names (str): "exact_match", "latency"
                 - Metric instances: LLMJudgeMetric(), SemanticSimilarityMetric()
                 - Mix of both
        max_concurrency: Maximum number of concurrent test executions (default: 10)
        compliance_config: Optional ABC compliance configuration. If provided,
                          runs compliance checks and reports warnings.
        enable_tracing: Enable OpenTelemetry tracing (overrides EVARIS_TRACING env var)
        enable_debug: Enable debug logging (overrides EVARIS_DEBUG env var)

    Returns:
        EvalResult containing aggregated results and individual test results

    Raises:
        ValueError: If data is empty or metrics are invalid
        TypeError: If required arguments are missing
        ABCViolationError: If compliance_config.strict_mode=True and critical violations found

    Example:
        >>> async def my_async_agent(query: str) -> str:
        ...     await asyncio.sleep(0.1)  # Simulate async work
        ...     return f"Hello {query}"
        ...
        >>> result = await evaluate_async(
        ...     name="greeting-test",
        ...     task=my_async_agent,
        ...     data=[{"input": "World", "expected": "Hello World"}],
        ...     metrics=["exact_match", "latency"],
        ...     max_concurrency=10
        ... )
        >>> print(result.accuracy)
        1.0

        >>> # Also works with sync agents (runs in thread pool)
        >>> def my_sync_agent(query: str) -> str:
        ...     return f"Hello {query}"
        ...
        >>> result = await evaluate_async(
        ...     name="sync-agent-test",
        ...     task=my_sync_agent,
        ...     data=test_cases,
        ...     metrics=["exact_match"]
        ... )
    """
    # Configure tracing and debug logging if specified
    from evaris.tracing import configure_debug_logging, configure_tracing

    if enable_tracing is not None:
        configure_tracing(service_name="evaris", enabled=enable_tracing)
    if enable_debug is not None:
        configure_debug_logging(enabled=enable_debug)

    tracer = get_tracer()
    debug = get_debug_logger()

    # Wrap agent if needed (converts AgentInterface to callable)
    task = _wrap_agent(task)

    with tracer.start_span("evaluation_async", attributes={"eval.name": name}) as eval_span:
        # Validate inputs
        if not data:
            raise ValueError("Data must contain at least one test case")

        tracer.set_attribute("eval.data_count", len(data))
        tracer.set_attribute("eval.metrics_count", len(metrics))
        tracer.set_attribute("eval.max_concurrency", max_concurrency)

        # Normalize data to Golden or TestCase objects
        with tracer.start_span("dataset_normalization"):
            normalized_data = [_normalize_data(d) for d in data]

            # Separate Goldens from TestCases
            goldens: list[Golden] = [item for item in normalized_data if isinstance(item, Golden)]
            test_cases: list[TestCase] = [
                item for item in normalized_data if isinstance(item, TestCase)
            ]

            tracer.set_attribute("dataset.goldens_count", len(goldens))
            tracer.set_attribute("dataset.test_cases_count", len(test_cases))

            debug.log_intermediate(
                "dataset_normalization",
                "Dataset split",
                goldens=len(goldens),
                existing_test_cases=len(test_cases),
            )

        # Generate TestCases from Goldens by running the agent
        if goldens:
            with tracer.start_span("generate_test_cases_async") as gen_span:
                # Import async helpers
                from evaris._async_helpers import _execute_agent_async

                # Generate test cases in parallel with concurrency control
                semaphore = asyncio.Semaphore(max_concurrency)

                async def generate_one(golden: Golden, index: int) -> TestCase:
                    async with semaphore:
                        # Measure latency when generating test cases from goldens
                        start_time = time.perf_counter()
                        try:
                            actual_output = await _execute_agent_async(task, golden.input)
                            end_time = time.perf_counter()
                            latency_ms = (end_time - start_time) * 1000
                        except Exception as e:
                            # If agent fails, set actual_output to None so _run_single_test
                            # will try running it again and properly catch/record the error
                            actual_output = None
                            end_time = time.perf_counter()
                            latency_ms = (end_time - start_time) * 1000
                            tracer.record_exception(e)
                            debug.log_error("generate_test_case", e, golden_index=index)

                        test_case = TestCase.from_golden(golden, actual_output)
                        # Copy metadata before modifying to avoid polluting Golden
                        test_case.metadata = test_case.metadata.copy()
                        test_case.metadata["_generated_latency_ms"] = latency_ms
                        return test_case

                # Generate all test cases in parallel
                generated_cases = await asyncio.gather(
                    *[generate_one(g, i) for i, g in enumerate(goldens)],
                    return_exceptions=True,
                )

                # Filter out exceptions and add to test_cases
                for case_result in generated_cases:
                    if isinstance(case_result, Exception):
                        # Log but continue - error already recorded
                        debug.log_error("generate_test_cases", case_result)
                    elif isinstance(case_result, TestCase):
                        test_cases.append(case_result)

                if gen_span:
                    gen_span.set_attribute("generated_count", len(goldens))

        tracer.set_attribute("eval.total_test_cases", len(test_cases))

        # ABC Compliance checking (if enabled)
        if compliance_config is not None:
            with tracer.start_span("compliance_check"):
                from evaris.compliance import ABCComplianceChecker, ABCViolationError

                checker = ABCComplianceChecker(compliance_config)

                # Baselines are now integrated into evaluate()
                report = checker.check_evaluation_config(
                    includes_baselines=(baselines is not False),
                    includes_statistics=False,
                    sample_size=len(test_cases),
                )

                tracer.set_attribute("compliance.has_warnings", report.has_warnings())
                tracer.set_attribute("compliance.has_critical", report.has_critical())

                # In strict mode, raise exception if critical violations
                if compliance_config.strict_mode and report.has_critical():
                    raise ABCViolationError(report)

                # Otherwise, warn if any issues
                if report.has_warnings():
                    warnings.warn(
                        f"\n{report.format_warnings()}\n"
                        "See docs/ABC_COMPLIANCE.md for guidance on addressing these issues.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Resolve metrics
        with tracer.start_span("metric_resolution"):
            metric_instances = _resolve_metrics(metrics)
            tracer.set_attribute("metrics.resolved_count", len(metric_instances))
            metric_names = [type(m).__name__ for m in metric_instances]
            debug.log_intermediate("metric_resolution", "Resolved metrics", metrics=metric_names)

        # Run evaluation on each test case in parallel
        with tracer.start_span("test_execution_async") as test_exec_span:
            # Import async helper
            from evaris._async_helpers import _run_single_test_async

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            async def run_one_test(test_case: TestCase, index: int) -> TestResult:
                async with semaphore:
                    result = await _run_single_test_async(task, test_case, metric_instances)

                    # Add progress event
                    if (index + 1) % 10 == 0 or (index + 1) == len(test_cases):
                        tracer.add_event(
                            "progress",
                            {"completed": index + 1, "total": len(test_cases)},
                        )

                    return result

            # Run all tests in parallel
            results = await asyncio.gather(
                *[run_one_test(tc, i) for i, tc in enumerate(test_cases)],
                return_exceptions=True,
            )

            # Filter out exceptions (convert to failed TestResults)
            final_results: list[TestResult] = []
            for i, test_result in enumerate(results):
                if isinstance(test_result, Exception):
                    # Create failed test result
                    final_results.append(
                        TestResult(
                            test_case=test_cases[i],
                            output=None,
                            metrics=[],
                            latency_ms=0.0,
                            error=str(test_result),
                        )
                    )
                    tracer.record_exception(test_result)
                    debug.log_error("test_execution", test_result, test_index=i)
                elif isinstance(test_result, TestResult):
                    final_results.append(test_result)

            if test_exec_span:
                test_exec_span.set_attribute("tests_executed", len(final_results))

        # Aggregate results
        with tracer.start_span("result_aggregation"):
            total = len(final_results)
            passed = sum(
                1 for r in final_results if r.error is None and all(m.passed for m in r.metrics)
            )
            failed = total - passed
            accuracy = passed / total if total > 0 else 0.0
            avg_latency = sum(r.latency_ms for r in final_results) / total if total > 0 else 0.0

            tracer.set_attribute("eval.total", total)
            tracer.set_attribute("eval.passed", passed)
            tracer.set_attribute("eval.failed", failed)
            tracer.set_attribute("eval.accuracy", round(accuracy, 4))
            tracer.set_attribute("eval.avg_latency_ms", round(avg_latency, 2))

            debug.log_intermediate(
                "result_aggregation",
                "Final results",
                total=total,
                passed=passed,
                failed=failed,
                accuracy=f"{accuracy:.2%}",
                avg_latency_ms=f"{avg_latency:.2f}",
            )

        # Create initial evaluation result
        eval_result = EvalResult(
            name=name,
            total=total,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            results=final_results,
            baseline_results=None,
            baseline_comparison=None,
        )

        # Run baseline comparison if enabled (ABC R.12, R.13)
        if baselines is not False:
            with tracer.start_span("baseline_comparison"):
                # Create baseline configuration
                baseline_config = (
                    baselines if isinstance(baselines, BaselineConfig) else BaselineConfig()
                )
                baseline_manager = BaselineManager(baseline_config)

                # Compare with baselines using same metrics
                baseline_comparison = baseline_manager.compare_with_baselines(
                    eval_result, metric_instances, agent_name=name
                )

                # Update eval result with baseline comparison
                eval_result.baseline_comparison = baseline_comparison

                debug.log_intermediate(
                    "baseline_comparison",
                    "Baseline comparison complete",
                    best_baseline=baseline_comparison.best_baseline,
                    improvement=baseline_comparison.improvement_over_best,
                    passes_all=baseline_comparison.passes_all_baselines,
                )

        # Mark evaluation as successful
        if eval_span:
            tracer.set_status("ok")

        return eval_result


def evaluate(
    name: str,
    task: Union[AgentFunction, AsyncAgentFunction, Any],
    data: DatasetInput,
    metrics: list[Union[str, Any]],
    max_concurrency: Optional[int] = None,
    baselines: Union[bool, BaselineConfig] = True,
    compliance_config: Optional[Any] = None,
    enable_tracing: Optional[bool] = None,
    enable_debug: Optional[bool] = None,
) -> EvalResult:
    """Evaluate an agent on a dataset using specified metrics with smart routing.

    This is the main entry point for running evaluations. It automatically detects
    whether the agent is async and routes to the appropriate evaluation function:
    - Async agents or agents with async metrics -> evaluate_async() (parallel execution)
    - Sync agents with sync metrics -> evaluate_sync() (sequential execution)

    Users can force sync execution by using evaluate_sync() directly, or force async
    execution by using evaluate_async() directly.

    Args:
        name: Name of the evaluation (for identification)
        task: The agent to evaluate. Can be:
              - A callable function (sync or async) that accepts input and returns output
              - An object implementing AgentInterface (with execute/a_execute methods)
              - A wrapped agent (LangChainAgentWrapper, etc.)
              Smart routing will automatically detect async capability.
        data: Dataset to evaluate on. Can be:
              - List of dicts with 'input' and optional 'expected' keys
              - List of TestCase objects
              - Path to a dataset file (not yet implemented)
        metrics: List of metrics to apply. Can be:
                 - Metric names (str): "exact_match", "latency"
                 - Metric instances: LLMJudgeMetric(), SemanticSimilarityMetric()
                 - Mix of both
        max_concurrency: Maximum concurrent test executions (default: 10).
                        Only used if async execution is chosen. Set to None
                        to force sync execution.
        compliance_config: Optional ABC compliance configuration. If provided,
                          runs compliance checks and reports warnings.
        enable_tracing: Enable OpenTelemetry tracing (overrides EVARIS_TRACING env var)
        enable_debug: Enable debug logging (overrides EVARIS_DEBUG env var)

    Returns:
        EvalResult containing aggregated results and individual test results

    Raises:
        ValueError: If data is empty or metrics are invalid
        TypeError: If required arguments are missing
        ABCViolationError: If compliance_config.strict_mode=True and critical violations found

    Example:
        >>> # Async agent - automatically uses parallel execution
        >>> async def my_async_agent(query: str) -> str:
        ...     await asyncio.sleep(0.1)
        ...     return f"Hello {query}"
        ...
        >>> result = evaluate(
        ...     name="async-greeting-test",
        ...     task=my_async_agent,
        ...     data=[{"input": "World", "expected": "Hello World"}],
        ...     metrics=["exact_match", "latency"]
        ... )

        >>> # Sync agent - automatically uses sequential execution
        >>> def my_sync_agent(query: str) -> str:
        ...     return f"Hello {query}"
        ...
        >>> result = evaluate(
        ...     name="sync-greeting-test",
        ...     task=my_sync_agent,
        ...     data=[{"input": "World", "expected": "Hello World"}],
        ...     metrics=["exact_match"]
        ... )

        >>> # Force async execution for sync agent (useful for I/O-bound agents)
        >>> result = evaluate_async(
        ...     name="forced-async",
        ...     task=my_sync_agent,
        ...     data=test_cases,
        ...     metrics=["exact_match"],
        ...     max_concurrency=10
        ... )

        >>> # Force sync execution for async agent
        >>> result = evaluate_sync(
        ...     name="forced-sync",
        ...     task=my_sync_agent,
        ...     data=test_cases,
        ...     metrics=["exact_match"]
        ... )
    """
    from evaris._async_helpers import _has_async_metrics, _is_async_agent

    debug = get_debug_logger()

    # Detect if agent or metrics are async
    is_async_agent = _is_async_agent(task)
    has_async_metrics = _has_async_metrics(metrics)

    # Decide routing
    use_async = is_async_agent or has_async_metrics

    debug.log_intermediate(
        "evaluate",
        "Smart routing decision",
        is_async_agent=is_async_agent,
        has_async_metrics=has_async_metrics,
        use_async=use_async,
    )

    # Route to appropriate function
    if use_async:
        # Use async evaluation with parallel execution

        # Run async evaluation in event loop
        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # Already in async context - cannot use asyncio.run()
            raise RuntimeError(
                "evaluate() detected an async agent but is being called from "
                "within an async context. Please use 'await evaluate_async()' instead."
            )
        except RuntimeError as e:
            # Check if this is the "no running loop" error (expected)
            if "no running event loop" in str(e).lower():
                # No event loop - safe to create one
                return asyncio.run(
                    evaluate_async(
                        name=name,
                        task=task,
                        data=data,
                        metrics=metrics,
                        max_concurrency=max_concurrency or 10,
                        baselines=baselines,
                        compliance_config=compliance_config,
                        enable_tracing=enable_tracing,
                        enable_debug=enable_debug,
                    )
                )
            else:
                # Re-raise the error about being in async context
                raise

    else:
        # Use sync evaluation
        return evaluate_sync(
            name=name,
            task=task,
            data=data,
            metrics=metrics,
            baselines=baselines,
            compliance_config=compliance_config,
            enable_tracing=enable_tracing,
            enable_debug=enable_debug,
        )


async def evaluate_stream(
    name: str,
    task: Union[AgentFunction, AsyncAgentFunction],
    data: DatasetInput,
    metrics: list[Union[str, Any]],
    max_concurrency: int = 10,
    compliance_config: Optional[Any] = None,
    enable_tracing: Optional[bool] = None,
    enable_debug: Optional[bool] = None,
) -> AsyncIterator[TestResult]:
    """Stream evaluation results as they complete (async generator).

    This is an async generator version of evaluate_async() that yields TestResult
    objects as soon as they complete, allowing for real-time progress monitoring
    and early result processing. Use this when you want to:
    - Display results in real-time as tests complete
    - Process results immediately without waiting for all tests
    - Build progress bars or streaming UIs
    - Implement early stopping based on results

    Args:
        name: Name of the evaluation (for identification)
        task: The agent function to evaluate (sync or async). Should accept
              the test case input and return the agent's output.
        data: Dataset to evaluate on. Can be:
              - List of dicts with 'input' and optional 'expected' keys
              - List of TestCase objects
              - Path to a dataset file (not yet implemented)
        metrics: List of metrics to apply. Can be:
                 - Metric names (str): "exact_match", "latency"
                 - Metric instances: LLMJudgeMetric(), SemanticSimilarityMetric()
                 - Mix of both
        max_concurrency: Maximum number of concurrent test executions (default: 10)
        compliance_config: Optional ABC compliance configuration. If provided,
                          runs compliance checks and reports warnings.
        enable_tracing: Enable OpenTelemetry tracing (overrides EVARIS_TRACING env var)
        enable_debug: Enable debug logging (overrides EVARIS_DEBUG env var)

    Yields:
        TestResult objects as they complete

    Raises:
        ValueError: If data is empty or metrics are invalid
        TypeError: If required arguments are missing
        ABCViolationError: If compliance_config.strict_mode=True and critical violations found

    Example:
        >>> async def my_async_agent(query: str) -> str:
        ...     await asyncio.sleep(0.1)
        ...     return f"Hello {query}"
        ...
        >>> async for result in evaluate_stream(
        ...     name="streaming-test",
        ...     task=my_async_agent,
        ...     data=[{"input": "World", "expected": "Hello World"}] * 100,
        ...     metrics=["exact_match", "latency"]
        ... ):
        ...     print(f"Test completed: {result.test_case.input}")
        ...     if all(m.passed for m in result.metrics):
        ...         print("  PASSED")
        ...     else:
        ...         print("  FAILED")

        >>> # With progress bar
        >>> from tqdm.asyncio import tqdm
        >>> results = []
        >>> async for result in tqdm(
        ...     evaluate_stream(
        ...         name="progress-test",
        ...         task=my_async_agent,
        ...         data=test_cases,
        ...         metrics=["exact_match"]
        ...     ),
        ...     total=len(test_cases)
        ... ):
        ...     results.append(result)
    """
    # Configure tracing and debug logging if specified
    from evaris.tracing import configure_debug_logging, configure_tracing

    if enable_tracing is not None:
        configure_tracing(service_name="evaris", enabled=enable_tracing)
    if enable_debug is not None:
        configure_debug_logging(enabled=enable_debug)

    tracer = get_tracer()
    debug = get_debug_logger()

    with tracer.start_span("evaluation_stream", attributes={"eval.name": name}) as eval_span:
        # Validate inputs
        if not data:
            raise ValueError("Data must contain at least one test case")

        tracer.set_attribute("eval.data_count", len(data))
        tracer.set_attribute("eval.metrics_count", len(metrics))
        tracer.set_attribute("eval.max_concurrency", max_concurrency)

        # Normalize data to Golden or TestCase objects
        with tracer.start_span("dataset_normalization"):
            normalized_data = [_normalize_data(d) for d in data]

            # Separate Goldens from TestCases
            goldens: list[Golden] = [item for item in normalized_data if isinstance(item, Golden)]
            test_cases: list[TestCase] = [
                item for item in normalized_data if isinstance(item, TestCase)
            ]

            tracer.set_attribute("dataset.goldens_count", len(goldens))
            tracer.set_attribute("dataset.test_cases_count", len(test_cases))

            debug.log_intermediate(
                "dataset_normalization",
                "Dataset split",
                goldens=len(goldens),
                existing_test_cases=len(test_cases),
            )

        # Generate TestCases from Goldens by running the agent
        if goldens:
            with tracer.start_span("generate_test_cases_async") as gen_span:
                # Import async helpers
                from evaris._async_helpers import _execute_agent_async

                # Generate test cases in parallel with concurrency control
                semaphore = asyncio.Semaphore(max_concurrency)

                async def generate_one(golden: Golden, index: int) -> TestCase:
                    async with semaphore:
                        # Measure latency when generating test cases from goldens
                        start_time = time.perf_counter()
                        try:
                            actual_output = await _execute_agent_async(task, golden.input)
                            end_time = time.perf_counter()
                            latency_ms = (end_time - start_time) * 1000
                        except Exception as e:
                            # If agent fails, set actual_output to None so _run_single_test
                            # will try running it again and properly catch/record the error
                            actual_output = None
                            end_time = time.perf_counter()
                            latency_ms = (end_time - start_time) * 1000
                            tracer.record_exception(e)
                            debug.log_error("generate_test_case", e, golden_index=index)

                        test_case = TestCase.from_golden(golden, actual_output)
                        # Copy metadata before modifying to avoid polluting Golden
                        test_case.metadata = test_case.metadata.copy()
                        test_case.metadata["_generated_latency_ms"] = latency_ms
                        return test_case

                # Generate all test cases in parallel
                generated_cases = await asyncio.gather(
                    *[generate_one(g, i) for i, g in enumerate(goldens)],
                    return_exceptions=True,
                )

                # Filter out exceptions and add to test_cases
                for case_result in generated_cases:
                    if isinstance(case_result, Exception):
                        # Log but continue - error already recorded
                        debug.log_error("generate_test_cases", case_result)
                    elif isinstance(case_result, TestCase):
                        test_cases.append(case_result)

                if gen_span:
                    gen_span.set_attribute("generated_count", len(goldens))

        tracer.set_attribute("eval.total_test_cases", len(test_cases))

        # ABC Compliance checking (if enabled)
        if compliance_config is not None:
            with tracer.start_span("compliance_check"):
                from evaris.compliance import ABCComplianceChecker, ABCViolationError

                checker = ABCComplianceChecker(compliance_config)

                # Baselines are now integrated into evaluate()
                report = checker.check_evaluation_config(
                    includes_baselines=False,
                    includes_statistics=False,
                    sample_size=len(test_cases),
                )

                tracer.set_attribute("compliance.has_warnings", report.has_warnings())
                tracer.set_attribute("compliance.has_critical", report.has_critical())

                # In strict mode, raise exception if critical violations
                if compliance_config.strict_mode and report.has_critical():
                    raise ABCViolationError(report)

                # Otherwise, warn if any issues
                if report.has_warnings():
                    warnings.warn(
                        f"\n{report.format_warnings()}\n"
                        "See docs/ABC_COMPLIANCE.md for guidance on addressing these issues.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Resolve metrics
        with tracer.start_span("metric_resolution"):
            metric_instances = _resolve_metrics(metrics)
            tracer.set_attribute("metrics.resolved_count", len(metric_instances))
            metric_names = [type(m).__name__ for m in metric_instances]
            debug.log_intermediate("metric_resolution", "Resolved metrics", metrics=metric_names)

        # Stream evaluation results as they complete
        with tracer.start_span("test_execution_stream") as test_exec_span:
            # Import async helper
            from evaris._async_helpers import _run_single_test_async

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrency)

            completed_count = 0

            async def run_one_test(test_case: TestCase, index: int) -> tuple[int, TestResult]:
                async with semaphore:
                    try:
                        result = await _run_single_test_async(task, test_case, metric_instances)
                        return (index, result)
                    except Exception as e:
                        # If test execution fails, return a failed TestResult
                        # with the actual test case
                        tracer.record_exception(e)
                        debug.log_error("test_execution", e, test_index=index)
                        return (
                            index,
                            TestResult(
                                test_case=test_case,
                                output=None,
                                metrics=[],
                                latency_ms=0.0,
                                error=str(e),
                            ),
                        )

            # Create tasks for all tests
            tasks = [asyncio.create_task(run_one_test(tc, i)) for i, tc in enumerate(test_cases)]

            # Yield results as they complete
            for coro in asyncio.as_completed(tasks):
                index, test_result = await coro
                completed_count += 1

                # Add progress event
                if completed_count % 10 == 0 or completed_count == len(test_cases):
                    tracer.add_event(
                        "progress",
                        {"completed": completed_count, "total": len(test_cases)},
                    )

                yield test_result

            if test_exec_span:
                test_exec_span.set_attribute("tests_executed", completed_count)

        # Mark evaluation as successful
        if eval_span:
            tracer.set_status("ok")
