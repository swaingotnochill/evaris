"""Exact match metric for evaluation."""

import asyncio
from typing import Any

from evaris.types import BaseMetric, MetricResult, TestCase


class ExactMatchMetric(BaseMetric):
    """Metric that checks for exact match between expected and actual output.

    Attributes:
        case_sensitive: Whether to perform case-sensitive comparison (default: True)
        strip_whitespace: Whether to strip leading/trailing whitespace (default: False)
    """

    def __init__(self, case_sensitive: bool = True, strip_whitespace: bool = False) -> None:
        """Initialize the ExactMatchMetric.

        Args:
            case_sensitive: Whether to perform case-sensitive comparison
            strip_whitespace: Whether to strip leading/trailing whitespace before comparison
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score the actual output against the expected output.

        Args:
            test_case: The test case containing the expected output
            actual_output: The actual output from the agent

        Returns:
            MetricResult with score 1.0 if exact match, 0.0 otherwise

        Raises:
            ValueError: If test_case.expected is None
        """
        if test_case.expected is None:
            raise ValueError(
                "ExactMatchMetric requires test case to have an 'expected' value. "
                f"Test case input: {test_case.input}"
            )

        expected = test_case.expected
        actual = actual_output

        # Normalize strings if needed
        if isinstance(expected, str) and isinstance(actual, str):
            if self.strip_whitespace:
                expected = expected.strip()
                actual = actual.strip()

            if not self.case_sensitive:
                expected = expected.lower()
                actual = actual.lower()

        # Perform comparison
        is_match = expected == actual

        return MetricResult(
            name="exact_match",
            score=1.0 if is_match else 0.0,
            passed=is_match,
            metadata={
                "expected": test_case.expected,
                "actual": actual_output,
                "case_sensitive": self.case_sensitive,
                "strip_whitespace": self.strip_whitespace,
            },
        )

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously score the actual output against the expected output.

        Since exact matching is CPU-bound and fast, this runs the sync version
        in a thread pool to avoid blocking the event loop.

        Args:
            test_case: The test case containing expected and actual_output

        Returns:
            MetricResult with score 1.0 if exact match, 0.0 otherwise

        Raises:
            ValueError: If test_case.expected or actual_output is None
        """
        if test_case.actual_output is None:
            raise ValueError("ExactMatchMetric requires test case to have 'actual_output'")

        # For simple CPU-bound operations, run in thread pool
        return await asyncio.to_thread(self.score, test_case, test_case.actual_output)
