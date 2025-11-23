"""Exact match metric for evaluation."""

import asyncio
from typing import Any

from evaris.types import BaseMetric, MetricResult, ReasoningStep, TestCase


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

    def _generate_reasoning_steps(
        self,
        expected: Any,
        actual: Any,
        expected_normalized: Any,
        actual_normalized: Any,
        is_match: bool,
    ) -> list[ReasoningStep]:
        """Generate reasoning steps for the exact match comparison.

        Args:
            expected: Original expected value
            actual: Original actual value
            expected_normalized: Normalized expected value
            actual_normalized: Normalized actual value
            is_match: Whether the values match

        Returns:
            List of reasoning steps
        """
        steps: list[ReasoningStep] = []
        step_num = 1

        # Check if normalization was configured
        is_string = isinstance(expected, str) and isinstance(actual, str)
        normalization_configured = self.strip_whitespace or not self.case_sensitive
        should_show_normalization = is_string and normalization_configured

        if should_show_normalization:
            # Step 1: Normalization
            norm_description_parts = []
            if self.strip_whitespace:
                norm_description_parts.append("whitespace stripping")
            if not self.case_sensitive:
                norm_description_parts.append("case normalization")

            steps.append(
                ReasoningStep(
                    step_number=step_num,
                    operation="text_normalization",
                    description=f"Normalizing text ({', '.join(norm_description_parts)})",
                    inputs={
                        "expected_original": expected,
                        "actual_original": actual,
                        "case_sensitive": self.case_sensitive,
                        "strip_whitespace": self.strip_whitespace,
                    },
                    outputs={
                        "expected_normalized": expected_normalized,
                        "actual_normalized": actual_normalized,
                    },
                    metadata={
                        "normalization_type": norm_description_parts,
                    },
                )
            )
            step_num += 1

        # Step 2 (or 1 if no normalization): Final comparison
        comparison_desc = (
            "Comparing normalized values"
            if should_show_normalization
            else "Comparing values directly"
        )

        steps.append(
            ReasoningStep(
                step_number=step_num,
                operation="comparison",
                description=comparison_desc,
                inputs={
                    "expected": expected_normalized,
                    "actual": actual_normalized,
                },
                outputs={
                    "match": is_match,
                    "score": 1.0 if is_match else 0.0,
                },
                metadata={
                    "comparison_type": "exact_match",
                    "data_type": type(expected).__name__,
                },
            )
        )

        return steps

    def _generate_reasoning_summary(
        self,
        expected: Any,
        actual: Any,
        is_match: bool,
    ) -> str:
        """Generate a human-readable reasoning summary.

        Args:
            expected: Expected value
            actual: Actual value
            is_match: Whether the values match

        Returns:
            Human-readable reasoning summary
        """
        if is_match:
            if self.strip_whitespace or not self.case_sensitive:
                norm_parts = []
                if self.strip_whitespace:
                    norm_parts.append("whitespace stripping")
                if not self.case_sensitive:
                    norm_parts.append("case normalization")
                return f"Exact match after {' and '.join(norm_parts)}"
            else:
                return "Exact match (values are identical)"
        else:
            # Mismatch
            expected_str = str(expected)[:50]  # Truncate for readability
            actual_str = str(actual)[:50]
            return f"Mismatch: expected '{expected_str}' but got '{actual_str}'"

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

        # Keep original values for reasoning
        expected_original = expected
        actual_original = actual

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

        # Generate reasoning
        reasoning_steps = self._generate_reasoning_steps(
            expected_original,
            actual_original,
            expected,
            actual,
            is_match,
        )
        reasoning_summary = self._generate_reasoning_summary(
            expected_original,
            actual_original,
            is_match,
        )

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
            reasoning=reasoning_summary,
            reasoning_steps=reasoning_steps,
            reasoning_type="logic",
        )

    async def a_measure(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Asynchronously score the actual output against the expected output.

        Since exact matching is CPU-bound and fast, this runs the sync version
        in a thread pool to avoid blocking the event loop.

        Args:
            test_case: The test case containing the expected output
            actual_output: The actual output from the agent

        Returns:
            MetricResult with score 1.0 if exact match, 0.0 otherwise

        Raises:
            ValueError: If test_case.expected is None
        """
        # For simple CPU-bound operations, run in thread pool
        return await asyncio.to_thread(self.score, test_case, actual_output)
