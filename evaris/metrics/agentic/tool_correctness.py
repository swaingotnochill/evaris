"""Tool Correctness metric for agent evaluation.

This metric evaluates whether an agent used the correct tools to complete a task.

Formula: Number of Correctly Used Tools / Total Tools Called

Compares actual tool calls against expected tools.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase


class ToolCorrectnessConfig(BaseModel):
    """Configuration for tool correctness metric."""

    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    tools_called_key: str = Field(
        default="tools_called",
        description="Key in metadata containing list of tools actually called",
    )
    expected_tools_key: str = Field(
        default="expected_tools",
        description="Key in metadata containing list of expected tools",
    )
    check_order: bool = Field(
        default=False,
        description="Whether to check if tools were called in correct order",
    )


class ToolCorrectnessMetric(BaseMetric):
    """Tool Correctness metric for agent evaluation.

    Measures how accurately an agent selected and used tools.

    Required metadata:
    - tools_called: List of tools the agent actually invoked
    - expected_tools: List of tools that should have been used

    Algorithm:
    1. Compare tools_called against expected_tools
    2. Calculate: correct_tools / total_called
    3. Optionally check order if configured

    Example:
        >>> metric = ToolCorrectnessMetric()
        >>> test_case = TestCase(
        ...     input="Get weather for NYC",
        ...     metadata={
        ...         "tools_called": ["get_weather"],
        ...         "expected_tools": ["get_weather"]
        ...     }
        ... )
        >>> result = await metric.a_measure(test_case, "Weather is sunny")
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ToolCorrectnessConfig] = None):
        """Initialize tool correctness metric."""
        self.config = config or ToolCorrectnessConfig()
        self.threshold = self.config.threshold

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs.

        Args:
            test_case: Must contain tools_called and expected_tools in metadata
            actual_output: Agent's response

        Raises:
            ValueError: If required metadata is missing
        """
        if not test_case.metadata:
            raise ValueError(
                "Tool correctness metric requires metadata with 'tools_called' and 'expected_tools'"
            )

        tools_key = self.config.tools_called_key
        expected_key = self.config.expected_tools_key

        if tools_key not in test_case.metadata:
            raise ValueError(
                f"Tool correctness metric requires '{tools_key}' in test_case.metadata"
            )

        if expected_key not in test_case.metadata:
            raise ValueError(
                f"Tool correctness metric requires '{expected_key}' in test_case.metadata"
            )

    def _calculate_correctness(
        self,
        tools_called: list[str],
        expected_tools: list[str],
    ) -> float:
        """Calculate tool correctness score.

        Args:
            tools_called: List of tools actually called
            expected_tools: List of expected tools

        Returns:
            Score between 0 and 1
        """
        if not tools_called:
            return 0.0

        expected_set = set(expected_tools)
        correct_count = sum(1 for tool in tools_called if tool in expected_set)

        return correct_count / len(tools_called)

    def _calculate_ordered_correctness(
        self,
        tools_called: list[str],
        expected_tools: list[str],
    ) -> float:
        """Calculate tool correctness with order consideration.

        Uses longest common subsequence approach.

        Args:
            tools_called: List of tools actually called
            expected_tools: List of expected tools in correct order

        Returns:
            Score between 0 and 1
        """
        if not tools_called or not expected_tools:
            return 0.0

        # LCS dynamic programming
        m, n = len(tools_called), len(expected_tools)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tools_called[i - 1] == expected_tools[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return lcs_length / max(len(tools_called), len(expected_tools))

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure tool correctness.

        Args:
            test_case: Test case with tool usage metadata
            actual_output: Agent's response

        Returns:
            MetricResult with correctness score
        """
        self.validate_inputs(test_case, actual_output)

        tools_key = self.config.tools_called_key
        expected_key = self.config.expected_tools_key

        tools_called = test_case.metadata[tools_key]
        expected_tools = test_case.metadata[expected_key]

        # Calculate score
        if self.config.check_order:
            score = self._calculate_ordered_correctness(tools_called, expected_tools)
        else:
            score = self._calculate_correctness(tools_called, expected_tools)

        passed = score >= self.threshold

        # Check for extra/missing tools
        called_set = set(tools_called)
        expected_set = set(expected_tools)

        return MetricResult(
            name="tool_correctness",
            score=score,
            passed=passed,
            metadata={
                "tools_called": tools_called,
                "expected_tools": expected_tools,
                "correct_tools": list(called_set & expected_set),
                "extra_tools": list(called_set - expected_set),
                "missing_tools": list(expected_set - called_set),
                "check_order": self.config.check_order,
            },
        )
