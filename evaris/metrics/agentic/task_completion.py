"""Task Completion metric for agent evaluation.

This metric evaluates how well an agent completed its intended task.
Uses LLM-as-judge to assess task completion based on input, output,
and optional trace data.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class TaskCompletionConfig(BaseModel):
    """Configuration for task completion metric."""

    provider: str = Field(
        default="openrouter",
        description="LLM provider name",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use (uses provider default if not specified)",
    )
    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    task_key: str = Field(
        default="task",
        description="Key in metadata containing task description (optional)",
    )
    trace_key: str = Field(
        default="trace",
        description="Key in metadata containing agent trace (optional)",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class TaskCompletionMetric(BaseMetric):
    """Task Completion metric for agent evaluation.

    Evaluates how effectively an agent accomplished its intended goal.

    Optional metadata:
    - task: Explicit task description (inferred from input if missing)
    - trace: Agent execution trace for detailed analysis

    Algorithm:
    1. Extract/infer task from input or metadata
    2. Analyze output (and trace if available) against task
    3. Score based on task-outcome alignment

    Example:
        >>> metric = TaskCompletionMetric()
        >>> test_case = TestCase(
        ...     input="Book a flight to NYC",
        ...     metadata={"task": "Book a flight"}
        ... )
        >>> result = await metric.a_measure(test_case, "Flight booked!")
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[TaskCompletionConfig] = None):
        """Initialize task completion metric."""
        self.config = config or TaskCompletionConfig()
        self.threshold = self.config.threshold
        self._provider = None

    def _get_provider(self) -> Any:
        """Get or create the LLM provider."""
        if self._provider is None:
            self._provider = get_provider(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        return self._provider

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs."""
        if not actual_output:
            raise ValueError("Task completion metric requires 'actual_output'")

    def _build_evaluation_prompt(
        self,
        input_query: str,
        actual_output: str,
        task: Optional[str] = None,
        trace: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Build prompt for task completion evaluation."""
        task_desc = task if task else f"Complete the following request: {input_query}"

        trace_section = ""
        if trace:
            trace_str = json.dumps(trace, indent=2)
            trace_section = f"""
Agent Execution Trace:
{trace_str}
"""

        return f"""Evaluate how well the agent completed its task.

Task: {task_desc}

User Input: {input_query}
{trace_section}
Agent Output:
{actual_output}

Respond with ONLY a JSON object in this format:
{{"score": <float 0.0-1.0>, "task_completed": true or false, "reasoning": "explanation"}}

Scoring guidelines:
- 1.0: Task fully and correctly completed
- 0.7-0.9: Task mostly completed with minor issues
- 0.4-0.6: Task partially completed
- 0.1-0.3: Task attempted but largely failed
- 0.0: Task not completed at all

Your response:"""

    def _parse_evaluation(self, response: str) -> dict[str, Any]:
        """Parse evaluation response."""
        try:
            data = json.loads(response.strip())
            return {
                "score": float(data.get("score", 0.0)),
                "task_completed": data.get("task_completed", False),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "score": 0.0,
                "task_completed": False,
                "reasoning": "Failed to parse evaluation",
            }

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure task completion.

        Args:
            test_case: Test case with input and optional task/trace
            actual_output: Agent's response

        Returns:
            MetricResult with completion score
        """
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        input_str = str(test_case.input) if test_case.input else ""
        output_str = str(actual_output)

        # Get optional task and trace
        task = None
        trace = None
        if test_case.metadata:
            task = test_case.metadata.get(self.config.task_key)
            trace = test_case.metadata.get(self.config.trace_key)

        # Evaluate task completion
        eval_prompt = self._build_evaluation_prompt(input_str, output_str, task, trace)
        eval_response = await provider.a_complete(eval_prompt)
        result = self._parse_evaluation(eval_response.content)

        score = max(0.0, min(1.0, result["score"]))
        passed = score >= self.threshold

        return MetricResult(
            name="task_completion",
            score=score,
            passed=passed,
            metadata={
                "task_completed": result["task_completed"],
                "reasoning": result["reasoning"],
                "task": task or f"Inferred from: {input_str[:100]}",
                "had_trace": trace is not None,
            },
        )
