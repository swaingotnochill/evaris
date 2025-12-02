"""DAG (Deep Acyclic Graph) metric for multi-criteria evaluation.

The DAG metric evaluates outputs based on a directed acyclic graph of
evaluation criteria, where each node represents a criterion and edges
represent dependencies.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class DAGConfig(BaseModel):
    """Configuration for DAG metric."""

    provider: str = Field(
        default="openrouter",
        description="LLM provider name",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use",
    )
    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    evaluation_nodes: list[str] = Field(
        default_factory=lambda: ["clarity", "accuracy", "completeness"],
        description="List of evaluation criteria nodes",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature",
    )


class DAGMetric(BaseMetric):
    """DAG (Deep Acyclic Graph) metric for multi-criteria evaluation.

    Evaluates outputs based on multiple interconnected criteria,
    providing both individual node scores and an aggregated score.
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[DAGConfig] = None):
        self.config = config or DAGConfig()
        self.threshold = self.config.threshold
        self._provider = None

    def _get_provider(self) -> Any:
        if self._provider is None:
            self._provider = get_provider(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        return self._provider

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        if not actual_output:
            raise ValueError("DAG metric requires 'actual_output'")

    def _build_prompt(self, input_text: str, output: str) -> str:
        nodes = ", ".join(self.config.evaluation_nodes)
        return f"""Evaluate the output based on the following criteria: {nodes}

Input: {input_text}
Output: {output}

For each criterion, provide a score from 0.0 to 1.0.

Respond with ONLY a JSON object:
{{"score": <overall 0.0 to 1.0>, "node_scores": {{"criterion1": 0.X, "criterion2": 0.Y, ...}}, "reasoning": "brief explanation"}}

Your response:"""

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        prompt = self._build_prompt(test_case.input, str(actual_output))
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        try:
            result = json.loads(response.content.strip())
            score = float(result.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            result = {"reasoning": "Failed to parse response", "node_scores": {}}

        passed = score >= self.threshold

        return MetricResult(
            name="dag",
            score=score,
            passed=passed,
            metadata={
                "node_scores": result.get("node_scores", {}),
                "reasoning": result.get("reasoning", ""),
                "evaluation_nodes": self.config.evaluation_nodes,
            },
        )
