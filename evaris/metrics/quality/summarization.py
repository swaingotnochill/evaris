"""Summarization quality metric for LLM outputs.

This metric evaluates the quality of generated summaries
based on coverage, conciseness, and accuracy.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class SummarizationConfig(BaseModel):
    """Configuration for summarization metric."""

    provider: str = Field(default="openrouter", description="LLM provider name")
    model: Optional[str] = Field(default=None, description="Model to use")
    threshold: float = Field(default=0.5, description="Score threshold for passing")
    source_key: str = Field(
        default="source_text",
        description="Key in metadata containing source text to summarize",
    )
    temperature: float = Field(default=0.0, description="LLM temperature")


class SummarizationMetric(BaseMetric):
    """Summarization quality metric.

    Evaluates summaries based on:
    - Coverage: Are key points from the source included?
    - Conciseness: Is the summary appropriately brief?
    - Accuracy: Are the statements in the summary correct?

    Requires: source_text in metadata (the original text)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
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
            raise ValueError("Summarization metric requires 'actual_output'")

        if not test_case.metadata or self.config.source_key not in test_case.metadata:
            raise ValueError(
                f"Summarization metric requires '{self.config.source_key}' in metadata"
            )

    def _build_evaluation_prompt(self, source_text: str, summary: str) -> str:
        return f"""Evaluate the quality of this summary.

Source Text:
{source_text}

Summary:
{summary}

Evaluate on these criteria:
1. Coverage (0-1): Does the summary include the key points?
2. Conciseness (0-1): Is it appropriately brief without losing meaning?
3. Accuracy (0-1): Are all statements in the summary factually correct?

Respond with ONLY a JSON object:
{{"coverage": <float>, "conciseness": <float>, "accuracy": <float>, "overall_score": <float 0-1>, "reasoning": "explanation"}}

Your response:"""

    def _parse_result(self, response: str) -> dict[str, Any]:
        try:
            data = json.loads(response.strip())
            return {
                "coverage": float(data.get("coverage", 0)),
                "conciseness": float(data.get("conciseness", 0)),
                "accuracy": float(data.get("accuracy", 0)),
                "overall_score": float(data.get("overall_score", 0)),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "coverage": 0.0,
                "conciseness": 0.0,
                "accuracy": 0.0,
                "overall_score": 0.0,
                "reasoning": "Failed to parse evaluation",
            }

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        source_text = test_case.metadata[self.config.source_key]
        summary = str(actual_output)

        prompt = self._build_evaluation_prompt(source_text, summary)
        response = await provider.a_complete(prompt)
        result = self._parse_result(response.content)

        score = result["overall_score"]
        passed = score >= self.threshold

        return MetricResult(
            name="summarization",
            score=score,
            passed=passed,
            metadata={
                "coverage": result["coverage"],
                "conciseness": result["conciseness"],
                "accuracy": result["accuracy"],
                "reasoning": result["reasoning"],
            },
        )
