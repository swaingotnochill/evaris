"""Multimodal Contextual Precision metric.

Evaluates retrieval precision for multimodal RAG systems.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class MultimodalContextualPrecisionConfig(BaseModel):
    """Configuration for multimodal contextual precision metric."""

    provider: str = Field(default="openrouter")
    model: Optional[str] = Field(default=None)
    threshold: float = Field(default=0.5)
    context_key: str = Field(default="retrieval_context")
    temperature: float = Field(default=0.0)


class MultimodalContextualPrecisionMetric(BaseMetric):
    """Multimodal Contextual Precision metric.

    Evaluates if retrieved multimodal content is precisely relevant.
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[MultimodalContextualPrecisionConfig] = None):
        self.config = config or MultimodalContextualPrecisionConfig()
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
            raise ValueError("Multimodal contextual precision requires 'actual_output'")

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        context = test_case.metadata.get(self.config.context_key, []) if test_case.metadata else []
        expected = test_case.expected or ""

        prompt = f"""Evaluate contextual precision for multimodal RAG.

Query: {test_case.input}
Expected: {expected}
Retrieved Context: {context}

Respond with ONLY a JSON object:
{{"score": <0.0 to 1.0>, "reasoning": "brief explanation"}}

Your response:"""

        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        try:
            result = json.loads(response.content.strip())
            score = float(result.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            result = {"reasoning": "Failed to parse response"}

        passed = score >= self.threshold

        return MetricResult(
            name="multimodal_contextual_precision",
            score=score,
            passed=passed,
            metadata={"reasoning": result.get("reasoning", "")},
        )
