"""Multimodal Faithfulness metric.

Evaluates whether the answer is faithful to the multimodal context.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class MultimodalFaithfulnessConfig(BaseModel):
    """Configuration for multimodal faithfulness metric."""

    provider: str = Field(default="openrouter")
    model: Optional[str] = Field(default=None)
    threshold: float = Field(default=0.5)
    image_key: str = Field(default="image")
    context_key: str = Field(default="context")
    temperature: float = Field(default=0.0)


class MultimodalFaithfulnessMetric(BaseMetric):
    """Multimodal Faithfulness metric.

    Evaluates if the answer is faithful to both the image and text context.
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[MultimodalFaithfulnessConfig] = None):
        self.config = config or MultimodalFaithfulnessConfig()
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
            raise ValueError("Multimodal faithfulness metric requires 'actual_output'")

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        context = test_case.metadata.get(self.config.context_key, "") if test_case.metadata else ""

        prompt = f"""Evaluate the faithfulness of the answer to the provided context.

Context: {context}
Query: {test_case.input}
Answer: {actual_output}

Task: Evaluate if the answer is faithful to the context and image.

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
            name="multimodal_faithfulness",
            score=score,
            passed=passed,
            metadata={"reasoning": result.get("reasoning", "")},
        )
