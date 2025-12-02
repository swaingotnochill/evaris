"""Multimodal Answer Relevancy metric.

Evaluates whether an answer is relevant to a multimodal query (text + image).
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class MultimodalAnswerRelevancyConfig(BaseModel):
    """Configuration for multimodal answer relevancy metric."""

    provider: str = Field(
        default="openrouter",
        description="LLM provider name (must support vision)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use (must be vision-capable)",
    )
    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    image_key: str = Field(
        default="image",
        description="Key in metadata containing the image",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature",
    )


class MultimodalAnswerRelevancyMetric(BaseMetric):
    """Multimodal Answer Relevancy metric.

    Evaluates if the answer correctly addresses a query about an image.

    Required: image in metadata
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[MultimodalAnswerRelevancyConfig] = None):
        self.config = config or MultimodalAnswerRelevancyConfig()
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
            raise ValueError("Multimodal answer relevancy metric requires 'actual_output'")
        image_key = self.config.image_key
        if not test_case.metadata or image_key not in test_case.metadata:
            raise ValueError(
                f"Multimodal answer relevancy metric requires '{image_key}' in metadata"
            )

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        prompt = f"""Evaluate if the answer is relevant to the question about the image.

Question about image: {test_case.input}
Answer: {actual_output}

Task: Evaluate if the answer correctly and relevantly addresses the question
considering the image content.

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
            name="multimodal_answer_relevancy",
            score=score,
            passed=passed,
            metadata={"reasoning": result.get("reasoning", "")},
        )
