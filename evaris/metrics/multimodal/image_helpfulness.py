"""Image Helpfulness metric for multimodal evaluation.

Evaluates whether an image is helpful for answering a user's question.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ImageHelpfulnessConfig(BaseModel):
    """Configuration for image helpfulness metric."""

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


class ImageHelpfulnessMetric(BaseMetric):
    """Image Helpfulness metric for multimodal evaluation.

    Evaluates whether the image is helpful for the given context/question.

    Required: image in metadata
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ImageHelpfulnessConfig] = None):
        self.config = config or ImageHelpfulnessConfig()
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
            raise ValueError("Image helpfulness metric requires 'actual_output'")
        image_key = self.config.image_key
        if not test_case.metadata or image_key not in test_case.metadata:
            raise ValueError(f"Image helpfulness metric requires '{image_key}' in metadata")

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        prompt = f"""Evaluate how helpful the image is for answering the user's query.

User Query: {test_case.input}
Response with Image: {actual_output}

Task: Evaluate if the image helps understand or answer the query.

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
            name="image_helpfulness",
            score=score,
            passed=passed,
            metadata={"reasoning": result.get("reasoning", "")},
        )
