"""Text-to-Image metric for evaluating image generation quality.

Evaluates how well a generated image matches the text prompt.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class TextToImageConfig(BaseModel):
    """Configuration for text-to-image metric."""

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
        default="generated_image",
        description="Key in metadata containing the generated image",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature",
    )


class TextToImageMetric(BaseMetric):
    """Text-to-Image metric for image generation evaluation.

    Evaluates:
    - Prompt adherence: Does the image match the text prompt?
    - Quality: Is the image visually coherent?
    - Details: Are requested elements present?

    Required: generated_image in metadata
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[TextToImageConfig] = None):
        self.config = config or TextToImageConfig()
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
            raise ValueError("Text-to-image metric requires 'actual_output'")
        image_key = self.config.image_key
        if not test_case.metadata or image_key not in test_case.metadata:
            raise ValueError(f"Text-to-image metric requires '{image_key}' in metadata")

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        prompt = f"""Evaluate the quality of the generated image against the prompt.

Image Generation Prompt: {test_case.input}

Task: Evaluate how well the generated image matches the prompt.

Respond with ONLY a JSON object:
{{"score": <0.0 to 1.0>, "prompt_adherence": <0.0 to 1.0>, "quality": <0.0 to 1.0>, "reasoning": "brief explanation"}}

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
            name="text_to_image",
            score=score,
            passed=passed,
            metadata={
                "prompt_adherence": result.get("prompt_adherence", 0.0),
                "quality": result.get("quality", 0.0),
                "reasoning": result.get("reasoning", ""),
            },
        )
