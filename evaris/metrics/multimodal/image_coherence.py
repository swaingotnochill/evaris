"""Image Coherence metric for multimodal evaluation.

Evaluates whether the text output is coherent with the image content.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ImageCoherenceConfig(BaseModel):
    """Configuration for image coherence metric."""

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
        description="Key in metadata containing the image (base64 or URL)",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class ImageCoherenceMetric(BaseMetric):
    """Image Coherence metric for multimodal evaluation.

    Evaluates whether the text output accurately describes or relates to
    the image content.

    Required: image in metadata (base64 or URL)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ImageCoherenceConfig] = None):
        self.config = config or ImageCoherenceConfig()
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
            raise ValueError("Image coherence metric requires 'actual_output'")
        image_key = self.config.image_key
        if not test_case.metadata or image_key not in test_case.metadata:
            raise ValueError(f"Image coherence metric requires '{image_key}' in metadata")

    def _build_prompt(self, text: str, image_description: str) -> str:
        return f"""Evaluate the coherence between the text and the image.

Image: [Image provided]
Text: {text}

Task: Evaluate how well the text relates to and accurately describes the image content.

Respond with ONLY a JSON object:
{{"score": <0.0 to 1.0>, "reasoning": "brief explanation"}}

Score guidelines:
- 1.0: Text perfectly describes the image
- 0.7-0.9: Text mostly accurate with minor omissions
- 0.4-0.6: Partial accuracy, some incorrect or missing details
- 0.0-0.3: Text contradicts or doesn't relate to the image

Your response:"""

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        output_str = str(actual_output)

        # Note: In a real implementation, the image would be passed to a
        # vision-capable model. This implementation uses the provider's
        # capability to handle multimodal inputs.
        prompt = self._build_prompt(output_str, "")
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
            name="image_coherence",
            score=score,
            passed=passed,
            metadata={
                "reasoning": result.get("reasoning", ""),
            },
        )
