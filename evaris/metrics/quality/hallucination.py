"""Hallucination detection metric for LLM outputs.

This metric evaluates whether the output contains factual hallucinations -
claims that are not supported by the provided context or are factually incorrect.

Different from Faithfulness: Hallucination checks against known facts,
while Faithfulness checks against provided context.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class HallucinationConfig(BaseModel):
    """Configuration for hallucination detection metric."""

    provider: str = Field(default="openrouter", description="LLM provider name")
    model: Optional[str] = Field(default=None, description="Model to use")
    threshold: float = Field(
        default=0.5,
        description="Maximum acceptable hallucination score. Lower is better.",
    )
    context_key: str = Field(
        default="context",
        description="Key in metadata containing factual context",
    )
    temperature: float = Field(default=0.0, description="LLM temperature")


class HallucinationMetric(BaseMetric):
    """Hallucination detection metric.

    Detects factual hallucinations in LLM output - claims that are
    not supported by facts or context.

    Note: Score represents proportion of hallucinated claims.
    Lower score = less hallucination = better.
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[HallucinationConfig] = None):
        self.config = config or HallucinationConfig()
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
            raise ValueError("Hallucination metric requires 'actual_output'")

    def _build_detection_prompt(
        self,
        actual_output: str,
        context: Optional[str] = None,
    ) -> str:
        context_section = f"\nFactual Context:\n{context}" if context else ""

        return f"""Analyze the following text for factual hallucinations.
A hallucination is a claim that is:
- Not supported by the provided context (if any)
- Factually incorrect or made up
- A fabricated statistic, date, or specific detail
{context_section}

Text to analyze:
{actual_output}

Respond with ONLY a JSON object:
{{"claims": [
  {{"claim": "claim text", "hallucinated": "yes" or "no", "reason": "explanation"}},
  ...
], "hallucination_count": <int>, "total_claims": <int>}}

Your response:"""

    def _parse_result(self, response: str) -> dict[str, Any]:
        try:
            data = json.loads(response.strip())
            return {
                "claims": data.get("claims", []),
                "hallucination_count": data.get("hallucination_count", 0),
                "total_claims": data.get("total_claims", 0),
            }
        except json.JSONDecodeError:
            return {"claims": [], "hallucination_count": 0, "total_claims": 0}

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        output_str = str(actual_output)

        context = None
        if test_case.metadata:
            context = test_case.metadata.get(self.config.context_key)

        prompt = self._build_detection_prompt(output_str, context)
        response = await provider.a_complete(prompt)
        result = self._parse_result(response.content)

        total = result["total_claims"]
        hallucinated = result["hallucination_count"]

        score = hallucinated / total if total > 0 else 0.0
        passed = score <= self.threshold

        return MetricResult(
            name="hallucination",
            score=score,
            passed=passed,
            metadata={
                "claims": result["claims"],
                "hallucination_count": hallucinated,
                "total_claims": total,
            },
        )
