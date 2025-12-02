"""Arena G-Eval metric for pairwise comparison evaluation.

Compares two responses and determines which is better based on custom criteria.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ArenaGEvalConfig(BaseModel):
    """Configuration for Arena G-Eval metric."""

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
    comparison_key: str = Field(
        default="comparison_output",
        description="Key in metadata containing the comparison output",
    )
    criteria: str = Field(
        default="helpfulness, accuracy, clarity",
        description="Evaluation criteria for comparison",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature",
    )


class ArenaGEvalMetric(BaseMetric):
    """Arena G-Eval metric for pairwise comparison.

    Compares two responses (A: actual_output, B: comparison_output)
    and scores based on how much better A is compared to B.

    Required: comparison_output in metadata
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ArenaGEvalConfig] = None):
        self.config = config or ArenaGEvalConfig()
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
            raise ValueError("Arena G-Eval metric requires 'actual_output'")
        comparison_key = self.config.comparison_key
        if not test_case.metadata or comparison_key not in test_case.metadata:
            raise ValueError(f"Arena G-Eval metric requires '{comparison_key}' in metadata")

    def _build_prompt(
        self,
        input_text: str,
        response_a: str,
        response_b: str,
    ) -> str:
        return f"""Compare these two responses based on: {self.config.criteria}

Query: {input_text}

Response A:
{response_a}

Response B:
{response_b}

Task: Determine which response is better and by how much.

Respond with ONLY a JSON object:
{{"winner": "A" or "B" or "tie", "score": <0.0 to 1.0 representing how good A is>, "reasoning": "brief explanation"}}

Score interpretation:
- 1.0: A is significantly better
- 0.75: A is somewhat better
- 0.5: Tie
- 0.25: B is somewhat better
- 0.0: B is significantly better

Your response:"""

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        comparison_key = self.config.comparison_key
        response_b = test_case.metadata.get(comparison_key, "")  # type: ignore

        prompt = self._build_prompt(test_case.input, str(actual_output), response_b)
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        try:
            result = json.loads(response.content.strip())
            score = float(result.get("score", 0.5))
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            result = {"reasoning": "Failed to parse response", "winner": "tie"}

        passed = score >= self.threshold

        return MetricResult(
            name="arena_g_eval",
            score=score,
            passed=passed,
            metadata={
                "winner": result.get("winner", "tie"),
                "reasoning": result.get("reasoning", ""),
                "criteria": self.config.criteria,
            },
        )
