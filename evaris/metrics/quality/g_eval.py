"""G-Eval metric for custom criteria evaluation.

G-Eval allows evaluation based on custom criteria defined by the user.
It uses chain-of-thought prompting to generate evaluation steps
and then scores the output based on those criteria.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class GEvalConfig(BaseModel):
    """Configuration for G-Eval metric."""

    provider: str = Field(default="openrouter", description="LLM provider name")
    model: Optional[str] = Field(default=None, description="Model to use")
    threshold: float = Field(default=0.5, description="Score threshold for passing")
    criteria: str = Field(
        default="",
        description="Evaluation criteria (what to evaluate for)",
    )
    evaluation_steps: Optional[list[str]] = Field(
        default=None,
        description="Optional explicit evaluation steps",
    )
    temperature: float = Field(default=0.0, description="LLM temperature")


class GEvalMetric(BaseMetric):
    """G-Eval metric for custom criteria evaluation.

    G-Eval (Generative Evaluation) uses LLM to evaluate outputs
    based on user-defined criteria. It can:
    1. Generate evaluation steps from criteria
    2. Apply those steps to evaluate the output
    3. Return a score with reasoning

    Example:
        >>> metric = GEvalMetric(config=GEvalConfig(
        ...     criteria="Evaluate for politeness and helpfulness"
        ... ))
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[GEvalConfig] = None):
        self.config = config or GEvalConfig()
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
            raise ValueError("G-Eval metric requires 'actual_output'")

        if not self.config.criteria:
            raise ValueError("G-Eval metric requires 'criteria' in config")

    def _build_evaluation_prompt(
        self,
        input_query: str,
        actual_output: str,
        criteria: str,
        evaluation_steps: Optional[list[str]] = None,
    ) -> str:
        steps_section = ""
        if evaluation_steps:
            steps_list = "\n".join(f"{i+1}. {step}" for i, step in enumerate(evaluation_steps))
            steps_section = f"""
Evaluation Steps:
{steps_list}
"""

        return f"""Evaluate the response based on the given criteria.

Criteria: {criteria}
{steps_section}
Input: {input_query}

Response to evaluate:
{actual_output}

Evaluate the response and provide a score from 0 to 1.
Think through each aspect of the criteria step by step.

Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "reasoning": "step-by-step explanation", "criteria_met": ["list of criteria met"], "criteria_not_met": ["list of criteria not met"]}}

Your response:"""

    def _parse_result(self, response: str) -> dict[str, Any]:
        try:
            data = json.loads(response.strip())
            return {
                "score": float(data.get("score", 0)),
                "reasoning": data.get("reasoning", ""),
                "criteria_met": data.get("criteria_met", []),
                "criteria_not_met": data.get("criteria_not_met", []),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "score": 0.0,
                "reasoning": "Failed to parse evaluation",
                "criteria_met": [],
                "criteria_not_met": [],
            }

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        input_str = str(test_case.input) if test_case.input else ""
        output_str = str(actual_output)

        prompt = self._build_evaluation_prompt(
            input_str,
            output_str,
            self.config.criteria,
            self.config.evaluation_steps,
        )
        response = await provider.a_complete(prompt)
        result = self._parse_result(response.content)

        score = max(0.0, min(1.0, result["score"]))
        passed = score >= self.threshold

        return MetricResult(
            name="g_eval",
            score=score,
            passed=passed,
            metadata={
                "criteria": self.config.criteria,
                "reasoning": result["reasoning"],
                "criteria_met": result["criteria_met"],
                "criteria_not_met": result["criteria_not_met"],
            },
        )
