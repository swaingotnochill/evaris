"""Role Adherence metric for conversational evaluation.

This metric checks whether an LLM maintains its assigned role/persona
throughout the conversation.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class RoleAdherenceConfig(BaseModel):
    """Configuration for role adherence metric."""

    provider: str = Field(
        default="openrouter",
        description="LLM provider name",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use (uses provider default if not specified)",
    )
    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    role_key: str = Field(
        default="system_role",
        description="Metadata key containing the system role",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class RoleAdherenceMetric(BaseMetric):
    """Role Adherence metric for conversational evaluation.

    Evaluates whether the LLM maintains its assigned role/persona
    and stays in character throughout the response.

    Required: system_role in metadata (or custom role_key)

    Checks for:
    - Tone consistency with role
    - Appropriate vocabulary/language
    - Character/persona maintenance
    - Role boundary adherence
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[RoleAdherenceConfig] = None):
        self.config = config or RoleAdherenceConfig()
        self.threshold = self.config.threshold
        self._provider = None

    def _get_provider(self) -> Any:
        """Get or create the LLM provider."""
        if self._provider is None:
            self._provider = get_provider(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        return self._provider

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs for role adherence metric."""
        if not actual_output:
            raise ValueError("Role adherence metric requires 'actual_output'")
        role_key = self.config.role_key
        if not test_case.metadata or role_key not in test_case.metadata:
            raise ValueError(f"Role adherence metric requires '{role_key}' in metadata")

    def _build_prompt(
        self,
        system_role: str,
        user_input: str,
        actual_output: str,
    ) -> str:
        """Build the evaluation prompt."""
        return f"""Evaluate if the assistant's response adheres to its assigned role.

Assigned Role/Persona:
{system_role}

User Input: {user_input}

Assistant's Response: {actual_output}

Task:
Evaluate how well the response adheres to the assigned role. Consider:
1. Tone consistency - Does the tone match the role?
2. Vocabulary - Is the language appropriate for the role?
3. Character maintenance - Does it stay in character?
4. Role boundaries - Does it respect role limitations?

Respond with ONLY a JSON object in this format:
{{"score": <0.0 to 1.0>, "violations": ["violation 1", "violation 2", ...], "reasoning": "brief explanation"}}

Score guidelines:
- 1.0: Perfect role adherence
- 0.7-0.9: Minor deviations but maintains character
- 0.4-0.6: Noticeable breaks in character
- 0.0-0.3: Significant role violations

Your response:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"score": 0.0, "violations": [], "reasoning": "Failed to parse response"}

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure role adherence."""
        self.validate_inputs(test_case, actual_output)

        role_key = self.config.role_key
        system_role = test_case.metadata.get(role_key, "")  # type: ignore
        output_str = str(actual_output)

        prompt = self._build_prompt(system_role, test_case.input, output_str)
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        result_data = self._parse_response(response.content)

        score = result_data.get("score", 0.0)
        passed = score >= self.threshold

        return MetricResult(
            name="role_adherence",
            score=score,
            passed=passed,
            metadata={
                "violations": result_data.get("violations", []),
                "reasoning": result_data.get("reasoning", ""),
                "system_role": system_role,
            },
        )
