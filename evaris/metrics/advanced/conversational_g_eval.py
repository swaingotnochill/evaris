"""Conversational G-Eval metric for multi-turn conversation evaluation.

G-Eval adapted for evaluating multi-turn conversations with custom criteria.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ConversationalGEvalConfig(BaseModel):
    """Configuration for conversational G-Eval metric."""

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
    criteria: str = Field(
        default="coherence, helpfulness, accuracy",
        description="Evaluation criteria for the conversation",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature",
    )


class ConversationalGEvalMetric(BaseMetric):
    """Conversational G-Eval metric for multi-turn conversations.

    Evaluates multi-turn conversations based on custom criteria,
    considering the full conversation context.

    Required: messages in metadata (conversation history)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ConversationalGEvalConfig] = None):
        self.config = config or ConversationalGEvalConfig()
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
            raise ValueError("Conversational G-Eval metric requires 'actual_output'")
        if not test_case.metadata or "messages" not in test_case.metadata:
            raise ValueError("Conversational G-Eval metric requires 'messages' in metadata")

    def _format_conversation(self, messages: list[dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        conversation: str,
        current_input: str,
        actual_output: str,
    ) -> str:
        return f"""Evaluate this conversation response based on: {self.config.criteria}

Conversation History:
{conversation}

Current Input: {current_input}
Response: {actual_output}

Respond with ONLY a JSON object:
{{"score": <0.0 to 1.0>, "criteria_scores": {{"criterion1": 0.X, ...}}, "reasoning": "brief explanation"}}

Your response:"""

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        messages = test_case.metadata.get("messages", [])  # type: ignore
        conversation = self._format_conversation(messages)

        prompt = self._build_prompt(conversation, test_case.input, str(actual_output))
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        try:
            result = json.loads(response.content.strip())
            score = float(result.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            result = {"reasoning": "Failed to parse response", "criteria_scores": {}}

        passed = score >= self.threshold

        return MetricResult(
            name="conversational_g_eval",
            score=score,
            passed=passed,
            metadata={
                "criteria_scores": result.get("criteria_scores", {}),
                "reasoning": result.get("reasoning", ""),
                "criteria": self.config.criteria,
            },
        )
