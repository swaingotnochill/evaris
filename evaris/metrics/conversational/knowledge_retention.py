"""Knowledge Retention metric for conversational evaluation.

This metric checks whether an LLM retains and correctly uses information
that was provided in earlier conversation turns.

Formula: Retained Knowledge Points / Total Knowledge Points
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class KnowledgeRetentionConfig(BaseModel):
    """Configuration for knowledge retention metric."""

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
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class KnowledgeRetentionMetric(BaseMetric):
    """Knowledge Retention metric for conversational evaluation.

    Evaluates whether the LLM retains and correctly uses information
    from earlier in the conversation.

    Required: messages in metadata (conversation history)

    Algorithm:
    1. Extract knowledge points shared in the conversation
    2. Check if each knowledge point is correctly retained in the response
    3. Calculate: retained_count / total_count
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[KnowledgeRetentionConfig] = None):
        self.config = config or KnowledgeRetentionConfig()
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
        """Validate inputs for knowledge retention metric."""
        if not actual_output:
            raise ValueError("Knowledge retention metric requires 'actual_output'")
        if not test_case.metadata or "messages" not in test_case.metadata:
            raise ValueError(
                "Knowledge retention metric requires 'messages' in metadata "
                "(conversation history)"
            )

    def _format_conversation(self, messages: list[dict[str, str]]) -> str:
        """Format conversation history for prompt."""
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
        """Build the evaluation prompt."""
        return f"""Evaluate if the assistant's response correctly retains knowledge from the conversation.

Conversation History:
{conversation}

Current User Input: {current_input}

Assistant's Response: {actual_output}

Task:
1. Identify key knowledge points that were shared in the conversation
   (names, dates, preferences, facts, etc.)
2. Check if each knowledge point is correctly retained in the response

Respond with ONLY a JSON object in this format:
{{"knowledge_points": [
  {{"info": "specific information from conversation", "retained": "yes" or "no"}},
  ...
], "retained_count": <number>, "total_count": <number>}}

If no specific knowledge needs to be retained, return:
{{"knowledge_points": [], "retained_count": 0, "total_count": 0}}

Your response:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"knowledge_points": [], "retained_count": 0, "total_count": 0}

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure knowledge retention."""
        self.validate_inputs(test_case, actual_output)

        messages = test_case.metadata.get("messages", [])  # type: ignore
        conversation = self._format_conversation(messages)
        output_str = str(actual_output)

        prompt = self._build_prompt(conversation, test_case.input, output_str)
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        result_data = self._parse_response(response.content)

        total = result_data.get("total_count", 0)
        retained = result_data.get("retained_count", 0)

        # If no knowledge points, consider it a pass (nothing to retain)
        if total == 0:
            score = 1.0
        else:
            score = retained / total

        passed = score >= self.threshold

        return MetricResult(
            name="knowledge_retention",
            score=score,
            passed=passed,
            metadata={
                "knowledge_points": result_data.get("knowledge_points", []),
                "retained_count": retained,
                "total_count": total,
            },
        )
