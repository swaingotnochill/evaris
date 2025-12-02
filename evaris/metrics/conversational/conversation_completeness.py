"""Conversation Completeness metric for conversational evaluation.

This metric checks whether all topics/requests raised in the conversation
have been adequately addressed and resolved.

Formula: Resolved Topics / Total Topics
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ConversationCompletenessConfig(BaseModel):
    """Configuration for conversation completeness metric."""

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


class ConversationCompletenessMetric(BaseMetric):
    """Conversation Completeness metric for conversational evaluation.

    Evaluates whether all topics and requests raised in the conversation
    have been adequately addressed.

    Required: messages in metadata (conversation history)

    Algorithm:
    1. Extract all topics/requests from the conversation
    2. Check if each topic has been adequately resolved
    3. Calculate: resolved_topics / total_topics
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ConversationCompletenessConfig] = None):
        self.config = config or ConversationCompletenessConfig()
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
        """Validate inputs for conversation completeness metric."""
        if not actual_output:
            raise ValueError("Conversation completeness metric requires 'actual_output'")
        if not test_case.metadata or "messages" not in test_case.metadata:
            raise ValueError(
                "Conversation completeness metric requires 'messages' in metadata "
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
        return f"""Evaluate if all topics in the conversation have been adequately addressed.

Conversation History:
{conversation}

Current User Input: {current_input}

Latest Assistant Response: {actual_output}

Task:
1. Identify all distinct topics, questions, or requests raised by the user
2. Determine if each topic has been adequately resolved/addressed

Respond with ONLY a JSON object in this format:
{{"topics": [
  {{"topic": "description of topic", "resolved": "yes" or "no"}},
  ...
], "resolved_count": <number>, "total_count": <number>, "completeness_score": <0.0 to 1.0>}}

A topic is "resolved" if:
- The question was answered
- The request was fulfilled
- The user's concern was adequately addressed

Your response:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "topics": [],
                "resolved_count": 0,
                "total_count": 0,
                "completeness_score": 0.0,
            }

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure conversation completeness."""
        self.validate_inputs(test_case, actual_output)

        messages = test_case.metadata.get("messages", [])  # type: ignore
        conversation = self._format_conversation(messages)
        output_str = str(actual_output)

        prompt = self._build_prompt(conversation, test_case.input, output_str)
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        result_data = self._parse_response(response.content)

        # Use the LLM's completeness score or calculate from counts
        score = result_data.get("completeness_score", 0.0)
        total = result_data.get("total_count", 0)
        resolved = result_data.get("resolved_count", 0)

        # Fallback calculation if score wasn't provided
        if score == 0.0 and total > 0:
            score = resolved / total

        passed = score >= self.threshold

        return MetricResult(
            name="conversation_completeness",
            score=score,
            passed=passed,
            metadata={
                "topics": result_data.get("topics", []),
                "resolved_count": resolved,
                "total_count": total,
            },
        )
