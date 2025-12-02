"""Conversation Relevancy metric for conversational evaluation.

This metric checks whether the assistant's response is contextually
relevant to the conversation.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ConversationRelevancyConfig(BaseModel):
    """Configuration for conversation relevancy metric."""

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


class ConversationRelevancyMetric(BaseMetric):
    """Conversation Relevancy metric for conversational evaluation.

    Evaluates whether the assistant's response is contextually relevant
    to both the current input and the broader conversation context.

    Required: messages in metadata (conversation history)

    Checks for:
    - Direct relevance to current input
    - Contextual coherence with conversation history
    - Appropriate acknowledgment of prior context
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ConversationRelevancyConfig] = None):
        self.config = config or ConversationRelevancyConfig()
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
        """Validate inputs for conversation relevancy metric."""
        if not actual_output:
            raise ValueError("Conversation relevancy metric requires 'actual_output'")
        if not test_case.metadata or "messages" not in test_case.metadata:
            raise ValueError(
                "Conversation relevancy metric requires 'messages' in metadata "
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
        return f"""Evaluate if the assistant's response is contextually relevant.

Conversation History:
{conversation}

Current User Input: {current_input}

Assistant's Response: {actual_output}

Task:
Evaluate how relevant the response is to:
1. The current user input
2. The broader conversation context

Respond with ONLY a JSON object in this format:
{{"score": <0.0 to 1.0>, "reasoning": "brief explanation"}}

Score guidelines:
- 1.0: Perfectly relevant - directly addresses input with full context awareness
- 0.7-0.9: Highly relevant - addresses input, minor context gaps
- 0.4-0.6: Partially relevant - somewhat related but misses key points
- 0.0-0.3: Irrelevant - off-topic or ignores conversation context

Your response:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"score": 0.0, "reasoning": "Failed to parse response"}

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure conversation relevancy."""
        self.validate_inputs(test_case, actual_output)

        messages = test_case.metadata.get("messages", [])  # type: ignore
        conversation = self._format_conversation(messages)
        output_str = str(actual_output)

        prompt = self._build_prompt(conversation, test_case.input, output_str)
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        result_data = self._parse_response(response.content)

        score = result_data.get("score", 0.0)
        passed = score >= self.threshold

        return MetricResult(
            name="conversation_relevancy",
            score=score,
            passed=passed,
            metadata={
                "reasoning": result_data.get("reasoning", ""),
            },
        )
