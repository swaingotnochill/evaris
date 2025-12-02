"""Context relevance metric for RAG evaluation.

This metric evaluates whether the retrieved context is relevant to the user's query.
It helps assess the quality of the retrieval system.
"""

from typing import Any, Optional

from pydantic import Field

from evaris.metrics.llm_judge import JudgeConfig, LLMJudgeMetric
from evaris.types import TestCase


class ContextRelevanceConfig(JudgeConfig):
    """Configuration for context relevance metric."""

    context_key: str = Field(
        default="context", description="Key in test_case.metadata containing the context"
    )


class ContextRelevanceMetric(LLMJudgeMetric):
    """Context relevance metric for RAG evaluation.

    Measures if the retrieved context is relevant to the input query.
    High score means the context contains information needed to answer the query.
    Low score means the context is irrelevant or noisy.

    ABC Compliance:
    - O.c.1: Uses LLM-as-a-judge for semantic verification
    """

    def __init__(self, config: Optional[ContextRelevanceConfig] = None):
        """Initialize context relevance metric.

        Args:
            config: Configuration for context relevance. If None, uses defaults.
        """
        super().__init__(config or ContextRelevanceConfig())
        # Re-type self.config for type checkers
        self.config: ContextRelevanceConfig = self.config  # type: ignore

    def _validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs for context relevance.

        Requires 'context' in metadata (or configured key).
        Does NOT require 'expected' output or 'actual_output' (though actual_output is passed).

        Args:
            test_case: Test case
            actual_output: Agent's output (ignored for this metric)

        Raises:
            ValueError: If context is missing
        """
        context_key = self.config.context_key
        if context_key not in test_case.metadata:
            raise ValueError(
                f"Context relevance metric requires '{
                    context_key}' in test_case.metadata"
            )

    def _get_default_prompt(self, test_case: TestCase, actual_output: Any) -> str:
        """Get context relevance judge prompt.

        Args:
            test_case: Test case with input (query) and context
            actual_output: Agent's actual output (ignored)

        Returns:
            Formatted judge prompt
        """
        query = test_case.input
        context = test_case.metadata[self.config.context_key]

        return f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Your task is to evaluate the RELEVANCE of the Retrieved Context to the User Query.

User Query:
{query}

Retrieved Context:
{context}

Evaluate if the Retrieved Context is relevant to the User Query.
- The Context should contain information useful for answering the Query.
- Irrelevant information decreases the score (noise).
- Missing information decreases the score.

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}

Score Guidelines:
- 1.0: The Context contains all necessary information to answer the Query, with minimal noise.
- 0.7: The Context contains most necessary information, but includes some irrelevant text.
- 0.3: The Context contains only a small amount of relevant information, mostly noise.
- 0.0: The Context is completely irrelevant to the Query.

Your response:"""
