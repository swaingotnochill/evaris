"""Faithfulness metric for RAG evaluation.

This metric evaluates whether the agent's output is factually consistent with the
provided context. It checks if the answer can be derived *solely* from the context,
detecting hallucinations.
"""

from typing import Any, Optional

from pydantic import Field

from evaris.metrics.llm_judge import JudgeConfig, LLMJudgeMetric
from evaris.types import TestCase


class FaithfulnessConfig(JudgeConfig):
    """Configuration for faithfulness metric."""

    context_key: str = Field(
        default="context", description="Key in test_case.metadata containing the context"
    )


class FaithfulnessMetric(LLMJudgeMetric):
    """Faithfulness metric for RAG evaluation.

    Measures if the generated answer is faithful to the retrieved context.
    High score means the answer is supported by the context.
    Low score means the answer contains hallucinations or information not in context.

    ABC Compliance:
    - O.c.1: Uses LLM-as-a-judge for semantic verification
    - O.c.2: Detects hallucinations (unsupported claims)
    """

    def __init__(self, config: Optional[FaithfulnessConfig] = None):
        """Initialize faithfulness metric.

        Args:
            config: Configuration for faithfulness. If None, uses defaults.
        """
        super().__init__(config or FaithfulnessConfig())
        # Re-type self.config for type checkers
        self.config: FaithfulnessConfig = self.config  # type: ignore

    def _validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs for faithfulness.

        Requires 'context' in metadata (or configured key).
        Does NOT require 'expected' output.

        Args:
            test_case: Test case
            actual_output: Agent's output

        Raises:
            ValueError: If context is missing
        """
        context_key = self.config.context_key
        if context_key not in test_case.metadata:
            raise ValueError(f"Faithfulness metric requires '{context_key}' in test_case.metadata")

    def _get_default_prompt(self, test_case: TestCase, actual_output: Any) -> str:
        """Get faithfulness judge prompt.

        Args:
            test_case: Test case with context
            actual_output: Agent's actual output

        Returns:
            Formatted judge prompt
        """
        context = test_case.metadata[self.config.context_key]

        return f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Your task is to evaluate the FAITHFULNESS of the Actual Output to the provided Context.

Context:
{context}

Actual Output:
{actual_output}

Evaluate if the Actual Output is faithful to the Context.
- The output must be derived SOLELY from the provided Context.
- Any claim not supported by the Context is a hallucination (Faithfulness = 0).
- If the output uses outside knowledge not present in the Context, it is not faithful.
- If the output contradicts the Context, it is not faithful.

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}

Score Guidelines:
- 1.0: All claims in the output are fully supported by the Context.
- 0.5: Some claims are supported, but some are not found in the Context.
- 0.0: The output contradicts the Context or contains entirely unsupported claims.

Your response:"""
