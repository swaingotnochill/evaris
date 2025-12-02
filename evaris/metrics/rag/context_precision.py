"""Context Precision metric for RAG evaluation.

This metric evaluates retriever quality by checking if relevant context nodes
rank higher than irrelevant ones.

Formula: Weighted cumulative precision
= (1 / |relevant nodes|) * sum(relevant_count_at_k / k * r_k)

Where:
- k = position in retrieval context (1 to n)
- r_k = 1 if node at position k is relevant, 0 otherwise
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ContextPrecisionConfig(BaseModel):
    """Configuration for context precision metric."""

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
    context_key: str = Field(
        default="retrieval_context",
        description="Key in test_case.metadata containing the retrieval context",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class ContextPrecisionMetric(BaseMetric):
    """Context Precision metric for RAG evaluation.

    Measures if relevant context nodes are ranked higher than irrelevant ones.
    Uses weighted cumulative precision to give higher weight to top-ranked nodes.

    Required inputs:
    - input: The user query
    - expected: The expected/reference answer
    - retrieval_context: List of retrieved context chunks (in metadata)

    Algorithm:
    1. LLM evaluates each context node for relevance to input/expected
    2. Calculate weighted cumulative precision favoring top positions

    Example:
        >>> metric = ContextPrecisionMetric()
        >>> test_case = TestCase(
        ...     input="What is Python?",
        ...     expected="Python is a programming language",
        ...     metadata={"retrieval_context": ["Python is...", "The weather..."]}
        ... )
        >>> result = await metric.a_measure(test_case, "Python is a language")
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ContextPrecisionConfig] = None):
        """Initialize context precision metric.

        Args:
            config: Configuration for the metric. If None, uses defaults.
        """
        self.config = config or ContextPrecisionConfig()
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
        """Validate inputs for context precision.

        Args:
            test_case: Test case with input, expected, and retrieval_context
            actual_output: Generated response

        Raises:
            ValueError: If required inputs are missing
        """
        if not test_case.input:
            raise ValueError("Context precision metric requires 'input' in test case")

        if not test_case.expected:
            raise ValueError("Context precision metric requires 'expected' in test case")

        context_key = self.config.context_key
        if not test_case.metadata or context_key not in test_case.metadata:
            raise ValueError(
                f"Context precision metric requires '{context_key}' in test_case.metadata"
            )

        context = test_case.metadata[context_key]
        if not context or not isinstance(context, list):
            raise ValueError(f"'{context_key}' must be a non-empty list of context strings")

    def _build_relevancy_prompt(
        self,
        input_query: str,
        expected_output: str,
        contexts: list[str],
    ) -> str:
        """Build prompt to evaluate context node relevancy.

        Args:
            input_query: The user query
            expected_output: The expected/reference answer
            contexts: List of context strings

        Returns:
            Prompt for relevancy evaluation
        """
        context_list = "\n".join(f"Node {i}: {ctx}" for i, ctx in enumerate(contexts))

        return f"""Evaluate if each context node is relevant for answering the question.
A node is relevant if it contains information that helps produce the expected answer.

Question: {input_query}

Expected Answer: {expected_output}

Context Nodes:
{context_list}

Respond with ONLY a JSON object in this format:
{{"verdicts": [
  {{"node": 0, "verdict": "yes" or "no", "reason": "brief explanation"}},
  {{"node": 1, "verdict": "yes" or "no", "reason": "brief explanation"}},
  ...
]}}

For each node:
- "yes" if the node contains information useful for the expected answer
- "no" if the node is irrelevant to answering the question

Your response:"""

    def _parse_verdicts(self, response: str, num_nodes: int) -> list[bool]:
        """Parse relevancy verdicts response.

        Args:
            response: LLM response with verdicts
            num_nodes: Number of context nodes expected

        Returns:
            List of boolean verdicts (True = relevant)
        """
        try:
            data = json.loads(response.strip())
            verdicts_data = data.get("verdicts", [])

            # Initialize all as irrelevant
            verdicts = [False] * num_nodes

            # Fill in verdicts from response
            for v in verdicts_data:
                node_idx = v.get("node", -1)
                if 0 <= node_idx < num_nodes:
                    verdicts[node_idx] = v.get("verdict", "").lower() == "yes"

            return verdicts

        except json.JSONDecodeError:
            return [False] * num_nodes

    def _calculate_weighted_precision(self, verdicts: list[bool]) -> float:
        """Calculate weighted cumulative precision.

        Formula: (1/|relevant|) * sum(precision_at_k * r_k)

        Args:
            verdicts: List of boolean verdicts in ranking order

        Returns:
            Weighted precision score between 0 and 1
        """
        if not verdicts:
            return 0.0

        num_relevant = sum(verdicts)
        if num_relevant == 0:
            return 0.0

        weighted_sum = 0.0
        relevant_count = 0

        for k, is_relevant in enumerate(verdicts, start=1):
            if is_relevant:
                relevant_count += 1
                # Precision at position k times relevance indicator
                precision_at_k = relevant_count / k
                weighted_sum += precision_at_k

        return weighted_sum / num_relevant

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure context precision.

        Args:
            test_case: Test case with input, expected, and retrieval_context
            actual_output: Generated response

        Returns:
            MetricResult with precision score
        """
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        context_key = self.config.context_key
        contexts = test_case.metadata[context_key]

        # Evaluate relevancy of each context node
        # expected is validated to be non-None in validate_inputs
        relevancy_prompt = self._build_relevancy_prompt(
            test_case.input,
            str(test_case.expected),
            contexts,
        )
        relevancy_response = await provider.a_complete(relevancy_prompt)
        verdicts = self._parse_verdicts(relevancy_response.content, len(contexts))

        # Calculate weighted precision
        score = self._calculate_weighted_precision(verdicts)
        passed = score >= self.threshold

        return MetricResult(
            name="context_precision",
            score=score,
            passed=passed,
            metadata={
                "context_count": len(contexts),
                "relevant_count": sum(verdicts),
                "verdicts": verdicts,
                "weighted_precision": score,
            },
        )
