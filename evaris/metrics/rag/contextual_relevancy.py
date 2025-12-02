"""Contextual Relevancy metric for RAG evaluation.

This metric evaluates how relevant the retrieved context is to the input query.
Unlike ContextRelevance (response-to-context), this measures context-to-query relevance.

Formula: Relevant Context Chunks / Total Context Chunks
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ContextualRelevancyConfig(BaseModel):
    """Configuration for contextual relevancy metric."""

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
        description="Key in metadata containing the retrieval context",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class ContextualRelevancyMetric(BaseMetric):
    """Contextual Relevancy metric for RAG evaluation.

    Evaluates whether the retrieved context chunks are relevant to the input query.
    This helps identify retrieval quality issues where irrelevant documents are retrieved.

    Required: input, retrieval_context in metadata

    Algorithm:
    1. For each context chunk, determine if it's relevant to the query
    2. Calculate: relevant_chunks / total_chunks
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ContextualRelevancyConfig] = None):
        self.config = config or ContextualRelevancyConfig()
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
        """Validate inputs for contextual relevancy metric."""
        if not test_case.input:
            raise ValueError("Contextual relevancy metric requires 'input' in test case")
        context_key = self.config.context_key
        if not test_case.metadata or context_key not in test_case.metadata:
            raise ValueError(f"Contextual relevancy metric requires '{context_key}' in metadata")

    def _build_prompt(
        self,
        input_query: str,
        context_chunks: list[str],
    ) -> str:
        """Build the evaluation prompt."""
        context_list = "\n".join(f"[{i}]: {chunk}" for i, chunk in enumerate(context_chunks))

        return f"""Evaluate the relevance of each context chunk to the given query.

Query: {input_query}

Context Chunks:
{context_list}

Task:
For each context chunk, determine if it contains information relevant to answering the query.

Respond with ONLY a JSON object in this format:
{{"verdicts": [
  {{"context_index": 0, "verdict": "yes" or "no", "reason": "brief explanation"}},
  ...
], "relevant_count": <number>, "total_count": <number>}}

A chunk is "relevant" if it:
- Contains information that helps answer the query
- Provides background or context for the query
- Is topically related to the query

Your response:"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"verdicts": [], "relevant_count": 0, "total_count": 0}

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure contextual relevancy."""
        self.validate_inputs(test_case, actual_output)

        context_key = self.config.context_key
        context_chunks = test_case.metadata.get(context_key, [])  # type: ignore

        if not context_chunks:
            return MetricResult(
                name="contextual_relevancy",
                score=0.0,
                passed=False,
                metadata={
                    "verdicts": [],
                    "reason": "No context chunks provided",
                },
            )

        prompt = self._build_prompt(test_case.input, context_chunks)
        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        result_data = self._parse_response(response.content)

        total = result_data.get("total_count", len(context_chunks))
        relevant = result_data.get("relevant_count", 0)

        # Calculate from verdicts if counts not provided
        if total == 0:
            verdicts = result_data.get("verdicts", [])
            total = len(verdicts)
            relevant = sum(1 for v in verdicts if v.get("verdict", "").lower() == "yes")

        score = relevant / total if total > 0 else 0.0
        passed = score >= self.threshold

        return MetricResult(
            name="contextual_relevancy",
            score=score,
            passed=passed,
            metadata={
                "verdicts": result_data.get("verdicts", []),
                "relevant_count": relevant,
                "total_count": total,
            },
        )
