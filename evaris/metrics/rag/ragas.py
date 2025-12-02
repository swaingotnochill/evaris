"""RAGAS (Retrieval Augmented Generation Assessment) composite metric.

RAGAS is a framework for evaluating RAG pipelines that combines multiple
metrics into a single score. This implementation computes the average of:
- Answer Relevancy
- Faithfulness
- Context Precision
- Context Recall

Reference: https://arxiv.org/abs/2309.15217
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.metrics.rag.answer_relevancy import AnswerRelevancyConfig, AnswerRelevancyMetric
from evaris.metrics.rag.context_precision import ContextPrecisionConfig, ContextPrecisionMetric
from evaris.metrics.rag.context_recall import ContextRecallConfig, ContextRecallMetric
from evaris.providers.factory import get_provider


class RAGASConfig(BaseModel):
    """Configuration for RAGAS composite metric."""

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
    faithfulness_context_key: str = Field(
        default="context",
        description="Key in metadata for faithfulness context",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )
    # Component toggles
    include_answer_relevancy: bool = Field(
        default=True,
        description="Include Answer Relevancy in composite score",
    )
    include_faithfulness: bool = Field(
        default=True,
        description="Include Faithfulness in composite score",
    )
    include_context_precision: bool = Field(
        default=True,
        description="Include Context Precision in composite score",
    )
    include_context_recall: bool = Field(
        default=True,
        description="Include Context Recall in composite score",
    )


class RAGASMetric(BaseMetric):
    """RAGAS composite metric for RAG evaluation.

    Combines multiple RAG metrics into a single comprehensive score.
    The final score is the average of all enabled component metrics.

    Components:
    - Answer Relevancy: Is the answer relevant to the question?
    - Faithfulness: Is the answer faithful to the context?
    - Context Precision: Are relevant contexts ranked higher?
    - Context Recall: Are all relevant facts in the context?

    Required:
    - input (query)
    - expected (ground truth)
    - actual_output (generated answer)
    - retrieval_context in metadata
    - context in metadata (for faithfulness)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[RAGASConfig] = None):
        self.config = config or RAGASConfig()
        self.threshold = self.config.threshold
        self._provider = None

        # Initialize component metrics
        self._answer_relevancy = AnswerRelevancyMetric(
            config=AnswerRelevancyConfig(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        )
        self._context_precision = ContextPrecisionMetric(
            config=ContextPrecisionConfig(
                provider=self.config.provider,
                model=self.config.model,
                context_key=self.config.context_key,
                temperature=self.config.temperature,
            )
        )
        self._context_recall = ContextRecallMetric(
            config=ContextRecallConfig(
                provider=self.config.provider,
                model=self.config.model,
                context_key=self.config.context_key,
                temperature=self.config.temperature,
            )
        )

    def _get_provider(self) -> Any:
        """Get or create the LLM provider for faithfulness."""
        if self._provider is None:
            self._provider = get_provider(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        return self._provider

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        """Validate inputs for RAGAS metric."""
        if not test_case.input:
            raise ValueError("RAGAS metric requires 'input' in test case")
        if not actual_output:
            raise ValueError("RAGAS metric requires 'actual_output'")

        context_key = self.config.context_key
        if not test_case.metadata or context_key not in test_case.metadata:
            raise ValueError(f"RAGAS metric requires '{context_key}' in metadata")

    async def _measure_faithfulness(
        self,
        test_case: TestCase,
        actual_output: str,
    ) -> float:
        """Measure faithfulness score using LLM judge."""
        faith_key = self.config.faithfulness_context_key
        context = test_case.metadata.get(faith_key, "")  # type: ignore

        if not context:
            # Try using retrieval_context as fallback
            retrieval_ctx = test_case.metadata.get(self.config.context_key, [])  # type: ignore
            if retrieval_ctx:
                context = (
                    "\n".join(retrieval_ctx)
                    if isinstance(retrieval_ctx, list)
                    else str(retrieval_ctx)
                )

        if not context:
            return 0.0

        prompt = f"""Evaluate the faithfulness of the response to the given context.

Context:
{context}

Response:
{actual_output}

Evaluate if the response is faithful to the context:
- All claims must be supported by the context
- No hallucinations or made-up information

Respond with ONLY a JSON object:
{{"score": <0.0 to 1.0>, "reasoning": "brief explanation"}}

Your response:"""

        provider = self._get_provider()
        response = await provider.a_complete(prompt)

        try:
            import json

            result = json.loads(response.content.strip())
            return float(result.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            return 0.0

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure RAGAS composite score."""
        self.validate_inputs(test_case, actual_output)

        output_str = str(actual_output)
        component_scores: dict[str, float] = {}
        scores: list[float] = []

        # Answer Relevancy
        if self.config.include_answer_relevancy:
            try:
                ar_result = await self._answer_relevancy.a_measure(test_case, output_str)
                component_scores["answer_relevancy"] = ar_result.score
                scores.append(ar_result.score)
            except Exception:
                component_scores["answer_relevancy"] = 0.0
                scores.append(0.0)

        # Faithfulness
        if self.config.include_faithfulness:
            try:
                faith_score = await self._measure_faithfulness(test_case, output_str)
                component_scores["faithfulness"] = faith_score
                scores.append(faith_score)
            except Exception:
                component_scores["faithfulness"] = 0.0
                scores.append(0.0)

        # Context Precision
        if self.config.include_context_precision:
            try:
                cp_result = await self._context_precision.a_measure(test_case, output_str)
                component_scores["context_precision"] = cp_result.score
                scores.append(cp_result.score)
            except Exception:
                component_scores["context_precision"] = 0.0
                scores.append(0.0)

        # Context Recall
        if self.config.include_context_recall:
            try:
                cr_result = await self._context_recall.a_measure(test_case, output_str)
                component_scores["context_recall"] = cr_result.score
                scores.append(cr_result.score)
            except Exception:
                component_scores["context_recall"] = 0.0
                scores.append(0.0)

        # Calculate average
        final_score = sum(scores) / len(scores) if scores else 0.0
        passed = final_score >= self.threshold

        return MetricResult(
            name="ragas",
            score=final_score,
            passed=passed,
            metadata={
                "component_scores": component_scores,
                "num_components": len(scores),
            },
        )
