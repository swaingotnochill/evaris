"""Context Recall metric for RAG evaluation.

This metric evaluates retriever quality by checking if the retrieval context
contains all the information needed to produce the expected answer.

Formula: Number of Attributable Statements / Total Statements in expected_output

Note: This metric evaluates expected_output (not actual_output) against context,
because we want to measure retriever coverage for an ideal response.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class ContextRecallConfig(BaseModel):
    """Configuration for context recall metric."""

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


class ContextRecallMetric(BaseMetric):
    """Context Recall metric for RAG evaluation.

    Measures if the retrieval context contains all information needed to
    produce the expected answer. Higher recall means better coverage.

    Required inputs:
    - input: The user query
    - expected: The expected/reference answer (evaluated for coverage)
    - retrieval_context: List of retrieved context chunks (in metadata)

    Algorithm:
    1. Extract statements from expected_output
    2. Check if each statement can be attributed to the retrieval_context
    3. Calculate: attributable_statements / total_statements

    Example:
        >>> metric = ContextRecallMetric()
        >>> test_case = TestCase(
        ...     input="What is Python?",
        ...     expected="Python is a programming language created by Guido",
        ...     metadata={"retrieval_context": ["Python is a language...", "Guido..."]}
        ... )
        >>> result = await metric.a_measure(test_case, "Python is a language")
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ContextRecallConfig] = None):
        """Initialize context recall metric.

        Args:
            config: Configuration for the metric. If None, uses defaults.
        """
        self.config = config or ContextRecallConfig()
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
        """Validate inputs for context recall.

        Args:
            test_case: Test case with input, expected, and retrieval_context
            actual_output: Generated response (not used in this metric)

        Raises:
            ValueError: If required inputs are missing
        """
        if not test_case.input:
            raise ValueError("Context recall metric requires 'input' in test case")

        if not test_case.expected:
            raise ValueError("Context recall metric requires 'expected' in test case")

        context_key = self.config.context_key
        if not test_case.metadata or context_key not in test_case.metadata:
            raise ValueError(
                f"Context recall metric requires '{context_key}' in test_case.metadata"
            )

        context = test_case.metadata[context_key]
        if not context or not isinstance(context, list):
            raise ValueError(f"'{context_key}' must be a non-empty list of context strings")

    def _build_extraction_prompt(self, expected_output: str) -> str:
        """Build prompt to extract statements from expected output.

        Args:
            expected_output: The expected/reference answer

        Returns:
            Prompt for statement extraction
        """
        return f"""Extract all distinct factual statements from the following text.
Break down the text into individual, self-contained statements.

Text:
{expected_output}

Respond with ONLY a JSON object in this format:
{{"statements": ["statement 1", "statement 2", ...]}}

Important:
- Each statement should be a single claim or fact
- Do not include questions or incomplete sentences
- If the text is empty or has no statements, return {{"statements": []}}

Your response:"""

    def _build_attribution_prompt(
        self,
        statements: list[str],
        contexts: list[str],
    ) -> str:
        """Build prompt to check statement attribution to context.

        Args:
            statements: List of statements from expected output
            contexts: List of retrieval context strings

        Returns:
            Prompt for attribution evaluation
        """
        statements_json = json.dumps(statements)
        context_text = "\n---\n".join(contexts)

        return f"""For each statement, determine if it can be attributed to the given context.
A statement is attributable if the context contains information that supports or implies it.

Context:
{context_text}

Statements to evaluate:
{statements_json}

Respond with ONLY a JSON object in this format:
{{"verdicts": [
  {{"statement": "statement text", "attributed": "yes" or "no", "reason": "brief explanation"}},
  ...
]}}

For each statement:
- "yes" if the statement can be supported by information in the context
- "no" if the context does not contain supporting information

Your response:"""

    def _parse_statements(self, response: str) -> list[str]:
        """Parse statement extraction response.

        Args:
            response: LLM response with extracted statements

        Returns:
            List of extracted statements
        """
        try:
            data = json.loads(response.strip())
            return data.get("statements", [])
        except json.JSONDecodeError:
            return []

    def _parse_attributions(self, response: str) -> list[dict[str, Any]]:
        """Parse attribution verdicts response.

        Args:
            response: LLM response with attribution verdicts

        Returns:
            List of verdict dictionaries
        """
        try:
            data = json.loads(response.strip())
            return data.get("verdicts", [])
        except json.JSONDecodeError:
            return []

    def _calculate_score(self, verdicts: list[dict[str, Any]]) -> float:
        """Calculate recall score from attribution verdicts.

        Args:
            verdicts: List of attribution verdict dictionaries

        Returns:
            Score between 0 and 1
        """
        if not verdicts:
            return 0.0

        attributed_count = sum(1 for v in verdicts if v.get("attributed", "").lower() == "yes")

        return attributed_count / len(verdicts)

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure context recall.

        Args:
            test_case: Test case with input, expected, and retrieval_context
            actual_output: Generated response (not directly used)

        Returns:
            MetricResult with recall score
        """
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        context_key = self.config.context_key
        contexts = test_case.metadata[context_key]

        # Step 1: Extract statements from expected output
        # expected is validated to be non-None in validate_inputs
        extraction_prompt = self._build_extraction_prompt(str(test_case.expected))
        extraction_response = await provider.a_complete(extraction_prompt)
        statements = self._parse_statements(extraction_response.content)

        if not statements:
            # No statements to evaluate
            return MetricResult(
                name="context_recall",
                score=0.0,
                passed=False,
                metadata={
                    "statements": [],
                    "verdicts": [],
                    "reason": "No statements extracted from expected output",
                },
            )

        # Step 2: Check attribution of each statement to context
        attribution_prompt = self._build_attribution_prompt(statements, contexts)
        attribution_response = await provider.a_complete(attribution_prompt)
        verdicts = self._parse_attributions(attribution_response.content)

        # Step 3: Calculate recall score
        score = self._calculate_score(verdicts)
        passed = score >= self.threshold

        return MetricResult(
            name="context_recall",
            score=score,
            passed=passed,
            metadata={
                "statements": statements,
                "verdicts": verdicts,
                "total_statements": len(statements),
                "attributed_statements": sum(
                    1 for v in verdicts if v.get("attributed", "").lower() == "yes"
                ),
                "context_count": len(contexts),
            },
        )
