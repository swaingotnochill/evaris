"""Answer Relevancy metric for RAG evaluation.

This metric evaluates whether the generated answer addresses the input query.
It extracts statements from the output and checks each for relevance.

Formula: Number of Relevant Statements / Total Number of Statements

This is a referenceless metric - it does not require expected output.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase
from evaris.providers.factory import get_provider


class AnswerRelevancyConfig(BaseModel):
    """Configuration for answer relevancy metric."""

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
    include_reasoning: bool = Field(
        default=True,
        description="Include reasoning for each verdict",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature for deterministic results",
    )


class AnswerRelevancyMetric(BaseMetric):
    """Answer Relevancy metric for RAG evaluation.

    Measures if the generated answer is relevant to the input query.
    This metric is referenceless - it only needs input and actual_output.

    Algorithm:
    1. Extract individual statements from the actual output
    2. Classify each statement as relevant or irrelevant to the input
    3. Calculate: relevant_statements / total_statements

    Example:
        >>> metric = AnswerRelevancyMetric()
        >>> test_case = TestCase(input="What is Python?", expected=None)
        >>> result = await metric.a_measure(test_case, "Python is a programming language.")
        >>> print(result.score)  # 1.0 if the answer is relevant
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[AnswerRelevancyConfig] = None):
        """Initialize answer relevancy metric.

        Args:
            config: Configuration for the metric. If None, uses defaults.
        """
        self.config = config or AnswerRelevancyConfig()
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
        """Validate inputs for answer relevancy.

        Args:
            test_case: Test case with input query
            actual_output: Generated response

        Raises:
            ValueError: If input or actual_output is missing
        """
        if not test_case.input:
            raise ValueError("Answer relevancy metric requires 'input' in test case")
        if not actual_output:
            raise ValueError("Answer relevancy metric requires 'actual_output'")

    def _build_extraction_prompt(self, actual_output: str) -> str:
        """Build prompt to extract statements from output.

        Args:
            actual_output: The generated response

        Returns:
            Prompt for statement extraction
        """
        return f"""Extract all distinct factual statements from the following text.
Break down the text into individual, self-contained statements.

Text:
{actual_output}

Respond with ONLY a JSON object in this format:
{{"statements": ["statement 1", "statement 2", ...]}}

Important:
- Each statement should be a single claim or fact
- Do not include questions or incomplete sentences
- If the text is empty or has no statements, return {{"statements": []}}

Your response:"""

    def _build_relevancy_prompt(self, input_query: str, statements: list[str]) -> str:
        """Build prompt to check statement relevancy.

        Args:
            input_query: The original user query
            statements: List of extracted statements

        Returns:
            Prompt for relevancy classification
        """
        statements_json = json.dumps(statements)

        return f"""Evaluate if each statement is relevant to answering the given question.
A statement is relevant if it directly or indirectly helps answer the question.

Question: {input_query}

Statements to evaluate:
{statements_json}

Respond with ONLY a JSON object in this format:
{{"verdicts": [
  {{"statement": "statement text", "verdict": "yes" or "no", "reason": "brief explanation"}},
  ...
]}}

For each statement:
- "yes" if the statement is relevant to answering the question
- "no" if the statement is off-topic or irrelevant

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

    def _parse_verdicts(self, response: str) -> list[dict[str, Any]]:
        """Parse relevancy verdicts response.

        Args:
            response: LLM response with verdicts

        Returns:
            List of verdict dictionaries
        """
        try:
            data = json.loads(response.strip())
            return data.get("verdicts", [])
        except json.JSONDecodeError:
            return []

    def _calculate_score(self, verdicts: list[dict[str, Any]]) -> float:
        """Calculate relevancy score from verdicts.

        Args:
            verdicts: List of verdict dictionaries

        Returns:
            Score between 0 and 1
        """
        if not verdicts:
            return 0.0

        relevant_count = sum(1 for v in verdicts if v.get("verdict", "").lower() == "yes")

        return relevant_count / len(verdicts)

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        """Measure answer relevancy.

        Args:
            test_case: Test case with input query
            actual_output: Generated response

        Returns:
            MetricResult with relevancy score
        """
        self.validate_inputs(test_case, actual_output)

        provider = self._get_provider()
        output_str = str(actual_output)

        # Step 1: Extract statements from output
        extraction_prompt = self._build_extraction_prompt(output_str)
        extraction_response = await provider.a_complete(extraction_prompt)
        statements = self._parse_statements(extraction_response.content)

        if not statements:
            # No statements to evaluate
            return MetricResult(
                name="answer_relevancy",
                score=0.0,
                passed=False,
                metadata={
                    "statements": [],
                    "verdicts": [],
                    "reason": "No statements extracted from output",
                },
            )

        # Step 2: Check relevancy of each statement
        relevancy_prompt = self._build_relevancy_prompt(test_case.input, statements)
        relevancy_response = await provider.a_complete(relevancy_prompt)
        verdicts = self._parse_verdicts(relevancy_response.content)

        # Step 3: Calculate score
        score = self._calculate_score(verdicts)
        passed = score >= self.threshold

        return MetricResult(
            name="answer_relevancy",
            score=score,
            passed=passed,
            metadata={
                "statements": statements,
                "verdicts": verdicts,
                "total_statements": len(statements),
                "relevant_statements": sum(
                    1 for v in verdicts if v.get("verdict", "").lower() == "yes"
                ),
            },
        )
