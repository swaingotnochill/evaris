"""ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.

ROUGE measures the overlap between generated text and reference text,
commonly used for summarization evaluation.

This is a deterministic metric - no LLM call needed.
"""

from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase


class ROUGEConfig(BaseModel):
    """Configuration for ROUGE metric."""

    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    rouge_type: str = Field(
        default="rouge-l",
        description="ROUGE type: 'rouge-1', 'rouge-2', or 'rouge-l'",
    )
    use_f1: bool = Field(
        default=True,
        description="Return F1 score (True) or just recall (False)",
    )


class ROUGEMetric(BaseMetric):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.

    Supports:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence

    Required: expected output (reference text)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[ROUGEConfig] = None):
        self.config = config or ROUGEConfig()
        self.threshold = self.config.threshold

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        if not actual_output:
            raise ValueError("ROUGE metric requires 'actual_output'")
        if not test_case.expected:
            raise ValueError("ROUGE metric requires 'expected' (reference text)")

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter:
        """Get n-gram counts from tokens."""
        if n > len(tokens):
            return Counter()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return Counter(ngrams)

    def _rouge_n(
        self,
        candidate_tokens: list[str],
        reference_tokens: list[str],
        n: int,
    ) -> tuple[float, float, float]:
        """Compute ROUGE-N scores.

        Returns:
            Tuple of (precision, recall, f1)
        """
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)

        if not candidate_ngrams or not reference_ngrams:
            return 0.0, 0.0, 0.0

        # Count overlapping n-grams
        overlap = sum((candidate_ngrams & reference_ngrams).values())

        precision = overlap / sum(candidate_ngrams.values()) if candidate_ngrams else 0.0
        recall = overlap / sum(reference_ngrams.values()) if reference_ngrams else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    def _lcs_length(self, x: list[str], y: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _rouge_l(
        self,
        candidate_tokens: list[str],
        reference_tokens: list[str],
    ) -> tuple[float, float, float]:
        """Compute ROUGE-L scores using longest common subsequence.

        Returns:
            Tuple of (precision, recall, f1)
        """
        if not candidate_tokens or not reference_tokens:
            return 0.0, 0.0, 0.0

        lcs_len = self._lcs_length(candidate_tokens, reference_tokens)

        precision = lcs_len / len(candidate_tokens)
        recall = lcs_len / len(reference_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        candidate = str(actual_output)
        reference = str(test_case.expected)

        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)

        rouge_type = self.config.rouge_type.lower()

        if rouge_type == "rouge-1":
            precision, recall, f1 = self._rouge_n(candidate_tokens, reference_tokens, 1)
        elif rouge_type == "rouge-2":
            precision, recall, f1 = self._rouge_n(candidate_tokens, reference_tokens, 2)
        elif rouge_type == "rouge-l":
            precision, recall, f1 = self._rouge_l(candidate_tokens, reference_tokens)
        else:
            raise ValueError(f"Unknown ROUGE type: {rouge_type}")

        score = f1 if self.config.use_f1 else recall
        passed = score >= self.threshold

        return MetricResult(
            name=f"rouge_{rouge_type.replace('-', '_')}",
            score=score,
            passed=passed,
            metadata={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "rouge_type": rouge_type,
                "candidate_length": len(candidate_tokens),
                "reference_length": len(reference_tokens),
            },
        )
