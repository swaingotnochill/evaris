"""BLEU (Bilingual Evaluation Understudy) metric.

BLEU measures the similarity between generated text and reference text
by computing n-gram precision with a brevity penalty.

This is a deterministic metric - no LLM call needed.
"""

import math
from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase


class BLEUConfig(BaseModel):
    """Configuration for BLEU metric."""

    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    max_n: int = Field(
        default=4,
        description="Maximum n-gram size (typically 4 for BLEU-4)",
    )
    weights: Optional[list[float]] = Field(
        default=None,
        description="Weights for each n-gram (default: uniform)",
    )


class BLEUMetric(BaseMetric):
    """BLEU (Bilingual Evaluation Understudy) metric.

    Computes n-gram precision between generated text and reference.
    Commonly used for machine translation evaluation.

    Formula: BP * exp(sum(w_n * log(p_n)))
    where BP = brevity penalty, p_n = n-gram precision

    Required: expected output (reference text)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[BLEUConfig] = None):
        self.config = config or BLEUConfig()
        self.threshold = self.config.threshold

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        if not actual_output:
            raise ValueError("BLEU metric requires 'actual_output'")
        if not test_case.expected:
            raise ValueError("BLEU metric requires 'expected' (reference text)")

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter:
        """Get n-gram counts from tokens."""
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return Counter(ngrams)

    def _modified_precision(
        self,
        candidate_tokens: list[str],
        reference_tokens: list[str],
        n: int,
    ) -> float:
        """Calculate modified n-gram precision."""
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)

        if not candidate_ngrams:
            return 0.0

        # Clip counts by reference counts
        clipped_counts = {
            ngram: min(count, reference_ngrams.get(ngram, 0))
            for ngram, count in candidate_ngrams.items()
        }

        numerator = sum(clipped_counts.values())
        denominator = sum(candidate_ngrams.values())

        return numerator / denominator if denominator > 0 else 0.0

    def _brevity_penalty(
        self,
        candidate_len: int,
        reference_len: int,
    ) -> float:
        """Calculate brevity penalty."""
        if candidate_len >= reference_len:
            return 1.0
        if candidate_len == 0:
            return 0.0
        return math.exp(1 - reference_len / candidate_len)

    def _compute_bleu(
        self,
        candidate: str,
        reference: str,
    ) -> tuple[float, dict[str, Any]]:
        """Compute BLEU score.

        Returns:
            Tuple of (score, metadata)
        """
        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)

        if not candidate_tokens or not reference_tokens:
            return 0.0, {"precisions": [], "brevity_penalty": 0.0}

        max_n = min(self.config.max_n, len(candidate_tokens), len(reference_tokens))
        weights = self.config.weights or [1.0 / max_n] * max_n

        precisions = []
        log_precisions = []

        for n in range(1, max_n + 1):
            p = self._modified_precision(candidate_tokens, reference_tokens, n)
            precisions.append(p)
            if p > 0:
                log_precisions.append(weights[n - 1] * math.log(p))
            else:
                # If any precision is 0, BLEU is 0
                return 0.0, {
                    "precisions": precisions,
                    "brevity_penalty": 0.0,
                    "zero_precision_at": n,
                }

        bp = self._brevity_penalty(len(candidate_tokens), len(reference_tokens))
        score = bp * math.exp(sum(log_precisions))

        return score, {
            "precisions": precisions,
            "brevity_penalty": bp,
            "candidate_length": len(candidate_tokens),
            "reference_length": len(reference_tokens),
        }

    async def a_measure(
        self,
        test_case: TestCase,
        actual_output: Any,
    ) -> MetricResult:
        self.validate_inputs(test_case, actual_output)

        candidate = str(actual_output)
        reference = str(test_case.expected)

        score, metadata = self._compute_bleu(candidate, reference)
        passed = score >= self.threshold

        return MetricResult(
            name="bleu",
            score=score,
            passed=passed,
            metadata=metadata,
        )
