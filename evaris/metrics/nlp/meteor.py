"""METEOR (Metric for Evaluation of Translation with Explicit ORdering) metric.

METEOR evaluates text similarity using:
- Exact word matching
- Stemming (morphological variants)
- Synonym matching (optional, simplified)
- Word order penalty (fragmentation)

Formula: METEOR = (1 - gamma * frag^beta) * F_mean
where F_mean = (P * R) / (alpha * P + (1 - alpha) * R)

This is a deterministic metric - no LLM call needed.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.core.protocols import BaseMetric
from evaris.core.types import MetricResult, TestCase


class METEORConfig(BaseModel):
    """Configuration for METEOR metric."""

    threshold: float = Field(
        default=0.5,
        description="Score threshold for passing (0.0-1.0)",
    )
    alpha: float = Field(
        default=0.9,
        description="Weight for precision in harmonic mean (0-1)",
    )
    beta: float = Field(
        default=3.0,
        description="Exponent for fragmentation penalty",
    )
    gamma: float = Field(
        default=0.5,
        description="Weight for fragmentation penalty (0-1)",
    )


class METEORMetric(BaseMetric):
    """METEOR metric for translation/paraphrasing evaluation.

    Unlike BLEU, METEOR:
    - Uses stemming for morphological matching
    - Considers word order through fragmentation penalty
    - Gives more weight to recall

    Required: expected output (reference text)
    """

    threshold: float = 0.5

    def __init__(self, config: Optional[METEORConfig] = None):
        self.config = config or METEORConfig()
        self.threshold = self.config.threshold

    def validate_inputs(self, test_case: TestCase, actual_output: Any) -> None:
        if not actual_output:
            raise ValueError("METEOR metric requires 'actual_output'")
        if not test_case.expected:
            raise ValueError("METEOR metric requires 'expected' (reference text)")

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def _stem(self, word: str) -> str:
        """Simple suffix-stripping stemmer.

        This is a simplified Porter-like stemmer for common English suffixes.
        For production use, consider using NLTK's PorterStemmer.
        """
        suffixes = ["ing", "ed", "es", "s", "ly", "er", "est", "ment", "ness", "tion", "ous"]
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        return word

    def _match_words(
        self,
        candidate_tokens: list[str],
        reference_tokens: list[str],
    ) -> tuple[list[tuple[int, int]], int, int]:
        """Match words between candidate and reference.

        Returns:
            Tuple of (matches, matched_candidate_count, matched_reference_count)
            where matches is list of (candidate_idx, reference_idx) pairs
        """
        matches: list[tuple[int, int]] = []
        used_candidate: set[int] = set()
        used_reference: set[int] = set()

        # First pass: exact matches
        for c_idx, c_word in enumerate(candidate_tokens):
            if c_idx in used_candidate:
                continue
            for r_idx, r_word in enumerate(reference_tokens):
                if r_idx in used_reference:
                    continue
                if c_word == r_word:
                    matches.append((c_idx, r_idx))
                    used_candidate.add(c_idx)
                    used_reference.add(r_idx)
                    break

        # Second pass: stem matches
        for c_idx, c_word in enumerate(candidate_tokens):
            if c_idx in used_candidate:
                continue
            c_stem = self._stem(c_word)
            for r_idx, r_word in enumerate(reference_tokens):
                if r_idx in used_reference:
                    continue
                r_stem = self._stem(r_word)
                if c_stem == r_stem:
                    matches.append((c_idx, r_idx))
                    used_candidate.add(c_idx)
                    used_reference.add(r_idx)
                    break

        return matches, len(used_candidate), len(used_reference)

    def _count_chunks(self, matches: list[tuple[int, int]]) -> int:
        """Count the number of chunks (contiguous matched sequences).

        A chunk is a sequence of matches where both candidate and reference
        indices are consecutive.
        """
        if not matches:
            return 0

        # Sort by candidate index
        sorted_matches = sorted(matches, key=lambda x: x[0])

        chunks = 1
        for i in range(1, len(sorted_matches)):
            prev_c, prev_r = sorted_matches[i - 1]
            curr_c, curr_r = sorted_matches[i]

            # Check if consecutive in both sequences
            if curr_c != prev_c + 1 or curr_r != prev_r + 1:
                chunks += 1

        return chunks

    def _compute_meteor(
        self,
        candidate: str,
        reference: str,
    ) -> tuple[float, dict[str, Any]]:
        """Compute METEOR score.

        Returns:
            Tuple of (score, metadata)
        """
        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)

        if not candidate_tokens or not reference_tokens:
            return 0.0, {
                "precision": 0.0,
                "recall": 0.0,
                "f_mean": 0.0,
                "fragmentation_penalty": 1.0,
            }

        # Match words
        matches, matched_cand, matched_ref = self._match_words(candidate_tokens, reference_tokens)

        if not matches:
            return 0.0, {
                "precision": 0.0,
                "recall": 0.0,
                "f_mean": 0.0,
                "fragmentation_penalty": 1.0,
                "matches": 0,
            }

        # Precision and Recall
        precision = matched_cand / len(candidate_tokens)
        recall = matched_ref / len(reference_tokens)

        # Weighted harmonic mean (F_mean)
        alpha = self.config.alpha
        if precision == 0 or recall == 0:
            f_mean = 0.0
        else:
            f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        # Fragmentation penalty
        chunks = self._count_chunks(matches)
        frag = chunks / len(matches) if matches else 0.0

        beta = self.config.beta
        gamma = self.config.gamma
        frag_penalty = 1 - gamma * (frag**beta)

        # Final METEOR score
        score = frag_penalty * f_mean

        return score, {
            "precision": precision,
            "recall": recall,
            "f_mean": f_mean,
            "chunks": chunks,
            "matches": len(matches),
            "fragmentation": frag,
            "fragmentation_penalty": frag_penalty,
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

        score, metadata = self._compute_meteor(candidate, reference)
        passed = score >= self.threshold

        return MetricResult(
            name="meteor",
            score=score,
            passed=passed,
            metadata=metadata,
        )
