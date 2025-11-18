"""Semantic similarity metric using embeddings.

This metric handles semantically equivalent expressions that may differ in wording.
Implements ABC check O.a.1 for considering equivalent expressions.
"""

import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaris.tracing import get_debug_logger, get_tracer
from evaris.types import BaseMetric, MetricResult, TestCase


class SemanticSimilarityConfig(BaseModel):
    """Configuration for semantic similarity metric."""

    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model",
    )
    threshold: float = Field(default=0.8, description="Similarity threshold for passing (0.0-1.0)")
    normalize: bool = Field(default=True, description="Normalize text before comparison")
    case_sensitive: bool = Field(default=False, description="Case sensitive comparison")


class SemanticSimilarityMetric(BaseMetric):
    """Semantic similarity metric using sentence embeddings.

    Measures semantic similarity between expected and actual outputs using
    sentence transformers. Handles semantically equivalent expressions.

    ABC Compliance:
    - O.a.1: Considers expressions semantically equivalent to ground truth
    - O.a.2: Handles redundant words used by agents

    Example:
        >>> from evaris.metrics.semantic_similarity import SemanticSimilarityMetric
        >>> metric = SemanticSimilarityMetric()
        >>> tc = TestCase(
        ...     input="What is the capital of France?",
        ...     expected="Paris"
        ... )
        >>> result = metric.score(tc, "The capital is Paris")
        >>> print(result.score)  # High similarity score
    """

    def __init__(self, config: Optional[SemanticSimilarityConfig] = None):
        """Initialize semantic similarity metric.

        Args:
            config: Configuration for semantic similarity. If None, uses defaults.
        """
        self.config = config or SemanticSimilarityConfig()
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model)
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        ABC O.a.2: Handles redundant words and variations.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not self.config.normalize:
            return text

        # Convert to string if not already
        text = str(text).strip()

        # Case normalization
        if not self.config.case_sensitive:
            text = text.lower()

        # Remove extra whitespace
        import re

        text = re.sub(r"\s+", " ", text)

        return text

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Generate embeddings
        assert self._model is not None, "Model not initialized"
        embeddings = self._model.encode([text1, text2])

        # Compute cosine similarity
        from numpy import dot
        from numpy.linalg import norm

        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))

        # Convert to float and ensure [0, 1] range
        similarity = float(similarity)
        similarity = (similarity + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]

        return similarity

    def score(self, test_case: TestCase, actual_output: Any) -> MetricResult:
        """Score agent output using semantic similarity.

        ABC Compliance:
        - O.a.1: Considers semantically equivalent expressions
        - O.a.2: Handles redundant words through normalization

        Args:
            test_case: Test case with expected output
            actual_output: Agent's actual output

        Returns:
            MetricResult with similarity score and metadata

        Raises:
            ValueError: If expected output is missing
        """
        tracer = get_tracer()
        debug = get_debug_logger()

        if test_case.expected is None:
            raise ValueError("Semantic similarity metric requires 'expected' value in test case")

        with tracer.start_span("semantic_similarity_score") as span:
            metadata: dict[str, Any] = {
                "expected": test_case.expected,
                "actual": actual_output,
                "model": self.config.model,
            }

            tracer.set_attribute("semantic.model", self.config.model)
            tracer.set_attribute("semantic.threshold", self.config.threshold)
            tracer.set_attribute("semantic.normalize", self.config.normalize)

            # Handle None actual_output as a special case
            if actual_output is None:
                tracer.set_attribute("semantic.error", "actual_output_is_none")
                tracer.set_status("error", "actual_output is None")
                return MetricResult(
                    name="semantic_similarity",
                    score=0.0,
                    passed=False,
                    metadata={
                        **metadata,
                        "error": "actual_output is None",
                        "error_type": "NoneType",
                    },
                )

            try:
                # Normalize texts
                with tracer.start_span("text_normalization"):
                    expected_text = self._normalize_text(str(test_case.expected))
                    actual_text = self._normalize_text(str(actual_output))

                    metadata["expected_normalized"] = expected_text
                    metadata["actual_normalized"] = actual_text

                    debug.log_intermediate(
                        "semantic_similarity",
                        "Text normalization",
                        expected_original=str(test_case.expected)[:100],
                        expected_normalized=expected_text[:100],
                        actual_original=str(actual_output)[:100],
                        actual_normalized=actual_text[:100],
                    )

                # Compute similarity
                with tracer.start_span("embedding_generation") as embed_span:
                    similarity = self._compute_similarity(expected_text, actual_text)
                    metadata["similarity"] = similarity
                    metadata["threshold"] = self.config.threshold

                    if embed_span:
                        embed_span.set_attribute("similarity_score", similarity)

                    debug.log_intermediate(
                        "semantic_similarity",
                        "Similarity computation",
                        similarity=similarity,
                        threshold=self.config.threshold,
                        passed=similarity >= self.config.threshold,
                    )

                # Determine pass/fail
                passed = similarity >= self.config.threshold

                tracer.set_attribute("semantic.similarity", similarity)
                tracer.set_attribute("semantic.passed", passed)
                tracer.set_status("ok")

                if span:
                    span.set_attribute("final_score", similarity)
                    span.set_attribute("passed", passed)

                return MetricResult(
                    name="semantic_similarity",
                    score=similarity,
                    passed=passed,
                    metadata=metadata,
                )

            except Exception as e:
                # Handle errors gracefully
                tracer.record_exception(e)
                tracer.set_status("error", str(e))
                debug.log_error("semantic_similarity", e, test_case=str(test_case)[:100])

                return MetricResult(
                    name="semantic_similarity",
                    score=0.0,
                    passed=False,
                    metadata={
                        **metadata,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

    async def a_measure(self, test_case: TestCase) -> MetricResult:
        """Asynchronously score semantic similarity.

        Since semantic similarity uses ML model inference (sentence transformers),
        which can be CPU/GPU-intensive and block the event loop, this runs the
        sync version in a thread pool to avoid blocking.

        ABC Compliance:
        - O.a.1: Considers semantically equivalent expressions
        - O.a.2: Handles redundant words through normalization

        Args:
            test_case: Test case with expected output and actual_output

        Returns:
            MetricResult with similarity score and metadata

        Raises:
            ValueError: If expected output or actual_output is missing
        """
        if test_case.actual_output is None:
            raise ValueError("Semantic similarity metric requires 'actual_output' in test case")

        # Run ML inference (embedding generation) in thread pool
        return await asyncio.to_thread(self.score, test_case, test_case.actual_output)
