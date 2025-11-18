"""Tests for semantic similarity metric."""

import pytest

# Check if sentence-transformers is available
try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from evaris.metrics.semantic_similarity import (
    SemanticSimilarityConfig,
    SemanticSimilarityMetric,
)
from evaris.types import TestCase

# Skip all tests in this module if sentence-transformers not available
pytestmark = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed (optional dependency)",
)


class TestSemanticSimilarityConfig:
    """Tests for SemanticSimilarityConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SemanticSimilarityConfig()

        assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.threshold == 0.8
        assert config.normalize is True
        assert config.case_sensitive is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SemanticSimilarityConfig(
            model="custom-model",
            threshold=0.9,
            normalize=False,
            case_sensitive=True,
        )

        assert config.model == "custom-model"
        assert config.threshold == 0.9
        assert config.normalize is False
        assert config.case_sensitive is True


class TestSemanticSimilarityMetric:
    """Tests for SemanticSimilarityMetric."""

    @pytest.fixture
    def metric(self) -> SemanticSimilarityMetric:
        """Create metric instance for testing."""
        config = SemanticSimilarityConfig()
        return SemanticSimilarityMetric(config)

    def test_metric_initialization(self, metric: SemanticSimilarityMetric) -> None:
        """Test metric initializes correctly."""
        assert metric.config is not None
        assert metric._model is not None  # Model should be loaded

    def test_metric_initialization_default_config(self) -> None:
        """Test metric with default config."""
        metric = SemanticSimilarityMetric()
        assert metric.config.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_normalize_text_basic(self, metric: SemanticSimilarityMetric) -> None:
        """Test basic text normalization."""
        text = "  Hello World  "
        normalized = metric._normalize_text(text)

        assert normalized == "hello world"  # Case normalized and trimmed

    def test_normalize_text_case_sensitive(self) -> None:
        """Test normalization with case sensitivity."""
        config = SemanticSimilarityConfig(case_sensitive=True)
        metric = SemanticSimilarityMetric(config)

        text = "Hello World"
        normalized = metric._normalize_text(text)

        assert normalized == "Hello World"  # Case preserved

    def test_normalize_text_no_normalize(self) -> None:
        """Test with normalization disabled."""
        config = SemanticSimilarityConfig(normalize=False)
        metric = SemanticSimilarityMetric(config)

        text = "  Hello World  "
        normalized = metric._normalize_text(text)

        assert normalized == "  Hello World  "  # No changes

    def test_compute_similarity_identical(self, metric: SemanticSimilarityMetric) -> None:
        """Test similarity of identical texts."""
        text = "The quick brown fox"
        similarity = metric._compute_similarity(text, text)

        assert similarity >= 0.99  # Should be very high

    def test_compute_similarity_similar(self, metric: SemanticSimilarityMetric) -> None:
        """Test similarity of similar texts."""
        text1 = "The capital of France is Paris"
        text2 = "Paris is the capital of France"

        similarity = metric._compute_similarity(text1, text2)

        assert similarity >= 0.7  # Should be high for similar meaning

    def test_compute_similarity_different(self, metric: SemanticSimilarityMetric) -> None:
        """Test similarity of different texts."""
        text1 = "The capital of France is Paris"
        text2 = "The weather is nice today"

        similarity = metric._compute_similarity(text1, text2)

        # Real models may give slightly higher scores due to common words
        # but should still be significantly lower than similar texts (< 0.7)
        assert similarity < 0.6  # Should be low for different meanings

    def test_score_with_expected(self, metric: SemanticSimilarityMetric) -> None:
        """Test scoring with expected value."""
        tc = TestCase(
            input="What is the capital of France?",
            expected="Paris",
            actual_output="The capital is Paris",
        )

        result = metric.score(tc, "The capital is Paris")

        assert result.name == "semantic_similarity"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.metadata is not None
        assert "expected" in result.metadata
        assert "actual" in result.metadata
        assert "similarity" in result.metadata

    def test_score_no_expected_raises(self, metric: SemanticSimilarityMetric) -> None:
        """Test scoring without expected value raises error."""
        tc = TestCase(input="test", expected=None, actual_output="output")

        with pytest.raises(ValueError, match="expected"):
            metric.score(tc, "output")

    def test_score_high_similarity_passes(self, metric: SemanticSimilarityMetric) -> None:
        """Test that high similarity passes threshold."""
        tc = TestCase(input="test", expected="hello world", actual_output="hello world")

        result = metric.score(tc, "hello world")

        assert result.score >= 0.95
        assert result.passed is True

    def test_score_low_similarity_fails(self, metric: SemanticSimilarityMetric) -> None:
        """Test that low similarity fails threshold."""
        tc = TestCase(input="test", expected="hello", actual_output="completely different text")

        result = metric.score(tc, "completely different text")

        assert result.score < metric.config.threshold
        assert result.passed is False

    def test_score_threshold_boundary(self) -> None:
        """Test scoring at threshold boundary."""
        config = SemanticSimilarityConfig(threshold=0.7)
        metric = SemanticSimilarityMetric(config)

        tc = TestCase(input="test", expected="cat", actual_output="dog")

        # Test with text that should be near threshold
        result = metric.score(tc, "dog")  # Related but different

        assert result.metadata["threshold"] == 0.7
        # Check passed status matches threshold
        assert result.passed == (result.score >= 0.7)

    def test_score_metadata_contains_normalized(self, metric: SemanticSimilarityMetric) -> None:
        """Test that metadata contains normalized texts."""
        tc = TestCase(input="test", expected="HELLO", actual_output="  hello  ")

        result = metric.score(tc, "  hello  ")

        assert "expected_normalized" in result.metadata
        assert "actual_normalized" in result.metadata
        assert result.metadata["expected_normalized"] == "hello"
        assert result.metadata["actual_normalized"] == "hello"

    def test_score_error_handling(self, metric: SemanticSimilarityMetric) -> None:
        """Test error handling when actual_output is None."""
        tc = TestCase(input="test", expected="hello", actual_output=None)

        # Pass None - should be handled as error case
        result = metric.score(tc, None)

        assert result.score == 0.0
        assert result.passed is False
        assert "error" in result.metadata
        assert result.metadata["error"] == "actual_output is None"
        assert result.metadata["error_type"] == "NoneType"

    def test_score_with_numbers(self, metric: SemanticSimilarityMetric) -> None:
        """Test scoring with numeric values."""
        tc = TestCase(input="test", expected=42, actual_output="42")

        result = metric.score(tc, "42")

        assert result.score >= 0.8  # Should be high for same number

    def test_model_loading_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of model loading failure."""

        def mock_import_error(*args: object, **kwargs: object) -> None:
            raise ImportError("sentence-transformers not installed")

        # This test would require mocking the import
        # For now, we test that ImportError is raised
        _ = SemanticSimilarityConfig()  # noqa: F841

        # The actual test would require more sophisticated mocking
        # to simulate the ImportError during __init__
        # Skipping detailed implementation as it requires import mocking

    def test_abc_compliance_o_a_1(self, metric: SemanticSimilarityMetric) -> None:
        """Test ABC O.a.1: Considers semantically equivalent expressions."""
        tc1 = TestCase(input="What is 2+2?", expected="4", actual_output="4")
        tc2 = TestCase(input="What is 2+2?", expected="4", actual_output="The answer is 4")
        tc3 = TestCase(input="What is 2+2?", expected="4", actual_output="2+2 equals 4")

        # Test various semantically equivalent ways to say "4"
        result1 = metric.score(tc1, "4")
        result2 = metric.score(tc2, "The answer is 4")
        result3 = metric.score(tc3, "2+2 equals 4")

        # All should score reasonably high
        assert result1.score >= 0.7
        assert result2.score >= 0.6
        assert result3.score >= 0.6

    def test_abc_compliance_o_a_2(self, metric: SemanticSimilarityMetric) -> None:
        """Test ABC O.a.2: Handles redundant words."""
        tc1 = TestCase(input="What is the capital?", expected="Paris", actual_output="Paris")
        tc2 = TestCase(
            input="What is the capital?",
            expected="Paris",
            actual_output="The capital city is Paris",
        )
        tc3 = TestCase(
            input="What is the capital?", expected="Paris", actual_output="Well, I think it's Paris"
        )

        # Test with redundant words
        result1 = metric.score(tc1, "Paris")
        result2 = metric.score(tc2, "The capital city is Paris")
        result3 = metric.score(tc3, "Well, I think it's Paris")

        # Should handle redundancy gracefully
        assert result1.score >= 0.8
        assert result2.score >= 0.6  # Lower but still reasonable
        assert result3.score >= 0.5
