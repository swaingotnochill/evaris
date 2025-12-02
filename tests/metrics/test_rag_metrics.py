"""Tests for RAG evaluation metrics.

This module tests:
- AnswerRelevancyMetric: Measures if the answer addresses the input query
- ContextPrecisionMetric: Measures if relevant contexts rank higher
- ContextRecallMetric: Measures if all relevant info is retrieved

Following TDD: tests written before implementation.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.types import TestCase


class TestAnswerRelevancyMetric:
    """Tests for AnswerRelevancyMetric.

    Formula: Number of Relevant Statements / Total Number of Statements
    Required inputs: input, actual_output (referenceless - no expected needed)
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.rag import AnswerRelevancyConfig

        config = AnswerRelevancyConfig()
        assert config.threshold == 0.5
        assert config.include_reasoning is True

    def test_config_custom(self) -> None:
        """Test custom configuration."""
        from evaris.metrics.rag import AnswerRelevancyConfig

        config = AnswerRelevancyConfig(
            threshold=0.7,
            include_reasoning=False,
        )
        assert config.threshold == 0.7
        assert config.include_reasoning is False

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        assert metric.name == "AnswerRelevancyMetric"

    def test_validate_inputs_missing_input(self) -> None:
        """Test validation fails when input is missing."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        test_case = TestCase(input=None, expected=None)

        with pytest.raises(ValueError, match="input"):
            metric.validate_inputs(test_case, "some output")

    def test_validate_inputs_missing_actual_output(self) -> None:
        """Test validation fails when actual_output is missing."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        test_case = TestCase(input="What is Python?", expected=None)

        with pytest.raises(ValueError, match="actual_output"):
            metric.validate_inputs(test_case, None)

    def test_validate_inputs_success(self) -> None:
        """Test validation passes with valid inputs."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric()
        test_case = TestCase(input="What is Python?", expected=None)

        # Should not raise
        metric.validate_inputs(test_case, "Python is a programming language.")

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.answer_relevancy.get_provider")
    async def test_measure_high_relevance(self, mock_get_provider: Any) -> None:
        """Test scoring when all statements are relevant."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        # Mock LLM responses for statement extraction and relevancy check
        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            side_effect=[
                # First call: extract statements
                MagicMock(
                    content='{"statements": ["Python is a programming language", "It is widely used"]}'
                ),
                # Second call: check relevancy
                MagicMock(
                    content='{"verdicts": [{"statement": "Python is a programming language", "verdict": "yes", "reason": "Directly answers the question"}, {"statement": "It is widely used", "verdict": "yes", "reason": "Relevant context"}]}'
                ),
            ]
        )
        mock_get_provider.return_value = mock_provider

        metric = AnswerRelevancyMetric()
        test_case = TestCase(input="What is Python?", expected=None)

        result = await metric.a_measure(
            test_case, "Python is a programming language. It is widely used."
        )

        assert result.score == 1.0
        assert result.passed is True
        assert result.name == "answer_relevancy"

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.answer_relevancy.get_provider")
    async def test_measure_partial_relevance(self, mock_get_provider: Any) -> None:
        """Test scoring when some statements are irrelevant."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            side_effect=[
                # Extract 4 statements
                MagicMock(
                    content='{"statements": ["Python is a language", "The sky is blue", "It supports OOP", "I like pizza"]}'
                ),
                # 2 relevant, 2 irrelevant
                MagicMock(
                    content='{"verdicts": [{"statement": "Python is a language", "verdict": "yes"}, {"statement": "The sky is blue", "verdict": "no"}, {"statement": "It supports OOP", "verdict": "yes"}, {"statement": "I like pizza", "verdict": "no"}]}'
                ),
            ]
        )
        mock_get_provider.return_value = mock_provider

        metric = AnswerRelevancyMetric()
        test_case = TestCase(input="What is Python?", expected=None)

        result = await metric.a_measure(test_case, "Mixed relevance output")

        assert result.score == 0.5  # 2/4 statements relevant
        assert result.passed is True  # >= 0.5 threshold

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.answer_relevancy.get_provider")
    async def test_measure_no_relevance(self, mock_get_provider: Any) -> None:
        """Test scoring when no statements are relevant."""
        from evaris.metrics.rag import AnswerRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            side_effect=[
                MagicMock(content='{"statements": ["The sky is blue", "Water is wet"]}'),
                MagicMock(
                    content='{"verdicts": [{"statement": "The sky is blue", "verdict": "no"}, {"statement": "Water is wet", "verdict": "no"}]}'
                ),
            ]
        )
        mock_get_provider.return_value = mock_provider

        metric = AnswerRelevancyMetric()
        test_case = TestCase(input="What is Python?", expected=None)

        result = await metric.a_measure(test_case, "Irrelevant output")

        assert result.score == 0.0
        assert result.passed is False


class TestContextPrecisionMetric:
    """Tests for ContextPrecisionMetric.

    Formula: Weighted cumulative precision favoring higher-ranked nodes.
    Required inputs: input, actual_output, expected_output, retrieval_context
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.rag import ContextPrecisionConfig

        config = ContextPrecisionConfig()
        assert config.threshold == 0.5
        assert config.context_key == "retrieval_context"

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.rag import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        assert metric.name == "ContextPrecisionMetric"

    def test_validate_inputs_missing_context(self) -> None:
        """Test validation fails when retrieval_context is missing."""
        from evaris.metrics.rag import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        test_case = TestCase(
            input="What is Python?", expected="Python is a programming language", metadata={}
        )

        with pytest.raises(ValueError, match="retrieval_context"):
            metric.validate_inputs(test_case, "some output")

    def test_validate_inputs_missing_expected(self) -> None:
        """Test validation fails when expected output is missing."""
        from evaris.metrics.rag import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        test_case = TestCase(
            input="What is Python?",
            expected=None,
            metadata={"retrieval_context": ["context 1", "context 2"]},
        )

        with pytest.raises(ValueError, match="expected"):
            metric.validate_inputs(test_case, "some output")

    def test_calculate_weighted_precision_all_relevant_top(self) -> None:
        """Test weighted precision when all relevant nodes are at top."""
        from evaris.metrics.rag import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        # [relevant, relevant, irrelevant, irrelevant]
        verdicts = [True, True, False, False]

        score = metric._calculate_weighted_precision(verdicts)

        # Perfect ranking: relevant items first
        # Position 1: 1/1 * 1 = 1
        # Position 2: 2/2 * 1 = 1
        # Sum = 2, num_relevant = 2
        # Score = 2/2 = 1.0
        assert score == 1.0

    def test_calculate_weighted_precision_mixed_order(self) -> None:
        """Test weighted precision with suboptimal ranking."""
        from evaris.metrics.rag import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        # [relevant, irrelevant, relevant, irrelevant]
        verdicts = [True, False, True, False]

        score = metric._calculate_weighted_precision(verdicts)

        # Position 1: 1/1 * 1 = 1.0
        # Position 2: irrelevant, skip
        # Position 3: 2/3 * 1 = 0.667
        # Sum = 1.667, num_relevant = 2
        # Score = 1.667/2 = 0.833
        assert 0.8 < score < 0.9

    def test_calculate_weighted_precision_all_irrelevant_first(self) -> None:
        """Test weighted precision when relevant nodes are at bottom."""
        from evaris.metrics.rag import ContextPrecisionMetric

        metric = ContextPrecisionMetric()
        # [irrelevant, irrelevant, relevant, relevant]
        verdicts = [False, False, True, True]

        score = metric._calculate_weighted_precision(verdicts)

        # Position 3: 1/3 * 1 = 0.333
        # Position 4: 2/4 * 1 = 0.5
        # Sum = 0.833, num_relevant = 2
        # Score = 0.833/2 = 0.417
        assert 0.4 < score < 0.5

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.context_precision.get_provider")
    async def test_measure_perfect_ranking(self, mock_get_provider: Any) -> None:
        """Test scoring with perfect context ranking."""
        from evaris.metrics.rag import ContextPrecisionMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"verdicts": [{"node": 0, "verdict": "yes"}, {"node": 1, "verdict": "yes"}, {"node": 2, "verdict": "no"}]}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextPrecisionMetric()
        test_case = TestCase(
            input="What is Python?",
            expected="Python is a programming language",
            metadata={
                "retrieval_context": [
                    "Python is a high-level programming language",
                    "Python supports multiple paradigms",
                    "The weather is nice today",
                ]
            },
        )

        result = await metric.a_measure(test_case, "Python is a programming language")

        assert result.score == 1.0
        assert result.passed is True


class TestContextRecallMetric:
    """Tests for ContextRecallMetric.

    Formula: Number of Attributable Statements / Total Statements in expected_output
    Required inputs: input, actual_output, expected_output, retrieval_context
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.rag import ContextRecallConfig

        config = ContextRecallConfig()
        assert config.threshold == 0.5
        assert config.context_key == "retrieval_context"

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.rag import ContextRecallMetric

        metric = ContextRecallMetric()
        assert metric.name == "ContextRecallMetric"

    def test_validate_inputs_missing_context(self) -> None:
        """Test validation fails when retrieval_context is missing."""
        from evaris.metrics.rag import ContextRecallMetric

        metric = ContextRecallMetric()
        test_case = TestCase(
            input="What is Python?", expected="Python is a programming language", metadata={}
        )

        with pytest.raises(ValueError, match="retrieval_context"):
            metric.validate_inputs(test_case, "some output")

    def test_validate_inputs_missing_expected(self) -> None:
        """Test validation fails when expected output is missing."""
        from evaris.metrics.rag import ContextRecallMetric

        metric = ContextRecallMetric()
        test_case = TestCase(
            input="What is Python?", expected=None, metadata={"retrieval_context": ["context 1"]}
        )

        with pytest.raises(ValueError, match="expected"):
            metric.validate_inputs(test_case, "some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.context_recall.get_provider")
    async def test_measure_full_recall(self, mock_get_provider: Any) -> None:
        """Test scoring when all expected statements are in context."""
        from evaris.metrics.rag import ContextRecallMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            side_effect=[
                # Extract statements from expected
                MagicMock(
                    content='{"statements": ["Python is a language", "Python is high-level"]}'
                ),
                # Check attribution
                MagicMock(
                    content='{"verdicts": [{"statement": "Python is a language", "attributed": "yes"}, {"statement": "Python is high-level", "attributed": "yes"}]}'
                ),
            ]
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextRecallMetric()
        test_case = TestCase(
            input="What is Python?",
            expected="Python is a high-level language",
            metadata={
                "retrieval_context": [
                    "Python is a high-level programming language",
                    "It was created by Guido van Rossum",
                ]
            },
        )

        result = await metric.a_measure(test_case, "actual output")

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.context_recall.get_provider")
    async def test_measure_partial_recall(self, mock_get_provider: Any) -> None:
        """Test scoring when only some expected statements are in context."""
        from evaris.metrics.rag import ContextRecallMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            side_effect=[
                MagicMock(
                    content='{"statements": ["Python is a language", "It was created in 1991", "It has a large community"]}'
                ),
                MagicMock(
                    content='{"verdicts": [{"statement": "Python is a language", "attributed": "yes"}, {"statement": "It was created in 1991", "attributed": "no"}, {"statement": "It has a large community", "attributed": "no"}]}'
                ),
            ]
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextRecallMetric()
        test_case = TestCase(
            input="What is Python?",
            expected="Python is a language created in 1991 with a large community",
            metadata={"retrieval_context": ["Python is a programming language"]},
        )

        result = await metric.a_measure(test_case, "actual output")

        assert abs(result.score - 0.333) < 0.01  # 1/3 statements attributed
        assert result.passed is False

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.context_recall.get_provider")
    async def test_measure_no_recall(self, mock_get_provider: Any) -> None:
        """Test scoring when no expected statements are in context."""
        from evaris.metrics.rag import ContextRecallMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            side_effect=[
                MagicMock(
                    content='{"statements": ["Python is interpreted", "Python is dynamically typed"]}'
                ),
                MagicMock(
                    content='{"verdicts": [{"statement": "Python is interpreted", "attributed": "no"}, {"statement": "Python is dynamically typed", "attributed": "no"}]}'
                ),
            ]
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextRecallMetric()
        test_case = TestCase(
            input="What is Python?",
            expected="Python is interpreted and dynamically typed",
            metadata={"retrieval_context": ["The weather is nice today"]},
        )

        result = await metric.a_measure(test_case, "actual output")

        assert result.score == 0.0
        assert result.passed is False


class TestContextualRelevancyMetric:
    """Tests for ContextualRelevancyMetric.

    Formula: Relevant Sentences in Context / Total Sentences in Context
    Required inputs: input, retrieval_context in metadata
    This evaluates how relevant retrieved context is to the input query.
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.rag import ContextualRelevancyConfig

        config = ContextualRelevancyConfig()
        assert config.threshold == 0.5
        assert config.context_key == "retrieval_context"

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.rag import ContextualRelevancyMetric

        metric = ContextualRelevancyMetric()
        assert metric.name == "ContextualRelevancyMetric"

    def test_validate_inputs_missing_context(self) -> None:
        """Test validation fails when retrieval_context is missing."""
        from evaris.metrics.rag import ContextualRelevancyMetric

        metric = ContextualRelevancyMetric()
        test_case = TestCase(input="What is Python?", expected=None, metadata={})

        with pytest.raises(ValueError, match="retrieval_context"):
            metric.validate_inputs(test_case, "some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.contextual_relevancy.get_provider")
    async def test_measure_all_relevant(self, mock_get_provider: Any) -> None:
        """Test scoring when all context sentences are relevant."""
        from evaris.metrics.rag import ContextualRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"verdicts": [{"context_index": 0, "verdict": "yes"}, {"context_index": 1, "verdict": "yes"}], "relevant_count": 2, "total_count": 2}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextualRelevancyMetric()
        test_case = TestCase(
            input="What is Python?",
            expected=None,
            metadata={
                "retrieval_context": [
                    "Python is a programming language",
                    "Python is used for web development",
                ]
            },
        )

        result = await metric.a_measure(test_case, "Python is a language")

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.contextual_relevancy.get_provider")
    async def test_measure_partial_relevance(self, mock_get_provider: Any) -> None:
        """Test scoring when only some context is relevant."""
        from evaris.metrics.rag import ContextualRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"verdicts": [{"context_index": 0, "verdict": "yes"}, {"context_index": 1, "verdict": "no"}], "relevant_count": 1, "total_count": 2}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextualRelevancyMetric()
        test_case = TestCase(
            input="What is Python?",
            expected=None,
            metadata={
                "retrieval_context": [
                    "Python is a programming language",
                    "The weather is sunny today",
                ]
            },
        )

        result = await metric.a_measure(test_case, "Python is a language")

        assert result.score == 0.5
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.contextual_relevancy.get_provider")
    async def test_measure_no_relevance(self, mock_get_provider: Any) -> None:
        """Test scoring when no context is relevant."""
        from evaris.metrics.rag import ContextualRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"verdicts": [{"context_index": 0, "verdict": "no"}, {"context_index": 1, "verdict": "no"}], "relevant_count": 0, "total_count": 2}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ContextualRelevancyMetric()
        test_case = TestCase(
            input="What is Python?",
            expected=None,
            metadata={"retrieval_context": ["The sky is blue", "Cats like to sleep"]},
        )

        result = await metric.a_measure(test_case, "Python is a language")

        assert result.score == 0.0
        assert result.passed is False


class TestRAGASMetric:
    """Tests for RAGASMetric.

    RAGAS is a composite metric that combines:
    - AnswerRelevancy
    - Faithfulness
    - ContextPrecision
    - ContextRecall

    Formula: Average of all component scores
    """

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from evaris.metrics.rag import RAGASConfig

        config = RAGASConfig()
        assert config.threshold == 0.5
        assert config.include_answer_relevancy is True
        assert config.include_faithfulness is True
        assert config.include_context_precision is True
        assert config.include_context_recall is True

    def test_metric_name(self) -> None:
        """Test metric has correct name."""
        from evaris.metrics.rag import RAGASMetric

        metric = RAGASMetric()
        assert metric.name == "RAGASMetric"

    def test_validate_inputs_missing_context(self) -> None:
        """Test validation fails when retrieval_context is missing."""
        from evaris.metrics.rag import RAGASMetric

        metric = RAGASMetric()
        test_case = TestCase(
            input="What is Python?", expected="Python is a programming language", metadata={}
        )

        with pytest.raises(ValueError, match="retrieval_context"):
            metric.validate_inputs(test_case, "some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.context_recall.get_provider")
    @patch("evaris.metrics.rag.context_precision.get_provider")
    @patch("evaris.metrics.rag.answer_relevancy.get_provider")
    @patch("evaris.metrics.rag.ragas.get_provider")
    async def test_measure_all_high_scores(
        self,
        mock_ragas_provider: Any,
        mock_ar_provider: Any,
        mock_cp_provider: Any,
        mock_cr_provider: Any,
    ) -> None:
        """Test scoring when all component metrics score high."""
        from evaris.metrics.rag import RAGASMetric

        # Mock for Answer Relevancy
        ar_mock = MagicMock()
        ar_mock.a_complete = AsyncMock(
            side_effect=[
                MagicMock(content='{"statements": ["Python is a language"]}'),
                MagicMock(content='{"verdicts": [{"verdict": "yes"}]}'),
            ]
        )
        mock_ar_provider.return_value = ar_mock

        # Mock for Faithfulness (in ragas module)
        faith_mock = MagicMock()
        faith_mock.a_complete = AsyncMock(
            return_value=MagicMock(content='{"score": 1.0, "reasoning": "Faithful"}')
        )
        mock_ragas_provider.return_value = faith_mock

        # Mock for Context Precision
        cp_mock = MagicMock()
        cp_mock.a_complete = AsyncMock(
            return_value=MagicMock(content='{"verdicts": [{"node": 0, "verdict": "yes"}]}')
        )
        mock_cp_provider.return_value = cp_mock

        # Mock for Context Recall
        cr_mock = MagicMock()
        cr_mock.a_complete = AsyncMock(
            side_effect=[
                MagicMock(content='{"statements": ["Python is a language"]}'),
                MagicMock(content='{"verdicts": [{"attributed": "yes"}]}'),
            ]
        )
        mock_cr_provider.return_value = cr_mock

        metric = RAGASMetric()
        test_case = TestCase(
            input="What is Python?",
            expected="Python is a language",
            metadata={
                "retrieval_context": ["Python is a programming language"],
                "context": "Python is a programming language",
            },
        )

        result = await metric.a_measure(test_case, "Python is a language")

        assert result.score == 1.0
        assert result.passed is True
        assert "component_scores" in result.metadata

    @pytest.mark.asyncio
    @patch("evaris.metrics.rag.context_recall.get_provider")
    @patch("evaris.metrics.rag.context_precision.get_provider")
    @patch("evaris.metrics.rag.answer_relevancy.get_provider")
    @patch("evaris.metrics.rag.ragas.get_provider")
    async def test_measure_mixed_scores(
        self,
        mock_ragas_provider: Any,
        mock_ar_provider: Any,
        mock_cp_provider: Any,
        mock_cr_provider: Any,
    ) -> None:
        """Test scoring with mixed component scores."""
        from evaris.metrics.rag import RAGASMetric

        # Mock for Answer Relevancy: 0.5 (1/2 relevant)
        ar_mock = MagicMock()
        ar_mock.a_complete = AsyncMock(
            side_effect=[
                MagicMock(content='{"statements": ["S1", "S2"]}'),
                MagicMock(content='{"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]}'),
            ]
        )
        mock_ar_provider.return_value = ar_mock

        # Mock for Faithfulness: 0.8
        faith_mock = MagicMock()
        faith_mock.a_complete = AsyncMock(
            return_value=MagicMock(content='{"score": 0.8, "reasoning": "Mostly faithful"}')
        )
        mock_ragas_provider.return_value = faith_mock

        # Mock for Context Precision: 0.5 (mixed relevance)
        cp_mock = MagicMock()
        cp_mock.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"verdicts": [{"node": 0, "verdict": "yes"}, {"node": 1, "verdict": "no"}]}'
            )
        )
        mock_cp_provider.return_value = cp_mock

        # Mock for Context Recall: 1.0 (all attributed)
        cr_mock = MagicMock()
        cr_mock.a_complete = AsyncMock(
            side_effect=[
                MagicMock(content='{"statements": ["S1"]}'),
                MagicMock(content='{"verdicts": [{"attributed": "yes"}]}'),
            ]
        )
        mock_cr_provider.return_value = cr_mock

        metric = RAGASMetric()
        test_case = TestCase(
            input="What is Python?",
            expected="Python is a language",
            metadata={"retrieval_context": ["ctx1", "ctx2"], "context": "ctx1"},
        )

        result = await metric.a_measure(test_case, "Mixed output")

        # Average of 0.5, 0.8, 1.0, 1.0 = 0.825
        assert 0.8 < result.score < 0.9
        assert result.passed is True

    def test_config_selective_metrics(self) -> None:
        """Test RAGAS can be configured to use only some metrics."""
        from evaris.metrics.rag import RAGASConfig

        config = RAGASConfig(
            include_answer_relevancy=True,
            include_faithfulness=False,
            include_context_precision=True,
            include_context_recall=False,
        )

        assert config.include_answer_relevancy is True
        assert config.include_faithfulness is False
        assert config.include_context_precision is True
        assert config.include_context_recall is False


class TestRAGMetricsIntegration:
    """Integration tests for RAG metrics working together."""

    def test_all_rag_metrics_importable(self) -> None:
        """Test all RAG metrics can be imported from evaris.metrics.rag."""
        from evaris.metrics.rag import (
            AnswerRelevancyConfig,
            AnswerRelevancyMetric,
            ContextPrecisionConfig,
            ContextPrecisionMetric,
            ContextRecallConfig,
            ContextRecallMetric,
            ContextualRelevancyConfig,
            ContextualRelevancyMetric,
            RAGASConfig,
            RAGASMetric,
        )

        # Verify all classes are properly defined
        assert AnswerRelevancyMetric is not None
        assert ContextPrecisionMetric is not None
        assert ContextRecallMetric is not None
        assert ContextualRelevancyMetric is not None
        assert RAGASMetric is not None

    def test_metrics_share_base_class(self) -> None:
        """Test all RAG metrics inherit from BaseMetric."""
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.rag import (
            AnswerRelevancyMetric,
            ContextPrecisionMetric,
            ContextRecallMetric,
            ContextualRelevancyMetric,
            RAGASMetric,
        )

        assert issubclass(AnswerRelevancyMetric, BaseMetric)
        assert issubclass(ContextPrecisionMetric, BaseMetric)
        assert issubclass(ContextRecallMetric, BaseMetric)
        assert issubclass(ContextualRelevancyMetric, BaseMetric)
        assert issubclass(RAGASMetric, BaseMetric)
