"""Tests for NLP evaluation metrics.

This module tests:
- BLEUMetric: Bilingual Evaluation Understudy score
- ROUGEMetric: Recall-Oriented Understudy for Gisting Evaluation

These are deterministic metrics - no mocking needed.
"""

import pytest

from evaris.types import TestCase


class TestBLEUMetric:
    """Tests for BLEUMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.nlp import BLEUConfig

        config = BLEUConfig()
        assert config.threshold == 0.5
        assert config.max_n == 4

    def test_metric_name(self) -> None:
        from evaris.metrics.nlp import BLEUMetric

        metric = BLEUMetric()
        assert metric.name == "BLEUMetric"

    def test_validate_inputs_missing_expected(self) -> None:
        from evaris.metrics.nlp import BLEUMetric

        metric = BLEUMetric()
        test_case = TestCase(input="Translate", expected=None)

        with pytest.raises(ValueError, match="expected"):
            metric.validate_inputs(test_case, "some output")

    @pytest.mark.asyncio
    async def test_perfect_match(self) -> None:
        from evaris.metrics.nlp import BLEUMetric

        metric = BLEUMetric()
        test_case = TestCase(input="Translate to English", expected="the cat sat on the mat")

        result = await metric.a_measure(test_case, "the cat sat on the mat")

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_partial_match(self) -> None:
        from evaris.metrics.nlp import BLEUConfig, BLEUMetric

        # Use BLEU-2 for partial match test since short sentences
        # don't have meaningful 4-gram overlap
        metric = BLEUMetric(config=BLEUConfig(max_n=2))
        test_case = TestCase(input="Translate", expected="the cat sat on the mat")

        result = await metric.a_measure(test_case, "the cat is on the mat")

        # Should have partial match with BLEU-2
        assert 0.0 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_no_match(self) -> None:
        from evaris.metrics.nlp import BLEUMetric

        metric = BLEUMetric()
        test_case = TestCase(input="Translate", expected="hello world")

        result = await metric.a_measure(test_case, "completely different text")

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_brevity_penalty(self) -> None:
        from evaris.metrics.nlp import BLEUMetric

        metric = BLEUMetric()
        test_case = TestCase(
            input="Translate", expected="the quick brown fox jumps over the lazy dog"
        )

        # Short candidate should have brevity penalty
        result = await metric.a_measure(test_case, "the fox")

        assert result.metadata["brevity_penalty"] < 1.0


class TestROUGEMetric:
    """Tests for ROUGEMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.nlp import ROUGEConfig

        config = ROUGEConfig()
        assert config.threshold == 0.5
        assert config.rouge_type == "rouge-l"
        assert config.use_f1 is True

    def test_metric_name(self) -> None:
        from evaris.metrics.nlp import ROUGEMetric

        metric = ROUGEMetric()
        assert metric.name == "ROUGEMetric"

    def test_validate_inputs_missing_expected(self) -> None:
        from evaris.metrics.nlp import ROUGEMetric

        metric = ROUGEMetric()
        test_case = TestCase(input="Summarize", expected=None)

        with pytest.raises(ValueError, match="expected"):
            metric.validate_inputs(test_case, "some output")

    @pytest.mark.asyncio
    async def test_rouge_l_perfect_match(self) -> None:
        from evaris.metrics.nlp import ROUGEMetric

        metric = ROUGEMetric()
        test_case = TestCase(input="Summarize", expected="the quick brown fox")

        result = await metric.a_measure(test_case, "the quick brown fox")

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata["precision"] == 1.0
        assert result.metadata["recall"] == 1.0

    @pytest.mark.asyncio
    async def test_rouge_1(self) -> None:
        from evaris.metrics.nlp import ROUGEConfig, ROUGEMetric

        metric = ROUGEMetric(config=ROUGEConfig(rouge_type="rouge-1"))
        test_case = TestCase(input="Summarize", expected="the quick brown fox jumps")

        result = await metric.a_measure(test_case, "the quick fox")

        # 3 unigrams match out of candidate's 3 (precision=1)
        # 3 unigrams match out of reference's 5 (recall=0.6)
        assert 0.0 < result.score < 1.0
        assert "rouge_1" in result.name

    @pytest.mark.asyncio
    async def test_rouge_2(self) -> None:
        from evaris.metrics.nlp import ROUGEConfig, ROUGEMetric

        metric = ROUGEMetric(config=ROUGEConfig(rouge_type="rouge-2"))
        test_case = TestCase(input="Summarize", expected="the quick brown fox")

        result = await metric.a_measure(test_case, "the quick brown fox")

        assert result.score == 1.0
        assert "rouge_2" in result.name

    @pytest.mark.asyncio
    async def test_rouge_l_subsequence(self) -> None:
        from evaris.metrics.nlp import ROUGEConfig, ROUGEMetric

        metric = ROUGEMetric(config=ROUGEConfig(rouge_type="rouge-l"))
        test_case = TestCase(input="Summarize", expected="the cat sat on the mat")

        # LCS should find "the" "on" "the" "mat" = 4
        result = await metric.a_measure(test_case, "the dog jumped on the mat")

        assert 0.0 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_rouge_no_match(self) -> None:
        from evaris.metrics.nlp import ROUGEMetric

        metric = ROUGEMetric()
        test_case = TestCase(input="Summarize", expected="hello world")

        result = await metric.a_measure(test_case, "goodbye universe")

        assert result.score == 0.0
        assert result.passed is False


class TestMETEORMetric:
    """Tests for METEORMetric.

    METEOR uses synonym matching and stemming for more flexible comparison.
    """

    def test_config_defaults(self) -> None:
        from evaris.metrics.nlp import METEORConfig

        config = METEORConfig()
        assert config.threshold == 0.5
        assert config.alpha == 0.9
        assert config.beta == 3.0
        assert config.gamma == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.nlp import METEORMetric

        metric = METEORMetric()
        assert metric.name == "METEORMetric"

    def test_validate_inputs_missing_expected(self) -> None:
        from evaris.metrics.nlp import METEORMetric

        metric = METEORMetric()
        test_case = TestCase(input="Translate", expected=None)

        with pytest.raises(ValueError, match="expected"):
            metric.validate_inputs(test_case, "some output")

    @pytest.mark.asyncio
    async def test_perfect_match(self) -> None:
        from evaris.metrics.nlp import METEORMetric

        metric = METEORMetric()
        test_case = TestCase(input="Translate to English", expected="the cat sat on the mat")

        result = await metric.a_measure(test_case, "the cat sat on the mat")

        # METEOR has a small fragmentation penalty even on perfect match
        # due to the chunk counting algorithm
        assert result.score > 0.99
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_partial_match(self) -> None:
        from evaris.metrics.nlp import METEORMetric

        metric = METEORMetric()
        test_case = TestCase(input="Translate", expected="the quick brown fox jumps")

        result = await metric.a_measure(test_case, "the fast brown fox leaps")

        # Should have partial match (some words match)
        assert 0.0 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_no_match(self) -> None:
        from evaris.metrics.nlp import METEORMetric

        metric = METEORMetric()
        test_case = TestCase(input="Translate", expected="hello world")

        result = await metric.a_measure(test_case, "goodbye universe")

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_word_order_penalty(self) -> None:
        from evaris.metrics.nlp import METEORMetric

        metric = METEORMetric()
        test_case = TestCase(input="Translate", expected="the cat sat on the mat")

        # Same words, different order should have lower score
        result = await metric.a_measure(test_case, "mat the on sat cat the")

        # Should match all words but with fragmentation penalty
        # When completely scrambled, score can be 0.5 (full penalty)
        assert 0.4 <= result.score <= 1.0
        assert result.metadata.get("fragmentation_penalty", 1.0) < 1.0


class TestNLPMetricsIntegration:
    """Integration tests for NLP metrics."""

    def test_all_nlp_metrics_importable(self) -> None:
        from evaris.metrics.nlp import (
            BLEUConfig,
            BLEUMetric,
            METEORConfig,
            METEORMetric,
            ROUGEConfig,
            ROUGEMetric,
        )

        assert BLEUMetric is not None
        assert ROUGEMetric is not None
        assert METEORMetric is not None

    def test_metrics_share_base_class(self) -> None:
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.nlp import BLEUMetric, METEORMetric, ROUGEMetric

        assert issubclass(BLEUMetric, BaseMetric)
        assert issubclass(ROUGEMetric, BaseMetric)
        assert issubclass(METEORMetric, BaseMetric)
