"""Tests for Quality evaluation metrics.

This module tests:
- HallucinationMetric: Detects factual hallucinations
- JsonCorrectnessMetric: Validates JSON output
- SummarizationMetric: Evaluates summary quality
- GEvalMetric: Custom criteria evaluation
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.types import TestCase


class TestHallucinationMetric:
    """Tests for HallucinationMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.quality import HallucinationConfig

        config = HallucinationConfig()
        assert config.threshold == 0.5
        assert config.context_key == "context"

    def test_metric_name(self) -> None:
        from evaris.metrics.quality import HallucinationMetric

        metric = HallucinationMetric()
        assert metric.name == "HallucinationMetric"

    @pytest.mark.asyncio
    @patch("evaris.metrics.quality.hallucination.get_provider")
    async def test_measure_no_hallucination(self, mock_get_provider: Any) -> None:
        from evaris.metrics.quality import HallucinationMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"claims": [{"claim": "Python is a programming language", "hallucinated": "no"}], "hallucination_count": 0, "total_claims": 1}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = HallucinationMetric()
        test_case = TestCase(input="What is Python?", expected=None)

        result = await metric.a_measure(test_case, "Python is a programming language.")

        assert result.score == 0.0
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.quality.hallucination.get_provider")
    async def test_measure_with_hallucination(self, mock_get_provider: Any) -> None:
        from evaris.metrics.quality import HallucinationMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"claims": [{"claim": "Python was created in 2020", "hallucinated": "yes"}], "hallucination_count": 1, "total_claims": 1}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = HallucinationMetric()
        test_case = TestCase(input="When was Python created?", expected=None)

        result = await metric.a_measure(test_case, "Python was created in 2020.")

        assert result.score == 1.0
        assert result.passed is False


class TestJsonCorrectnessMetric:
    """Tests for JsonCorrectnessMetric - deterministic, no mocking needed."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessConfig

        config = JsonCorrectnessConfig()
        assert config.threshold == 1.0

    def test_metric_name(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        assert metric.name == "JsonCorrectnessMetric"

    @pytest.mark.asyncio
    async def test_valid_json(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        test_case = TestCase(input="Generate JSON", expected=None)

        result = await metric.a_measure(test_case, '{"name": "test", "value": 123}')

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata["valid_json"] is True

    @pytest.mark.asyncio
    async def test_invalid_json(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        test_case = TestCase(input="Generate JSON", expected=None)

        result = await metric.a_measure(test_case, "not valid json {")

        assert result.score == 0.0
        assert result.passed is False
        assert result.metadata["valid_json"] is False

    @pytest.mark.asyncio
    async def test_json_in_code_block(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        test_case = TestCase(input="Generate JSON", expected=None)

        result = await metric.a_measure(test_case, '```json\n{"key": "value"}\n```')

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_json_with_expected_keys(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        test_case = TestCase(
            input="Generate JSON", expected=None, metadata={"expected_keys": ["name", "age"]}
        )

        result = await metric.a_measure(test_case, '{"name": "John", "age": 30}')

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_json_missing_keys(self) -> None:
        from evaris.metrics.quality import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        test_case = TestCase(
            input="Generate JSON",
            expected=None,
            metadata={"expected_keys": ["name", "age", "email"]},
        )

        result = await metric.a_measure(test_case, '{"name": "John"}')

        assert result.score < 1.0  # Only 1/3 keys present
        assert "missing_keys" in result.metadata


class TestSummarizationMetric:
    """Tests for SummarizationMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.quality import SummarizationConfig

        config = SummarizationConfig()
        assert config.threshold == 0.5
        assert config.source_key == "source_text"

    def test_metric_name(self) -> None:
        from evaris.metrics.quality import SummarizationMetric

        metric = SummarizationMetric()
        assert metric.name == "SummarizationMetric"

    def test_validate_inputs_missing_source(self) -> None:
        from evaris.metrics.quality import SummarizationMetric

        metric = SummarizationMetric()
        test_case = TestCase(input="Summarize", expected=None, metadata={})

        with pytest.raises(ValueError, match="source_text"):
            metric.validate_inputs(test_case, "Some summary")

    @pytest.mark.asyncio
    @patch("evaris.metrics.quality.summarization.get_provider")
    async def test_measure_good_summary(self, mock_get_provider: Any) -> None:
        from evaris.metrics.quality import SummarizationMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"coverage": 0.9, "conciseness": 0.8, "accuracy": 0.95, "overall_score": 0.88, "reasoning": "Good summary"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = SummarizationMetric()
        test_case = TestCase(
            input="Summarize",
            expected=None,
            metadata={"source_text": "Long text to be summarized..."},
        )

        result = await metric.a_measure(test_case, "Brief summary.")

        assert result.score == 0.88
        assert result.passed is True


class TestGEvalMetric:
    """Tests for GEvalMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.quality import GEvalConfig

        config = GEvalConfig()
        assert config.threshold == 0.5
        assert config.criteria == ""

    def test_metric_name(self) -> None:
        from evaris.metrics.quality import GEvalConfig, GEvalMetric

        metric = GEvalMetric(config=GEvalConfig(criteria="Test criteria"))
        assert metric.name == "GEvalMetric"

    def test_validate_inputs_missing_criteria(self) -> None:
        from evaris.metrics.quality import GEvalMetric

        metric = GEvalMetric()  # No criteria set
        test_case = TestCase(input="Test", expected=None)

        with pytest.raises(ValueError, match="criteria"):
            metric.validate_inputs(test_case, "Output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.quality.g_eval.get_provider")
    async def test_measure_with_criteria(self, mock_get_provider: Any) -> None:
        from evaris.metrics.quality import GEvalConfig, GEvalMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.85, "reasoning": "Response is polite", "criteria_met": ["politeness"], "criteria_not_met": []}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = GEvalMetric(config=GEvalConfig(criteria="Evaluate for politeness"))
        test_case = TestCase(input="How are you?", expected=None)

        result = await metric.a_measure(test_case, "I am doing well, thank you for asking!")

        assert result.score == 0.85
        assert result.passed is True
        assert result.metadata["criteria"] == "Evaluate for politeness"


class TestQualityMetricsIntegration:
    """Integration tests for Quality metrics."""

    def test_all_quality_metrics_importable(self) -> None:
        from evaris.metrics.quality import (
            GEvalConfig,
            GEvalMetric,
            HallucinationConfig,
            HallucinationMetric,
            JsonCorrectnessConfig,
            JsonCorrectnessMetric,
            SummarizationConfig,
            SummarizationMetric,
        )

        assert HallucinationMetric is not None
        assert JsonCorrectnessMetric is not None
        assert SummarizationMetric is not None
        assert GEvalMetric is not None

    def test_metrics_share_base_class(self) -> None:
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.quality import (
            GEvalMetric,
            HallucinationMetric,
            JsonCorrectnessMetric,
            SummarizationMetric,
        )

        assert issubclass(HallucinationMetric, BaseMetric)
        assert issubclass(JsonCorrectnessMetric, BaseMetric)
        assert issubclass(SummarizationMetric, BaseMetric)
        assert issubclass(GEvalMetric, BaseMetric)
