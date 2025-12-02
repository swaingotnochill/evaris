"""Tests for Multimodal evaluation metrics.

This module tests image and multimodal metrics:
- ImageCoherenceMetric: Text-image coherence
- ImageHelpfulnessMetric: Image helpfulness evaluation
- TextToImageMetric: Text-to-image generation quality
- MultimodalAnswerRelevancyMetric: Multimodal RAG relevancy
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.types import TestCase


class TestImageCoherenceMetric:
    """Tests for ImageCoherenceMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.multimodal import ImageCoherenceConfig

        config = ImageCoherenceConfig()
        assert config.threshold == 0.5
        assert config.image_key == "image"

    def test_metric_name(self) -> None:
        from evaris.metrics.multimodal import ImageCoherenceMetric

        metric = ImageCoherenceMetric()
        assert metric.name == "ImageCoherenceMetric"

    def test_validate_inputs_missing_image(self) -> None:
        from evaris.metrics.multimodal import ImageCoherenceMetric

        metric = ImageCoherenceMetric()
        test_case = TestCase(input="Describe this image", expected=None, metadata={})

        with pytest.raises(ValueError, match="image"):
            metric.validate_inputs(test_case, "A cat sitting on a mat")

    @pytest.mark.asyncio
    @patch("evaris.metrics.multimodal.image_coherence.get_provider")
    async def test_measure_high_coherence(self, mock_get_provider: Any) -> None:
        from evaris.metrics.multimodal import ImageCoherenceMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.95, "reasoning": "Text accurately describes the image content"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ImageCoherenceMetric()
        test_case = TestCase(
            input="Describe this image",
            expected=None,
            metadata={"image": "base64_encoded_image_data_here"},
        )

        result = await metric.a_measure(test_case, "A orange tabby cat sitting on a woven mat")

        assert result.score == 0.95
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.multimodal.image_coherence.get_provider")
    async def test_measure_low_coherence(self, mock_get_provider: Any) -> None:
        from evaris.metrics.multimodal import ImageCoherenceMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.2, "reasoning": "Text describes a dog but image shows a cat"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ImageCoherenceMetric()
        test_case = TestCase(
            input="Describe this image",
            expected=None,
            metadata={"image": "base64_encoded_image_data_here"},
        )

        result = await metric.a_measure(test_case, "A large brown dog running")

        assert result.score == 0.2
        assert result.passed is False


class TestImageHelpfulnessMetric:
    """Tests for ImageHelpfulnessMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.multimodal import ImageHelpfulnessConfig

        config = ImageHelpfulnessConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.multimodal import ImageHelpfulnessMetric

        metric = ImageHelpfulnessMetric()
        assert metric.name == "ImageHelpfulnessMetric"

    @pytest.mark.asyncio
    @patch("evaris.metrics.multimodal.image_helpfulness.get_provider")
    async def test_measure_helpful_image(self, mock_get_provider: Any) -> None:
        from evaris.metrics.multimodal import ImageHelpfulnessMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.9, "reasoning": "Image clearly illustrates the concept"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ImageHelpfulnessMetric()
        test_case = TestCase(
            input="Show me how to tie a knot",
            expected=None,
            metadata={"image": "base64_knot_diagram"},
        )

        result = await metric.a_measure(test_case, "Here is a diagram showing the steps")

        assert result.score == 0.9
        assert result.passed is True


class TestTextToImageMetric:
    """Tests for TextToImageMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.multimodal import TextToImageConfig

        config = TextToImageConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.multimodal import TextToImageMetric

        metric = TextToImageMetric()
        assert metric.name == "TextToImageMetric"

    @pytest.mark.asyncio
    @patch("evaris.metrics.multimodal.text_to_image.get_provider")
    async def test_measure_good_generation(self, mock_get_provider: Any) -> None:
        from evaris.metrics.multimodal import TextToImageMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.85, "prompt_adherence": 0.9, "quality": 0.8, "reasoning": "Image matches prompt well"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = TextToImageMetric()
        test_case = TestCase(
            input="Generate an image of a sunset over mountains",
            expected=None,
            metadata={"generated_image": "base64_sunset_image"},
        )

        result = await metric.a_measure(test_case, "Generated image")

        assert result.score == 0.85
        assert result.passed is True


class TestMultimodalAnswerRelevancyMetric:
    """Tests for MultimodalAnswerRelevancyMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.multimodal import MultimodalAnswerRelevancyConfig

        config = MultimodalAnswerRelevancyConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.multimodal import MultimodalAnswerRelevancyMetric

        metric = MultimodalAnswerRelevancyMetric()
        assert metric.name == "MultimodalAnswerRelevancyMetric"

    @pytest.mark.asyncio
    @patch("evaris.metrics.multimodal.multimodal_answer_relevancy.get_provider")
    async def test_measure_relevant_answer(self, mock_get_provider: Any) -> None:
        from evaris.metrics.multimodal import MultimodalAnswerRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.92, "reasoning": "Answer correctly identifies and describes the image content"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = MultimodalAnswerRelevancyMetric()
        test_case = TestCase(
            input="What breed is this dog?",
            expected=None,
            metadata={"image": "base64_golden_retriever_image"},
        )

        result = await metric.a_measure(test_case, "This is a Golden Retriever")

        assert result.score == 0.92
        assert result.passed is True


class TestMultimodalMetricsIntegration:
    """Integration tests for Multimodal metrics."""

    def test_all_multimodal_metrics_importable(self) -> None:
        from evaris.metrics.multimodal import (
            ImageCoherenceConfig,
            ImageCoherenceMetric,
            ImageHelpfulnessConfig,
            ImageHelpfulnessMetric,
            MultimodalAnswerRelevancyConfig,
            MultimodalAnswerRelevancyMetric,
            MultimodalContextualPrecisionConfig,
            MultimodalContextualPrecisionMetric,
            MultimodalContextualRecallConfig,
            MultimodalContextualRecallMetric,
            MultimodalContextualRelevancyConfig,
            MultimodalContextualRelevancyMetric,
            MultimodalFaithfulnessConfig,
            MultimodalFaithfulnessMetric,
            TextToImageConfig,
            TextToImageMetric,
        )

        assert ImageCoherenceMetric is not None
        assert ImageHelpfulnessMetric is not None
        assert TextToImageMetric is not None
        assert MultimodalAnswerRelevancyMetric is not None

    def test_metrics_share_base_class(self) -> None:
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.multimodal import (
            ImageCoherenceMetric,
            ImageHelpfulnessMetric,
            MultimodalAnswerRelevancyMetric,
            TextToImageMetric,
        )

        assert issubclass(ImageCoherenceMetric, BaseMetric)
        assert issubclass(ImageHelpfulnessMetric, BaseMetric)
        assert issubclass(TextToImageMetric, BaseMetric)
        assert issubclass(MultimodalAnswerRelevancyMetric, BaseMetric)
