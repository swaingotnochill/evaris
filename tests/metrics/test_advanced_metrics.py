"""Tests for Advanced framework evaluation metrics.

This module tests:
- DAGMetric: Deep Acyclic Graph evaluation
- ConversationalGEvalMetric: G-Eval for conversations
- ArenaGEvalMetric: Pairwise comparison evaluation
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.types import TestCase


class TestDAGMetric:
    """Tests for DAGMetric (Deep Acyclic Graph evaluation)."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.advanced import DAGConfig

        config = DAGConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.advanced import DAGMetric

        metric = DAGMetric()
        assert metric.name == "DAGMetric"

    @pytest.mark.asyncio
    @patch("evaris.metrics.advanced.dag.get_provider")
    async def test_measure_with_criteria(self, mock_get_provider: Any) -> None:
        from evaris.metrics.advanced import DAGConfig, DAGMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.85, "node_scores": {"clarity": 0.9, "completeness": 0.8}, "reasoning": "Good overall"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = DAGMetric(config=DAGConfig(evaluation_nodes=["clarity", "completeness"]))
        test_case = TestCase(input="Explain Python", expected=None)

        result = await metric.a_measure(test_case, "Python is a programming language...")

        assert result.score == 0.85
        assert result.passed is True
        assert "node_scores" in result.metadata


class TestConversationalGEvalMetric:
    """Tests for ConversationalGEvalMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.advanced import ConversationalGEvalConfig

        config = ConversationalGEvalConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.advanced import ConversationalGEvalMetric

        metric = ConversationalGEvalMetric()
        assert metric.name == "ConversationalGEvalMetric"

    def test_validate_inputs_missing_messages(self) -> None:
        from evaris.metrics.advanced import ConversationalGEvalMetric

        metric = ConversationalGEvalMetric()
        test_case = TestCase(input="Hello", expected=None, metadata={})

        with pytest.raises(ValueError, match="messages"):
            metric.validate_inputs(test_case, "Some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.advanced.conversational_g_eval.get_provider")
    async def test_measure_good_conversation(self, mock_get_provider: Any) -> None:
        from evaris.metrics.advanced import ConversationalGEvalConfig, ConversationalGEvalMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.9, "criteria_scores": {"coherence": 0.95, "helpfulness": 0.85}, "reasoning": "Excellent conversation"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ConversationalGEvalMetric(
            config=ConversationalGEvalConfig(criteria="coherence, helpfulness")
        )
        test_case = TestCase(
            input="Thanks for your help!",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "How do I learn Python?"},
                    {"role": "assistant", "content": "Start with basic tutorials..."},
                ]
            },
        )

        result = await metric.a_measure(test_case, "You're welcome!")

        assert result.score == 0.9
        assert result.passed is True


class TestArenaGEvalMetric:
    """Tests for ArenaGEvalMetric (pairwise comparison)."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.advanced import ArenaGEvalConfig

        config = ArenaGEvalConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.advanced import ArenaGEvalMetric

        metric = ArenaGEvalMetric()
        assert metric.name == "ArenaGEvalMetric"

    def test_validate_inputs_missing_comparison(self) -> None:
        from evaris.metrics.advanced import ArenaGEvalMetric

        metric = ArenaGEvalMetric()
        test_case = TestCase(input="Query", expected=None, metadata={})

        with pytest.raises(ValueError, match="comparison_output"):
            metric.validate_inputs(test_case, "Response A")

    @pytest.mark.asyncio
    @patch("evaris.metrics.advanced.arena_g_eval.get_provider")
    async def test_measure_comparison(self, mock_get_provider: Any) -> None:
        from evaris.metrics.advanced import ArenaGEvalMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"winner": "A", "score": 0.75, "reasoning": "Response A is more detailed"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ArenaGEvalMetric()
        test_case = TestCase(
            input="Explain machine learning",
            expected=None,
            metadata={"comparison_output": "A brief ML explanation"},
        )

        result = await metric.a_measure(test_case, "A detailed ML explanation with examples")

        assert result.score == 0.75
        assert result.passed is True
        assert result.metadata.get("winner") == "A"


class TestAdvancedMetricsIntegration:
    """Integration tests for Advanced metrics."""

    def test_all_advanced_metrics_importable(self) -> None:
        from evaris.metrics.advanced import (
            ArenaGEvalConfig,
            ArenaGEvalMetric,
            ConversationalGEvalConfig,
            ConversationalGEvalMetric,
            DAGConfig,
            DAGMetric,
        )

        assert DAGMetric is not None
        assert ConversationalGEvalMetric is not None
        assert ArenaGEvalMetric is not None

    def test_metrics_share_base_class(self) -> None:
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.advanced import (
            ArenaGEvalMetric,
            ConversationalGEvalMetric,
            DAGMetric,
        )

        assert issubclass(DAGMetric, BaseMetric)
        assert issubclass(ConversationalGEvalMetric, BaseMetric)
        assert issubclass(ArenaGEvalMetric, BaseMetric)
