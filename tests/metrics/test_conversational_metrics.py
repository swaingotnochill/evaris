"""Tests for Conversational evaluation metrics.

This module tests:
- KnowledgeRetentionMetric: Checks if LLM retains info from earlier turns
- RoleAdherenceMetric: Checks if LLM maintains assigned role
- ConversationCompletenessMetric: Checks if all topics are resolved
- ConversationRelevancyMetric: Checks if responses are contextually relevant
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaris.types import TestCase


class TestKnowledgeRetentionMetric:
    """Tests for KnowledgeRetentionMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.conversational import KnowledgeRetentionConfig

        config = KnowledgeRetentionConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.conversational import KnowledgeRetentionMetric

        metric = KnowledgeRetentionMetric()
        assert metric.name == "KnowledgeRetentionMetric"

    def test_validate_inputs_missing_messages(self) -> None:
        from evaris.metrics.conversational import KnowledgeRetentionMetric

        metric = KnowledgeRetentionMetric()
        test_case = TestCase(input="Hello", expected=None, metadata={})

        with pytest.raises(ValueError, match="messages"):
            metric.validate_inputs(test_case, "Some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.knowledge_retention.get_provider")
    async def test_measure_good_retention(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import KnowledgeRetentionMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"knowledge_points": [{"info": "user name is John", "retained": "yes"}], "retained_count": 1, "total_count": 1}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = KnowledgeRetentionMetric()
        test_case = TestCase(
            input="What is my name?",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "My name is John"},
                    {"role": "assistant", "content": "Nice to meet you, John!"},
                ]
            },
        )

        result = await metric.a_measure(test_case, "Your name is John.")

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.knowledge_retention.get_provider")
    async def test_measure_poor_retention(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import KnowledgeRetentionMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"knowledge_points": [{"info": "user name is John", "retained": "no"}], "retained_count": 0, "total_count": 1}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = KnowledgeRetentionMetric()
        test_case = TestCase(
            input="What is my name?",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "My name is John"},
                    {"role": "assistant", "content": "Nice to meet you!"},
                ]
            },
        )

        result = await metric.a_measure(test_case, "I don't know your name.")

        assert result.score == 0.0
        assert result.passed is False


class TestRoleAdherenceMetric:
    """Tests for RoleAdherenceMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.conversational import RoleAdherenceConfig

        config = RoleAdherenceConfig()
        assert config.threshold == 0.5
        assert config.role_key == "system_role"

    def test_metric_name(self) -> None:
        from evaris.metrics.conversational import RoleAdherenceMetric

        metric = RoleAdherenceMetric()
        assert metric.name == "RoleAdherenceMetric"

    def test_validate_inputs_missing_role(self) -> None:
        from evaris.metrics.conversational import RoleAdherenceMetric

        metric = RoleAdherenceMetric()
        test_case = TestCase(input="Hello", expected=None, metadata={})

        with pytest.raises(ValueError, match="system_role"):
            metric.validate_inputs(test_case, "Some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.role_adherence.get_provider")
    async def test_measure_good_adherence(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import RoleAdherenceMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.95, "violations": [], "reasoning": "Response maintains professional tone throughout"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = RoleAdherenceMetric()
        test_case = TestCase(
            input="Tell me a joke",
            expected=None,
            metadata={
                "system_role": "You are a professional customer service agent. Be polite and helpful."
            },
        )

        result = await metric.a_measure(
            test_case, "I'd be happy to help! Here's a light joke for you..."
        )

        assert result.score == 0.95
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.role_adherence.get_provider")
    async def test_measure_role_violation(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import RoleAdherenceMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.2, "violations": ["used inappropriate language", "broke character"], "reasoning": "Response deviated from professional role"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = RoleAdherenceMetric()
        test_case = TestCase(
            input="Tell me a joke",
            expected=None,
            metadata={
                "system_role": "You are a professional customer service agent. Be polite and helpful."
            },
        )

        result = await metric.a_measure(test_case, "LOL here's a rude joke...")

        assert result.score == 0.2
        assert result.passed is False


class TestConversationCompletenessMetric:
    """Tests for ConversationCompletenessMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.conversational import ConversationCompletenessConfig

        config = ConversationCompletenessConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.conversational import ConversationCompletenessMetric

        metric = ConversationCompletenessMetric()
        assert metric.name == "ConversationCompletenessMetric"

    def test_validate_inputs_missing_messages(self) -> None:
        from evaris.metrics.conversational import ConversationCompletenessMetric

        metric = ConversationCompletenessMetric()
        test_case = TestCase(input="Hello", expected=None, metadata={})

        with pytest.raises(ValueError, match="messages"):
            metric.validate_inputs(test_case, "Some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.conversation_completeness.get_provider")
    async def test_measure_complete_conversation(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import ConversationCompletenessMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"topics": [{"topic": "account balance", "resolved": "yes"}], "resolved_count": 1, "total_count": 1, "completeness_score": 1.0}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ConversationCompletenessMetric()
        test_case = TestCase(
            input="Thanks, that answers my question",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "What's my account balance?"},
                    {"role": "assistant", "content": "Your account balance is $1,234.56"},
                ]
            },
        )

        result = await metric.a_measure(
            test_case, "You're welcome! Let me know if you need anything else."
        )

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.conversation_completeness.get_provider")
    async def test_measure_incomplete_conversation(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import ConversationCompletenessMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"topics": [{"topic": "account balance", "resolved": "no"}, {"topic": "transfer funds", "resolved": "no"}], "resolved_count": 0, "total_count": 2, "completeness_score": 0.0}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ConversationCompletenessMetric()
        test_case = TestCase(
            input="I need to check my balance and transfer money",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "I need to check my balance and transfer money"},
                    {"role": "assistant", "content": "Let me help you with that."},
                ]
            },
        )

        result = await metric.a_measure(test_case, "Is there anything else?")

        assert result.score == 0.0
        assert result.passed is False


class TestConversationRelevancyMetric:
    """Tests for ConversationRelevancyMetric."""

    def test_config_defaults(self) -> None:
        from evaris.metrics.conversational import ConversationRelevancyConfig

        config = ConversationRelevancyConfig()
        assert config.threshold == 0.5

    def test_metric_name(self) -> None:
        from evaris.metrics.conversational import ConversationRelevancyMetric

        metric = ConversationRelevancyMetric()
        assert metric.name == "ConversationRelevancyMetric"

    def test_validate_inputs_missing_messages(self) -> None:
        from evaris.metrics.conversational import ConversationRelevancyMetric

        metric = ConversationRelevancyMetric()
        test_case = TestCase(input="Hello", expected=None, metadata={})

        with pytest.raises(ValueError, match="messages"):
            metric.validate_inputs(test_case, "Some output")

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.conversation_relevancy.get_provider")
    async def test_measure_relevant_response(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import ConversationRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.95, "reasoning": "Response directly addresses the question about weather in context of travel plans"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ConversationRelevancyMetric()
        test_case = TestCase(
            input="Will it rain tomorrow?",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "I'm planning a picnic tomorrow"},
                    {
                        "role": "assistant",
                        "content": "That sounds lovely! Where are you planning to go?",
                    },
                ]
            },
        )

        result = await metric.a_measure(
            test_case,
            "Based on the forecast, it should be sunny tomorrow - perfect for your picnic!",
        )

        assert result.score == 0.95
        assert result.passed is True

    @pytest.mark.asyncio
    @patch("evaris.metrics.conversational.conversation_relevancy.get_provider")
    async def test_measure_irrelevant_response(self, mock_get_provider: Any) -> None:
        from evaris.metrics.conversational import ConversationRelevancyMetric

        mock_provider = MagicMock()
        mock_provider.a_complete = AsyncMock(
            return_value=MagicMock(
                content='{"score": 0.1, "reasoning": "Response about cooking is unrelated to the weather question"}'
            )
        )
        mock_get_provider.return_value = mock_provider

        metric = ConversationRelevancyMetric()
        test_case = TestCase(
            input="Will it rain tomorrow?",
            expected=None,
            metadata={
                "messages": [
                    {"role": "user", "content": "I'm planning a picnic tomorrow"},
                    {"role": "assistant", "content": "That sounds lovely!"},
                ]
            },
        )

        result = await metric.a_measure(test_case, "Here's a great recipe for pasta carbonara.")

        assert result.score == 0.1
        assert result.passed is False


class TestConversationalMetricsIntegration:
    """Integration tests for Conversational metrics."""

    def test_all_conversational_metrics_importable(self) -> None:
        from evaris.metrics.conversational import (
            ConversationCompletenessConfig,
            ConversationCompletenessMetric,
            ConversationRelevancyConfig,
            ConversationRelevancyMetric,
            KnowledgeRetentionConfig,
            KnowledgeRetentionMetric,
            RoleAdherenceConfig,
            RoleAdherenceMetric,
        )

        assert KnowledgeRetentionMetric is not None
        assert RoleAdherenceMetric is not None
        assert ConversationCompletenessMetric is not None
        assert ConversationRelevancyMetric is not None

    def test_metrics_share_base_class(self) -> None:
        from evaris.core.protocols import BaseMetric
        from evaris.metrics.conversational import (
            ConversationCompletenessMetric,
            ConversationRelevancyMetric,
            KnowledgeRetentionMetric,
            RoleAdherenceMetric,
        )

        assert issubclass(KnowledgeRetentionMetric, BaseMetric)
        assert issubclass(RoleAdherenceMetric, BaseMetric)
        assert issubclass(ConversationCompletenessMetric, BaseMetric)
        assert issubclass(ConversationRelevancyMetric, BaseMetric)
