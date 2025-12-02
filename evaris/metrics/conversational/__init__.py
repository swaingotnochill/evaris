"""Conversational evaluation metrics.

This module provides metrics for evaluating multi-turn conversations:
- KnowledgeRetentionMetric: Checks if LLM retains info from earlier turns
- RoleAdherenceMetric: Checks if LLM maintains assigned role
- ConversationCompletenessMetric: Checks if all topics are resolved
- ConversationRelevancyMetric: Checks if responses are contextually relevant
"""

from evaris.metrics.conversational.conversation_completeness import (
    ConversationCompletenessConfig,
    ConversationCompletenessMetric,
)
from evaris.metrics.conversational.conversation_relevancy import (
    ConversationRelevancyConfig,
    ConversationRelevancyMetric,
)
from evaris.metrics.conversational.knowledge_retention import (
    KnowledgeRetentionConfig,
    KnowledgeRetentionMetric,
)
from evaris.metrics.conversational.role_adherence import (
    RoleAdherenceConfig,
    RoleAdherenceMetric,
)

__all__ = [
    "KnowledgeRetentionMetric",
    "KnowledgeRetentionConfig",
    "RoleAdherenceMetric",
    "RoleAdherenceConfig",
    "ConversationCompletenessMetric",
    "ConversationCompletenessConfig",
    "ConversationRelevancyMetric",
    "ConversationRelevancyConfig",
]
