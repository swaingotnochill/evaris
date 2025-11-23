"""Built-in metrics for evaluation."""

from evaris.metrics.context_relevance import ContextRelevanceConfig, ContextRelevanceMetric
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.metrics.faithfulness import FaithfulnessConfig, FaithfulnessMetric
from evaris.metrics.llm_judge import LLMJudgeConfig, LLMJudgeMetric
from evaris.metrics.semantic_similarity import (
    SemanticSimilarityConfig,
    SemanticSimilarityMetric,
)

__all__ = [
    "ExactMatchMetric",
    "SemanticSimilarityMetric",
    "SemanticSimilarityConfig",
    "LLMJudgeMetric",
    "LLMJudgeConfig",
    "FaithfulnessMetric",
    "FaithfulnessConfig",
    "ContextRelevanceMetric",
    "ContextRelevanceConfig",
]
