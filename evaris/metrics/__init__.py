"""Built-in metrics for evaluation.

This module provides a comprehensive set of evaluation metrics:

Core Metrics:
- ExactMatchMetric: Exact string matching
- SemanticSimilarityMetric: Embedding-based similarity
- LLMJudgeMetric: Custom LLM-based evaluation
- FaithfulnessMetric: Factual consistency checking
- ContextRelevanceMetric: Context relevance scoring

RAG Metrics (evaris.metrics.rag):
- AnswerRelevancyMetric: Response relevance to query
- ContextPrecisionMetric: Retrieval precision
- ContextRecallMetric: Retrieval recall
- ContextualRelevancyMetric: Context-to-query relevance
- RAGASMetric: Composite RAG evaluation

Safety Metrics (evaris.metrics.safety):
- BiasMetric: Bias detection
- ToxicityMetric: Toxicity detection
- PIILeakageMetric: PII leak detection
- NonAdviceMetric: Professional advice detection
- MisuseMetric: Harmful use detection
- RoleViolationMetric: System prompt compliance

Agentic Metrics (evaris.metrics.agentic):
- ToolCorrectnessMetric: Tool usage accuracy
- TaskCompletionMetric: Task fulfillment

Quality Metrics (evaris.metrics.quality):
- HallucinationMetric: Hallucination detection
- JsonCorrectnessMetric: JSON validation
- SummarizationMetric: Summary quality
- GEvalMetric: Custom criteria evaluation

NLP Metrics (evaris.metrics.nlp):
- BLEUMetric: BLEU score for translation
- ROUGEMetric: ROUGE score for summarization
- METEORMetric: METEOR score with synonyms/stemming

Conversational Metrics (evaris.metrics.conversational):
- KnowledgeRetentionMetric: Memory retention
- RoleAdherenceMetric: Role/persona consistency
- ConversationCompletenessMetric: Topic resolution
- ConversationRelevancyMetric: Contextual relevance

Multimodal Metrics (evaris.metrics.multimodal):
- ImageCoherenceMetric: Text-image coherence
- ImageHelpfulnessMetric: Image helpfulness
- TextToImageMetric: Text-to-image generation quality
- MultimodalAnswerRelevancyMetric: Multimodal RAG relevancy
- MultimodalFaithfulnessMetric: Multimodal factual accuracy
- MultimodalContextualRelevancyMetric: Multimodal context relevance
- MultimodalContextualRecallMetric: Multimodal retrieval recall
- MultimodalContextualPrecisionMetric: Multimodal retrieval precision

Advanced Metrics (evaris.metrics.advanced):
- DAGMetric: Deep Acyclic Graph multi-criteria evaluation
- ConversationalGEvalMetric: G-Eval for conversations
- ArenaGEvalMetric: Pairwise comparison evaluation
"""

# Core metrics (backward compatible)
from evaris.metrics.context_relevance import ContextRelevanceConfig, ContextRelevanceMetric
from evaris.metrics.exact_match import ExactMatchMetric
from evaris.metrics.faithfulness import FaithfulnessConfig, FaithfulnessMetric
from evaris.metrics.llm_judge import LLMJudgeConfig, LLMJudgeMetric
from evaris.metrics.semantic_similarity import (
    SemanticSimilarityConfig,
    SemanticSimilarityMetric,
)

# RAG metrics
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

# Safety metrics
from evaris.metrics.safety import (
    BiasConfig,
    BiasMetric,
    MisuseConfig,
    MisuseMetric,
    NonAdviceConfig,
    NonAdviceMetric,
    PIILeakageConfig,
    PIILeakageMetric,
    RoleViolationConfig,
    RoleViolationMetric,
    ToxicityConfig,
    ToxicityMetric,
)

# Agentic metrics
from evaris.metrics.agentic import (
    TaskCompletionConfig,
    TaskCompletionMetric,
    ToolCorrectnessConfig,
    ToolCorrectnessMetric,
)

# Quality metrics
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

# NLP metrics
from evaris.metrics.nlp import (
    BLEUConfig,
    BLEUMetric,
    METEORConfig,
    METEORMetric,
    ROUGEConfig,
    ROUGEMetric,
)

# Conversational metrics
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

# Multimodal metrics
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

# Advanced metrics
from evaris.metrics.advanced import (
    ArenaGEvalConfig,
    ArenaGEvalMetric,
    ConversationalGEvalConfig,
    ConversationalGEvalMetric,
    DAGConfig,
    DAGMetric,
)

__all__ = [
    # Core metrics
    "ExactMatchMetric",
    "SemanticSimilarityMetric",
    "SemanticSimilarityConfig",
    "LLMJudgeMetric",
    "LLMJudgeConfig",
    "FaithfulnessMetric",
    "FaithfulnessConfig",
    "ContextRelevanceMetric",
    "ContextRelevanceConfig",
    # RAG metrics
    "AnswerRelevancyMetric",
    "AnswerRelevancyConfig",
    "ContextPrecisionMetric",
    "ContextPrecisionConfig",
    "ContextRecallMetric",
    "ContextRecallConfig",
    "ContextualRelevancyMetric",
    "ContextualRelevancyConfig",
    "RAGASMetric",
    "RAGASConfig",
    # Safety metrics
    "BiasMetric",
    "BiasConfig",
    "ToxicityMetric",
    "ToxicityConfig",
    "PIILeakageMetric",
    "PIILeakageConfig",
    "NonAdviceMetric",
    "NonAdviceConfig",
    "MisuseMetric",
    "MisuseConfig",
    "RoleViolationMetric",
    "RoleViolationConfig",
    # Agentic metrics
    "ToolCorrectnessMetric",
    "ToolCorrectnessConfig",
    "TaskCompletionMetric",
    "TaskCompletionConfig",
    # Quality metrics
    "HallucinationMetric",
    "HallucinationConfig",
    "JsonCorrectnessMetric",
    "JsonCorrectnessConfig",
    "SummarizationMetric",
    "SummarizationConfig",
    "GEvalMetric",
    "GEvalConfig",
    # NLP metrics
    "BLEUMetric",
    "BLEUConfig",
    "ROUGEMetric",
    "ROUGEConfig",
    "METEORMetric",
    "METEORConfig",
    # Conversational metrics
    "KnowledgeRetentionMetric",
    "KnowledgeRetentionConfig",
    "RoleAdherenceMetric",
    "RoleAdherenceConfig",
    "ConversationCompletenessMetric",
    "ConversationCompletenessConfig",
    "ConversationRelevancyMetric",
    "ConversationRelevancyConfig",
    # Multimodal metrics
    "ImageCoherenceMetric",
    "ImageCoherenceConfig",
    "ImageHelpfulnessMetric",
    "ImageHelpfulnessConfig",
    "TextToImageMetric",
    "TextToImageConfig",
    "MultimodalAnswerRelevancyMetric",
    "MultimodalAnswerRelevancyConfig",
    "MultimodalFaithfulnessMetric",
    "MultimodalFaithfulnessConfig",
    "MultimodalContextualRelevancyMetric",
    "MultimodalContextualRelevancyConfig",
    "MultimodalContextualRecallMetric",
    "MultimodalContextualRecallConfig",
    "MultimodalContextualPrecisionMetric",
    "MultimodalContextualPrecisionConfig",
    # Advanced metrics
    "DAGMetric",
    "DAGConfig",
    "ConversationalGEvalMetric",
    "ConversationalGEvalConfig",
    "ArenaGEvalMetric",
    "ArenaGEvalConfig",
]
