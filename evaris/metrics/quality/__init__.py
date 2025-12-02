"""Quality evaluation metrics for LLM outputs.

This module provides metrics for evaluating output quality:
- HallucinationMetric: Detects factual hallucinations
- JsonCorrectnessMetric: Validates JSON output structure
- SummarizationMetric: Evaluates summary quality
- GEvalMetric: Custom criteria evaluation (G-Eval)
"""

from evaris.metrics.quality.g_eval import GEvalConfig, GEvalMetric
from evaris.metrics.quality.hallucination import HallucinationConfig, HallucinationMetric
from evaris.metrics.quality.json_correctness import (
    JsonCorrectnessConfig,
    JsonCorrectnessMetric,
)
from evaris.metrics.quality.summarization import SummarizationConfig, SummarizationMetric

__all__ = [
    "HallucinationMetric",
    "HallucinationConfig",
    "JsonCorrectnessMetric",
    "JsonCorrectnessConfig",
    "SummarizationMetric",
    "SummarizationConfig",
    "GEvalMetric",
    "GEvalConfig",
]
