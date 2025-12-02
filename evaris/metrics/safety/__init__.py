"""Safety evaluation metrics for LLM outputs.

This module provides metrics for evaluating LLM safety:
- BiasMetric: Detects gender, political, racial, geographical bias
- ToxicityMetric: Detects personal attacks, hate, threats
- PIILeakageMetric: Detects exposed personal information
- NonAdviceMetric: Ensures no unauthorized professional advice
- MisuseMetric: Detects potential misuse patterns
- RoleViolationMetric: Detects persona/role violations
"""

from evaris.metrics.safety.bias import BiasConfig, BiasMetric
from evaris.metrics.safety.misuse import MisuseConfig, MisuseMetric
from evaris.metrics.safety.non_advice import NonAdviceConfig, NonAdviceMetric
from evaris.metrics.safety.pii_leakage import PIILeakageConfig, PIILeakageMetric
from evaris.metrics.safety.role_violation import RoleViolationConfig, RoleViolationMetric
from evaris.metrics.safety.toxicity import ToxicityConfig, ToxicityMetric

__all__ = [
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
]
