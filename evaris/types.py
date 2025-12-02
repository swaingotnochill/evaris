"""Core type definitions for Evaris evaluation framework.

This module re-exports types from evaris.core for backward compatibility.
New code should import directly from evaris.core.

Deprecated:
    Import from evaris.core.types instead:
    >>> from evaris.core.types import Golden, TestCase, MetricResult
"""

# Re-export all types from core for backward compatibility
from evaris.core.types import (
    AgentFunction,
    AsyncAgentFunction,
    DatasetInput,
    EvalResult,
    Golden,
    MetricResult,
    MultiModalInput,
    MultiModalOutput,
    ReasoningStep,
    TestCase,
    TestResult,
)

# Re-export BaseMetric from protocols for backward compatibility
from evaris.core.protocols import BaseMetric

__all__ = [
    "Golden",
    "TestCase",
    "ReasoningStep",
    "MetricResult",
    "TestResult",
    "EvalResult",
    "BaseMetric",
    "MultiModalInput",
    "MultiModalOutput",
    "AgentFunction",
    "AsyncAgentFunction",
    "DatasetInput",
]
