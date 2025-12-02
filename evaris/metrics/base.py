"""Base metric class and utilities for Evaris metrics.

This module provides the foundation for all evaluation metrics:
- BaseMetric: Abstract base class with common functionality
- Metric utilities: Validation, scoring helpers

All metrics should inherit from BaseMetric and implement a_measure().
"""

from evaris.core.protocols import BaseMetric
from evaris.core.registry import (
    MetricRegistry,
    get_metric_registry,
    register_metric,
    resolve_metric,
)
from evaris.core.types import MetricResult, ReasoningStep, TestCase

__all__ = [
    "BaseMetric",
    "MetricResult",
    "ReasoningStep",
    "TestCase",
    "MetricRegistry",
    "get_metric_registry",
    "register_metric",
    "resolve_metric",
]
