"""Core module for Evaris evaluation framework.

This module contains the fundamental types, protocols, and registries
that form the foundation of the Evaris platform.

Modules:
    types: Core data types (Golden, TestCase, MetricResult, etc.)
    protocols: Abstract interfaces (Metric, Tool, Agent)
    registry: Global registries for metrics and tools
"""

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
from evaris.core.protocols import (
    BaseMetric,
    BaseTool,
    ToolResult,
)
from evaris.core.registry import (
    MetricRegistry,
    ToolRegistry,
    get_metric_registry,
    get_tool_registry,
    register_metric,
    register_tool,
)

__all__ = [
    # Types
    "Golden",
    "TestCase",
    "ReasoningStep",
    "MetricResult",
    "TestResult",
    "EvalResult",
    "MultiModalInput",
    "MultiModalOutput",
    "AgentFunction",
    "AsyncAgentFunction",
    "DatasetInput",
    # Protocols
    "BaseMetric",
    "BaseTool",
    "ToolResult",
    # Registry
    "MetricRegistry",
    "ToolRegistry",
    "get_metric_registry",
    "get_tool_registry",
    "register_metric",
    "register_tool",
]
