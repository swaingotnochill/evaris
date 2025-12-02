"""LLM Judge metric with tool calling support.

This module provides the LLM Judge metric which can evaluate agent outputs
using LLM-based judgment. It supports three modes:

1. Standalone Mode (default): Direct LLM evaluation
   >>> judge = LLMJudge(provider="openrouter")

2. Tool Mode: LLM can call tools to verify outputs
   >>> judge = LLMJudge(
   ...     provider="openrouter",
   ...     mode="tools",
   ...     tools=["code_executor", "web_search"]
   ... )

3. Meta Mode: LLM synthesizes judgment from other metrics' results
   >>> # Pass metric_results in test_case.metadata
   >>> judge = LLMJudge(mode="meta")
   >>> # The judge will synthesize from previous metric evaluations

Example usage:
    >>> from evaris.metrics.llm_judge import LLMJudge, JudgeConfig, JudgeMode
    >>> from evaris.types import TestCase
    >>>
    >>> # Simple standalone evaluation
    >>> judge = LLMJudge()
    >>> test_case = TestCase(input="What is 2+2?", expected="4")
    >>> result = await judge.a_measure(test_case, "The answer is 4")
    >>>
    >>> # With tools
    >>> judge = LLMJudge(mode="tools", tools=["code_executor"])
    >>> result = await judge.a_measure(test_case, actual_output)
"""

# New modular implementation
from evaris.metrics.llm_judge.judge import (
    JudgeConfig,
    JudgeMode,
    LLMJudge,
    LLMJudgeMetric,  # Alias for backward compatibility
)

# Legacy implementation (for users who need the old API)
from evaris.metrics.llm_judge_legacy import (
    LLMJudgeConfig,
    LLMJudgeMetric as LLMJudgeLegacy,
)

__all__ = [
    # New API
    "LLMJudge",
    "JudgeConfig",
    "JudgeMode",
    # Backward compatibility
    "LLMJudgeMetric",
    "LLMJudgeConfig",
    "LLMJudgeLegacy",
]
