"""Evaris: AI Agent Evaluation and Observability Framework.

A comprehensive evaluation framework for AI agents with ABC (Agentic Benchmark
Checklist) compliance and profiling capabilities.
"""

from evaris.baselines import BaselineConfig, BaselineManager
from evaris.compliance import (
    ABCComplianceChecker,
    ABCComplianceConfig,
    ABCComplianceReport,
    ABCViolationError,
    check_compliance,
)

# Contamination detection
from evaris.contamination import ContaminationConfig, ContaminationDetector
from evaris.dataset import EvaluationDataset, load_dataset
from evaris.evaluate import evaluate, evaluate_async, evaluate_stream, evaluate_sync

# Metrics
from evaris.metrics.exact_match import ExactMatchMetric

# Oracle validation
from evaris.oracle import OracleValidationConfig, OracleValidator

# Statistics and analysis
from evaris.statistics import StatisticalAnalyzer, StatisticalConfig
from evaris.types import (
    BaseMetric,
    EvalResult,
    Golden,
    MetricResult,
    ReasoningStep,
    TestCase,
    TestResult,
)

# Validation
from evaris.validation import TestCaseValidator, ValidationConfig

# Cloud Client
from evaris.client import (
    EvarisClient,
    AssessmentResult,
    AssessmentSummary,
    Span,
    TraceResult,
    LogResult,
    get_client,
)

# Retry Configuration
from evaris.retry import (
    RetryConfig,
    RetryExhaustedError,
    default_retry_config,
    aggressive_retry_config,
    no_retry_config,
)

__version__ = "0.0.1-dev-001"

__all__ = [
    # Core API
    "evaluate",
    "evaluate_async",
    "evaluate_sync",
    "evaluate_stream",
    "load_dataset",
    "EvaluationDataset",
    # Types
    "Golden",
    "TestCase",
    "EvalResult",
    "MetricResult",
    "TestResult",
    "BaseMetric",
    "ReasoningStep",
    # Built-in Metrics
    "ExactMatchMetric",
    # Statistics
    "StatisticalAnalyzer",
    "StatisticalConfig",
    # Baselines
    "BaselineManager",
    "BaselineConfig",
    # Validation
    "TestCaseValidator",
    "ValidationConfig",
    # Contamination
    "ContaminationDetector",
    "ContaminationConfig",
    # Oracle
    "OracleValidator",
    "OracleValidationConfig",
    # ABC Compliance
    "ABCComplianceChecker",
    "ABCComplianceConfig",
    "ABCComplianceReport",
    "ABCViolationError",
    "check_compliance",
    # Cloud Client
    "EvarisClient",
    "AssessmentResult",
    "AssessmentSummary",
    "Span",
    "TraceResult",
    "LogResult",
    "get_client",
    # Retry Configuration
    "RetryConfig",
    "RetryExhaustedError",
    "default_retry_config",
    "aggressive_retry_config",
    "no_retry_config",
]
