"""API request/response schemas for evaris-server."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ==============================================================================
# Evaluate Schemas
# ==============================================================================


class EvalStatus(str, Enum):
    """Status of an evaluation run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"


class TestCaseInput(BaseModel):
    """A single test case to evaluate."""

    input: Any = Field(..., description="Input to the agent/model")
    expected: Any | None = Field(None, description="Expected output (optional)")
    actual_output: Any = Field(..., description="Actual output from agent/model")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class EvaluateRequest(BaseModel):
    """Request to run an evaluation."""

    name: str = Field(..., description="Name for this evaluation run")
    test_cases: list[TestCaseInput] = Field(
        ..., description="Test cases to evaluate"
    )
    metrics: list[str] = Field(
        ...,
        description="Metric names to run (e.g., 'faithfulness', 'toxicity')",
    )
    dataset_id: str | None = Field(
        None, description="Optional dataset ID for tracking"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the eval run",
    )


class MetricScore(BaseModel):
    """Score for a single metric on a single test case."""

    name: str
    score: float
    passed: bool
    threshold: float = 0.5
    reasoning: str | None = None
    reasoning_steps: list[dict[str, Any]] | None = None
    reasoning_type: str | None = None  # "logic", "llm", "hybrid"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestResultOutput(BaseModel):
    """Result for a single test case."""

    id: str | None = None
    input: Any
    expected: Any | None
    actual_output: Any
    scores: list[MetricScore]
    passed: bool
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    """Summary statistics for a single metric."""

    accuracy: float = Field(..., description="Percentage of tests that passed")
    avg_score: float = Field(..., description="Average score across all tests")
    min_score: float | None = None
    max_score: float | None = None
    std_dev: float | None = None


class EvalSummary(BaseModel):
    """Summary statistics for an evaluation run."""

    total: int
    passed: int
    failed: int
    accuracy: float
    avg_latency_ms: float | None = None
    metrics: dict[str, MetricSummary] = Field(
        default_factory=dict,
        description="Per-metric statistics",
    )


class EvaluateResponse(BaseModel):
    """Response from an evaluation run."""

    eval_id: str
    organization_id: str
    project_id: str
    name: str
    status: EvalStatus
    summary: EvalSummary | None = None
    results: list[TestResultOutput] | None = None
    created_at: datetime
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# List/Get Evaluation Schemas
# ==============================================================================


class EvalListItem(BaseModel):
    """Evaluation item for list responses (without full results)."""

    eval_id: str
    organization_id: str
    project_id: str
    name: str
    status: EvalStatus
    total: int | None = None
    passed: int | None = None
    failed: int | None = None
    accuracy: float | None = None
    created_at: datetime
    completed_at: datetime | None = None


class EvalListResponse(BaseModel):
    """Response for listing evaluations."""

    evaluations: list[EvalListItem]
    total: int
    limit: int
    offset: int


# ==============================================================================
# Trace Schemas
# ==============================================================================


class SpanInput(BaseModel):
    """A single span within a trace."""

    name: str = Field(..., description="Span name")
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = None
    input: Any | None = None
    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list["SpanInput"] = Field(default_factory=list)


class TraceRequest(BaseModel):
    """Request to store a trace."""

    name: str = Field(..., description="Trace name")
    spans: list[SpanInput] = Field(..., description="Spans in this trace")
    duration_ms: float | None = None
    eval_id: str | None = Field(None, description="Associated evaluation ID")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    """Response after storing a trace."""

    trace_id: str
    organization_id: str
    project_id: str
    name: str
    span_count: int
    duration_ms: float | None
    created_at: datetime


# ==============================================================================
# Log Schemas
# ==============================================================================


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogRequest(BaseModel):
    """Request to store a log entry."""

    level: LogLevel = Field(default=LogLevel.INFO)
    message: str = Field(..., description="Log message")
    source: str = Field(default="evaris-sdk", description="Log source identifier")
    agent_id: str | None = Field(None, description="Agent identifier")
    eval_id: str | None = Field(None, description="Associated evaluation ID")
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = Field(
        None, description="Log timestamp (defaults to now)"
    )


class LogResponse(BaseModel):
    """Response after storing a log entry."""

    log_id: str
    organization_id: str
    project_id: str
    level: LogLevel
    created_at: datetime


# ==============================================================================
# Health Check
# ==============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    database: str = "connected"
