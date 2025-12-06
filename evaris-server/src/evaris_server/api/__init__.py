"""API routes for evaris-server."""

from evaris_server.api.routes import router
from evaris_server.api.schemas import (
    EvalStatus,
    EvalSummary,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    LogLevel,
    LogRequest,
    LogResponse,
    MetricScore,
    SpanInput,
    TestCaseInput,
    TestResultOutput,
    TraceRequest,
    TraceResponse,
)

__all__ = [
    "router",
    # Schemas
    "EvalStatus",
    "EvalSummary",
    "EvaluateRequest",
    "EvaluateResponse",
    "HealthResponse",
    "LogLevel",
    "LogRequest",
    "LogResponse",
    "MetricScore",
    "SpanInput",
    "TestCaseInput",
    "TestResultOutput",
    "TraceRequest",
    "TraceResponse",
]
