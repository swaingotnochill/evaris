"""API routes for evaris-server internal endpoints.

These endpoints are called by evaris-web (the API gateway) and require
internal JWT authentication. All requests include organization context for RLS.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from evaris_server.api.schemas import (
    EvalListItem,
    EvalListResponse,
    EvalStatus,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    LogRequest,
    LogResponse,
    TraceRequest,
    TraceResponse,
)
from evaris_server.db import Database, get_database
from evaris_server.middleware.auth import InternalAuthContext, verify_internal_request
from evaris_server.services import RunnerService

router = APIRouter()


# ==============================================================================
# Health Check (public)
# ==============================================================================


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(db: Database = Depends(get_database)) -> HealthResponse:
    """Health check endpoint.

    Used by load balancers and orchestration systems to verify service health.
    Checks database connectivity.
    """
    db_healthy = await db.health_check()

    return HealthResponse(
        status="ok" if db_healthy else "degraded",
        version="0.1.0",
        database="connected" if db_healthy else "disconnected",
    )


# ==============================================================================
# Internal Evaluate Endpoints
# ==============================================================================


@router.post(
    "/internal/evaluate",
    response_model=EvaluateResponse,
    tags=["evaluate"],
    status_code=status.HTTP_201_CREATED,
)
async def run_evaluation(
    request: EvaluateRequest,
    auth: Annotated[InternalAuthContext, Depends(verify_internal_request)],
    db: Database = Depends(get_database),
) -> EvaluateResponse:
    """Run an evaluation with specified metrics.

    This endpoint:
    1. Validates the request and auth context
    2. Runs each metric against each test case
    3. Stores results in the database with RLS context
    4. Returns summary statistics and detailed results

    The organization_id from the JWT determines RLS scope - results are only
    visible to the authenticated organization.

    Args:
        request: Evaluation request with test cases and metrics
        auth: Internal auth context from JWT (includes organization_id, project_id)
        db: Database connection

    Returns:
        EvaluateResponse with results and summary

    Raises:
        HTTPException: If no valid metrics or other errors occur
    """
    runner = RunnerService(db=db)

    try:
        result = await runner.run_assessment(
            request=request,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            user_id=auth.user_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get(
    "/internal/evaluations",
    response_model=EvalListResponse,
    tags=["evaluate"],
)
async def list_evaluations(
    auth: Annotated[InternalAuthContext, Depends(verify_internal_request)],
    db: Database = Depends(get_database),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: EvalStatus | None = Query(default=None, alias="status"),
) -> EvalListResponse:
    """List evaluations for the authenticated organization/project.

    Returns paginated list of evaluations without full results.
    Use GET /internal/evaluations/{eval_id} to get full results.

    Args:
        auth: Internal auth context from JWT
        db: Database connection
        limit: Max number of results (1-100)
        offset: Pagination offset
        status_filter: Optional status filter

    Returns:
        EvalListResponse with paginated evaluations
    """
    async with db.with_org_context(auth.organization_id) as client:
        # Build where clause
        where = {"projectId": auth.project_id}
        if status_filter:
            where["status"] = status_filter.value

        # Get total count
        total = await client.eval.count(where=where)

        # Get evaluations
        evals = await client.eval.find_many(
            where=where,
            order={"createdAt": "desc"},
            skip=offset,
            take=limit,
        )

        evaluations = [
            EvalListItem(
                eval_id=e.id,
                organization_id=e.organizationId,
                project_id=e.projectId,
                name=e.name,
                status=EvalStatus(e.status),
                total=e.total,
                passed=e.passed,
                failed=e.failed,
                accuracy=e.accuracy,
                created_at=e.createdAt,
                completed_at=e.completedAt,
            )
            for e in evals
        ]

        return EvalListResponse(
            evaluations=evaluations,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get(
    "/internal/evaluations/{eval_id}",
    response_model=EvaluateResponse,
    tags=["evaluate"],
)
async def get_evaluation(
    eval_id: str,
    auth: Annotated[InternalAuthContext, Depends(verify_internal_request)],
    db: Database = Depends(get_database),
    include_results: bool = Query(default=True),
) -> EvaluateResponse:
    """Get a single evaluation by ID with full results.

    Args:
        eval_id: The evaluation ID
        auth: Internal auth context from JWT
        db: Database connection
        include_results: Whether to include individual test results

    Returns:
        EvaluateResponse with evaluation details and optionally results

    Raises:
        HTTPException 404: If evaluation not found
    """
    async with db.with_org_context(auth.organization_id) as client:
        # Get evaluation with optional results
        include = {"testResults": include_results} if include_results else None

        evaluation = await client.eval.find_unique(
            where={"id": eval_id},
            include=include,
        )

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {eval_id} not found",
            )

        # Verify project access
        if evaluation.projectId != auth.project_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {eval_id} not found",
            )

        # Build response
        from evaris_server.api.schemas import (
            EvalSummary,
            MetricScore,
            MetricSummary,
            TestResultOutput,
        )

        # Parse summary from JSON
        summary = None
        if evaluation.summary:
            summary_data = (
                evaluation.summary
                if isinstance(evaluation.summary, dict)
                else json.loads(evaluation.summary)
            )
            metrics_summary = {}
            for name, data in summary_data.get("metrics", {}).items():
                metrics_summary[name] = MetricSummary(
                    accuracy=data.get("accuracy", 0),
                    avg_score=data.get("avg_score", data.get("mean", 0)),
                    min_score=data.get("min_score"),
                    max_score=data.get("max_score"),
                )
            summary = EvalSummary(
                total=evaluation.total or 0,
                passed=evaluation.passed or 0,
                failed=evaluation.failed or 0,
                accuracy=evaluation.accuracy or 0,
                avg_latency_ms=summary_data.get("avg_latency_ms"),
                metrics=metrics_summary,
            )

        # Parse test results
        results = None
        if include_results and hasattr(evaluation, "testResults"):
            results = []
            for tr in evaluation.testResults:
                scores_data = (
                    tr.scores
                    if isinstance(tr.scores, list)
                    else json.loads(tr.scores or "[]")
                )
                scores = [
                    MetricScore(
                        name=s.get("name", ""),
                        score=s.get("score", 0),
                        passed=s.get("passed", False),
                        threshold=s.get("threshold", 0.5),
                        reasoning=s.get("reasoning"),
                        metadata=s.get("metadata", {}),
                    )
                    for s in scores_data
                ]
                results.append(
                    TestResultOutput(
                        id=tr.id,
                        input=tr.input,
                        expected=tr.expected,
                        actual_output=tr.actualOutput,
                        scores=scores,
                        passed=tr.passed,
                        latency_ms=tr.latencyMs,
                        error=tr.error,
                        metadata=tr.metadata or {},
                    )
                )

        return EvaluateResponse(
            eval_id=evaluation.id,
            organization_id=evaluation.organizationId,
            project_id=evaluation.projectId,
            name=evaluation.name,
            status=EvalStatus(evaluation.status),
            summary=summary,
            results=results,
            created_at=evaluation.createdAt,
            completed_at=evaluation.completedAt,
            metadata=evaluation.metadata or {},
        )


# ==============================================================================
# Internal Trace Endpoint
# ==============================================================================


@router.post(
    "/internal/trace",
    response_model=TraceResponse,
    tags=["observability"],
    status_code=status.HTTP_201_CREATED,
)
async def store_trace(
    request: TraceRequest,
    auth: Annotated[InternalAuthContext, Depends(verify_internal_request)],
    db: Database = Depends(get_database),
) -> TraceResponse:
    """Store a trace with spans.

    Traces represent a single execution flow (e.g., an agent run) with
    nested spans showing the call hierarchy and timing.

    Args:
        request: Trace data with spans
        auth: Internal auth context from JWT
        db: Database connection

    Returns:
        TraceResponse with trace ID and metadata
    """
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    # Calculate total duration if not provided
    duration_ms = request.duration_ms
    if duration_ms is None and request.spans:
        duration_ms = sum(s.duration_ms or 0 for s in request.spans)

    try:
        async with db.with_org_context(auth.organization_id) as client:
            # Create trace
            trace = await client.trace.create(
                data={
                    "id": trace_id,
                    "traceId": trace_id,
                    "organizationId": auth.organization_id,
                    "rootSpanName": request.spans[0].name if request.spans else request.name,
                    "serviceName": "evaris-sdk",
                    "duration": int(duration_ms or 0),
                    "spanCount": len(request.spans),
                    "startTime": now,
                    "evalId": request.eval_id,
                }
            )

            # Create spans
            span_count = await _create_spans(
                client=client,
                trace_id=trace_id,
                spans=request.spans,
                parent_span_id=None,
            )

        return TraceResponse(
            trace_id=trace_id,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            name=request.name,
            span_count=span_count,
            duration_ms=duration_ms,
            created_at=now,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store trace: {str(e)}",
        )


async def _create_spans(
    client,
    trace_id: str,
    spans: list,
    parent_span_id: str | None,
) -> int:
    """Recursively create spans in the database.

    Args:
        client: Prisma client
        trace_id: Parent trace ID
        spans: List of SpanInput objects
        parent_span_id: Parent span ID (None for root spans)

    Returns:
        Total count of spans created
    """
    count = 0

    for span in spans:
        span_id = f"span_{uuid.uuid4().hex[:12]}"

        await client.span.create(
            data={
                "id": span_id,
                "spanId": span_id,
                "traceId": trace_id,
                "parentSpanId": parent_span_id,
                "operationName": span.name,
                "serviceName": "evaris-sdk",
                "startTime": 0,  # Relative to trace start
                "duration": int(span.duration_ms or 0),
                "attributes": json.dumps(span.metadata) if span.metadata else "{}",
            }
        )
        count += 1

        # Create children recursively
        if span.children:
            count += await _create_spans(
                client=client,
                trace_id=trace_id,
                spans=span.children,
                parent_span_id=span_id,
            )

    return count


# ==============================================================================
# Internal Log Endpoint
# ==============================================================================


@router.post(
    "/internal/log",
    response_model=LogResponse,
    tags=["observability"],
    status_code=status.HTTP_201_CREATED,
)
async def store_log(
    request: LogRequest,
    auth: Annotated[InternalAuthContext, Depends(verify_internal_request)],
    db: Database = Depends(get_database),
) -> LogResponse:
    """Store a log entry.

    Logs are structured entries for debugging and monitoring agent behavior.

    Args:
        request: Log entry data
        auth: Internal auth context from JWT
        db: Database connection

    Returns:
        LogResponse with log ID
    """
    log_id = f"log_{uuid.uuid4().hex[:12]}"
    timestamp = request.timestamp or datetime.now(timezone.utc)

    try:
        async with db.with_org_context(auth.organization_id) as client:
            await client.log.create(
                data={
                    "id": log_id,
                    "organizationId": auth.organization_id,
                    "level": request.level.value,
                    "source": request.source,
                    "agentId": request.agent_id,
                    "message": request.message,
                    "metadata": json.dumps(request.metadata) if request.metadata else "{}",
                    "timestamp": timestamp,
                    "evalId": request.eval_id,
                }
            )

        return LogResponse(
            log_id=log_id,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            level=request.level,
            created_at=timestamp,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store log: {str(e)}",
        )
