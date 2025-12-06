"""Runner service - runs LLM judge and computes metrics.

This service integrates with the main evaris package to run assessments
using the existing metrics engine. Results are stored in PostgreSQL with
Row Level Security enforced at the organization level.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from evaris_server.api.schemas import (
    EvalStatus,
    EvalSummary,
    EvaluateRequest,
    EvaluateResponse,
    MetricScore,
    MetricSummary,
    TestCaseInput,
    TestResultOutput,
)
from evaris_server.config import Settings, get_settings
from evaris_server.db import Database


class RunnerService:
    """Service for running assessments using the evaris metrics engine.

    This service:
    1. Receives assessment requests from evaris-web (via internal API)
    2. Runs metrics using the evaris package
    3. Stores results in PostgreSQL with RLS context
    4. Returns structured results with summaries

    All database operations are scoped to the organization via RLS.
    """

    def __init__(self, db: Database, settings: Settings | None = None):
        """Initialize runner service.

        Args:
            db: Database client for storing results
            settings: Server settings
        """
        self.db = db
        self.settings = settings or get_settings()

    def _get_metric_instance(self, metric_name: str) -> Any:
        """Get a metric instance by name.

        Uses the evaris metrics registry to resolve metric names.

        Args:
            metric_name: Name of the metric (e.g., 'faithfulness', 'toxicity')

        Returns:
            Instantiated metric object

        Raises:
            ValueError: If metric is unknown
        """
        # Import here to avoid circular imports and lazy loading
        from evaris.core.registry import get_metric_registry

        registry = get_metric_registry()

        # Try to get from registry first
        try:
            metric_class = registry.get(metric_name)
            return metric_class()
        except KeyError:
            pass

        # Fallback to direct imports for common metrics
        metric_map = {
            "faithfulness": "evaris.metrics.FaithfulnessMetric",
            "exact_match": "evaris.metrics.ExactMatchMetric",
            "semantic_similarity": "evaris.metrics.SemanticSimilarityMetric",
            "toxicity": "evaris.metrics.safety.ToxicityMetric",
            "bias": "evaris.metrics.safety.BiasMetric",
            "hallucination": "evaris.metrics.quality.HallucinationMetric",
            "answer_relevancy": "evaris.metrics.rag.AnswerRelevancyMetric",
            "context_precision": "evaris.metrics.rag.ContextPrecisionMetric",
            "context_recall": "evaris.metrics.rag.ContextRecallMetric",
        }

        if metric_name in metric_map:
            module_path, class_name = metric_map[metric_name].rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            metric_class = getattr(module, class_name)
            return metric_class()

        raise ValueError(f"Unknown metric: {metric_name}")

    async def _run_metric(
        self,
        metric: Any,
        test_case: TestCaseInput,
    ) -> MetricScore:
        """Run a single metric on a test case.

        Args:
            metric: Metric instance to run
            test_case: Test case data

        Returns:
            MetricScore with results
        """
        from evaris.types import TestCase as EvarisTestCase

        # Convert to evaris TestCase
        evaris_tc = EvarisTestCase(
            input=test_case.input,
            expected=test_case.expected,
            actual_output=test_case.actual_output,
            metadata=test_case.metadata,
        )

        try:
            # Run the metric
            result = await metric.a_measure(evaris_tc, test_case.actual_output)

            return MetricScore(
                name=result.name,
                score=result.score,
                passed=result.passed,
                threshold=getattr(result, "threshold", 0.5),
                reasoning=result.metadata.get("reasoning"),
                reasoning_steps=result.metadata.get("reasoning_steps"),
                reasoning_type=result.metadata.get("reasoning_type"),
                metadata=result.metadata,
            )
        except Exception as e:
            # Return failed score on error
            return MetricScore(
                name=getattr(metric, "name", metric.__class__.__name__),
                score=0.0,
                passed=False,
                threshold=0.5,
                reasoning=f"Error: {str(e)}",
                metadata={"error": str(e)},
            )

    async def run_assessment(
        self,
        request: EvaluateRequest,
        organization_id: str,
        project_id: str,
        user_id: str,
    ) -> EvaluateResponse:
        """Run assessment and store results.

        This method:
        1. Validates and instantiates requested metrics
        2. Runs each metric against each test case
        3. Computes summary statistics per metric and overall
        4. Stores the evaluation and results in the database
        5. Returns the complete response

        Args:
            request: Assessment request with test cases and metrics
            organization_id: Organization ID for RLS context
            project_id: Project ID for scoping
            user_id: User ID for audit trail

        Returns:
            EvaluateResponse with results and summary

        Raises:
            ValueError: If no valid metrics specified
        """
        assessment_id = f"assessment_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Initialize metrics
        metrics = []
        for metric_name in request.metrics:
            try:
                metric = self._get_metric_instance(metric_name)
                metrics.append((metric_name, metric))
            except ValueError as e:
                # Skip unknown metrics with warning - logged but continue
                print(f"Warning: {e}")

        if not metrics:
            raise ValueError("No valid metrics specified")

        # Run assessments
        results: list[TestResultOutput] = []
        metric_stats: dict[str, dict[str, Any]] = {
            name: {"scores": [], "passed": 0, "count": 0} for name, _ in metrics
        }

        for test_case in request.test_cases:
            scores: list[MetricScore] = []

            for metric_name, metric in metrics:
                score = await self._run_metric(metric, test_case)
                scores.append(score)

                # Update stats
                metric_stats[metric_name]["scores"].append(score.score)
                metric_stats[metric_name]["count"] += 1
                if score.passed:
                    metric_stats[metric_name]["passed"] += 1

            # Determine if test case passed (all metrics passed)
            all_passed = all(s.passed for s in scores)

            results.append(
                TestResultOutput(
                    input=test_case.input,
                    expected=test_case.expected,
                    actual_output=test_case.actual_output,
                    scores=scores,
                    passed=all_passed,
                    metadata=test_case.metadata,
                )
            )

        # Calculate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Build per-metric summaries with full statistics
        metrics_summary: dict[str, MetricSummary] = {}
        for name, stats in metric_stats.items():
            if stats["count"] > 0:
                scores_list = stats["scores"]
                avg_score = sum(scores_list) / len(scores_list)
                accuracy = stats["passed"] / stats["count"]

                # Calculate std dev if we have multiple scores
                std_dev = None
                if len(scores_list) > 1:
                    mean = avg_score
                    variance = sum((x - mean) ** 2 for x in scores_list) / len(
                        scores_list
                    )
                    std_dev = variance**0.5

                metrics_summary[name] = MetricSummary(
                    accuracy=accuracy,
                    avg_score=avg_score,
                    min_score=min(scores_list) if scores_list else None,
                    max_score=max(scores_list) if scores_list else None,
                    std_dev=std_dev,
                )

        summary = EvalSummary(
            total=total,
            passed=passed,
            failed=failed,
            accuracy=passed / total if total > 0 else 0.0,
            metrics=metrics_summary,
        )

        completed_at = datetime.now(timezone.utc)

        # Determine final status
        status = EvalStatus.PASSED if failed == 0 else EvalStatus.FAILED

        # Store in database with RLS context
        await self._store_assessment(
            assessment_id=assessment_id,
            organization_id=organization_id,
            project_id=project_id,
            dataset_id=request.dataset_id,
            name=request.name,
            status=status,
            summary=summary,
            results=results,
            metadata=request.metadata,
            created_at=now,
            completed_at=completed_at,
        )

        return EvaluateResponse(
            eval_id=assessment_id,  # Use assessment_id as the eval_id
            organization_id=organization_id,
            project_id=project_id,
            name=request.name,
            status=status,
            summary=summary,
            results=results,
            created_at=now,
            completed_at=completed_at,
            metadata=request.metadata,
        )

    async def _store_assessment(
        self,
        assessment_id: str,
        organization_id: str,
        project_id: str,
        dataset_id: str | None,
        name: str,
        status: EvalStatus,
        summary: EvalSummary,
        results: list[TestResultOutput],
        metadata: dict[str, Any],
        created_at: datetime,
        completed_at: datetime | None,
    ) -> None:
        """Store assessment and results in the database using Prisma.

        Uses the organization context for RLS enforcement.

        Args:
            assessment_id: Assessment ID
            organization_id: Organization ID (tenant)
            project_id: Project ID
            dataset_id: Optional dataset ID
            name: Assessment name
            status: Final status
            summary: Summary statistics
            results: Individual test results
            metadata: Additional metadata
            created_at: Creation timestamp
            completed_at: Completion timestamp
        """
        # Serialize summary for storage
        # Convert MetricSummary objects to dicts for JSON storage
        summary_dict = {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "accuracy": summary.accuracy,
            "avg_latency_ms": summary.avg_latency_ms,
            "metrics": {
                name: {
                    "accuracy": ms.accuracy,
                    "avg_score": ms.avg_score,
                    "min_score": ms.min_score,
                    "max_score": ms.max_score,
                    "std_dev": ms.std_dev,
                }
                for name, ms in summary.metrics.items()
            },
        }

        async with self.db.with_org_context(organization_id) as client:
            # Create the record in the Eval table
            # Note: Prisma Python uses lowercase model names
            await client.eval.create(
                data={
                    "id": assessment_id,
                    "name": name,
                    "status": status.value,
                    "organizationId": organization_id,
                    "projectId": project_id,
                    "datasetId": dataset_id,
                    "total": summary.total,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "accuracy": summary.accuracy,
                    "summary": json.dumps(summary_dict),
                    "metadata": json.dumps(metadata) if metadata else "{}",
                    "completedAt": completed_at,
                }
            )

            # Create test result records
            for i, result in enumerate(results):
                result_id = f"{assessment_id}_result_{i}"

                # Serialize scores for storage
                scores_data = [
                    {
                        "name": s.name,
                        "score": s.score,
                        "passed": s.passed,
                        "threshold": s.threshold,
                        "reasoning": s.reasoning,
                        "reasoning_steps": s.reasoning_steps,
                        "reasoning_type": s.reasoning_type,
                        "metadata": s.metadata,
                    }
                    for s in result.scores
                ]

                await client.testresult.create(
                    data={
                        "id": result_id,
                        "evalId": assessment_id,  # Foreign key to Eval table
                        "input": json.dumps(result.input),
                        "expected": json.dumps(result.expected) if result.expected else None,
                        "actualOutput": json.dumps(result.actual_output),
                        "scores": json.dumps(scores_data),
                        "passed": result.passed,
                        "latencyMs": result.latency_ms,
                        "error": result.error,
                        "metadata": json.dumps(result.metadata) if result.metadata else "{}",
                    }
                )
