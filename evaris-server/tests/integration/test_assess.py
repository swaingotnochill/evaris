"""Integration tests for the /internal/evaluate endpoint."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

# Import DirectTestClient from conftest (pytest will handle this)


class TestAssessEndpointAuth:
    """Tests for authentication on /internal/evaluate."""

    def test_missing_token_returns_401(self, client: TestClient) -> None:
        """Request without X-Context-Token should return 401."""
        response = client.post(
            "/internal/evaluate",
            json={
                "name": "test-run",
                "test_cases": [],
                "metrics": ["exact_match"],
            },
        )

        assert response.status_code == 401
        assert "Missing" in response.json()["detail"]

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        """Request with invalid JWT should return 401."""
        response = client.post(
            "/internal/evaluate",
            headers={"X-Context-Token": "invalid-token"},
            json={
                "name": "test-run",
                "test_cases": [],
                "metrics": ["exact_match"],
            },
        )

        assert response.status_code == 401

    def test_expired_token_returns_401(
        self, client: TestClient, expired_token: str
    ) -> None:
        """Request with expired JWT should return 401."""
        response = client.post(
            "/internal/evaluate",
            headers={"X-Context-Token": expired_token},
            json={
                "name": "test-run",
                "test_cases": [],
                "metrics": ["exact_match"],
            },
        )

        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()


class TestAssessEndpointValidation:
    """Tests for request validation on /internal/evaluate."""

    def test_missing_name_returns_422(self, direct_client: DirectTestClient) -> None:
        """Request without name should return 422."""
        response = direct_client.client.post(
            "/internal/evaluate",
            headers=direct_client.headers,
            json={
                "test_cases": [],
                "metrics": ["exact_match"],
            },
        )

        assert response.status_code == 422

    def test_missing_metrics_returns_422(self, direct_client: DirectTestClient) -> None:
        """Request without metrics should return 422."""
        response = direct_client.client.post(
            "/internal/evaluate",
            headers=direct_client.headers,
            json={
                "name": "test-run",
                "test_cases": [],
            },
        )

        assert response.status_code == 422

    def test_empty_metrics_returns_400(
        self, direct_client: DirectTestClient, sample_test_case: dict[str, Any]
    ) -> None:
        """Request with empty metrics list should return 400."""
        response = direct_client.post_assess(
            name="test-run",
            test_cases=[sample_test_case],
            metrics=[],
        )

        # Empty metrics list should be rejected
        assert response.status_code in [400, 422]


class TestAssessEndpointSuccess:
    """Tests for successful assessment requests."""

    def test_valid_request_returns_201(
        self,
        direct_client,
        sample_test_case: dict[str, Any],
        test_project_id: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valid assessment request should return 201 with results."""
        # Mock the runner service to avoid actual metric execution
        from evaris_server.services import RunnerService
        from evaris_server.api.schemas import (
            EvaluateResponse,
            EvalSummary,
            EvalStatus,
            MetricScore,
            TestResultOutput,
        )
        from datetime import datetime, timezone

        async def mock_run_assessment(self, request, project_id):
            return EvaluateResponse(
                eval_id="run_test123",
                project_id=project_id,
                name=request.name,
                status=EvalStatus.COMPLETED,
                summary=EvalSummary(
                    total=1,
                    passed=1,
                    failed=0,
                    accuracy=1.0,
                    metrics={"exact_match": {"mean": 1.0, "passed_count": 1}},
                ),
                results=[
                    TestResultOutput(
                        input=sample_test_case["input"],
                        expected=sample_test_case["expected"],
                        actual_output=sample_test_case["actual_output"],
                        scores=[
                            MetricScore(
                                name="exact_match",
                                score=1.0,
                                passed=True,
                                reasoning="Exact match found",
                                metadata={},
                            )
                        ],
                        passed=True,
                        metadata={},
                    )
                ],
                created_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                metadata={},
            )

        monkeypatch.setattr(RunnerService, "run_assessment", mock_run_assessment)

        response = direct_client.post_assess(
            name="test-assessment",
            test_cases=[sample_test_case],
            metrics=["exact_match"],
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-assessment"
        assert data["status"] == "completed"
        assert data["project_id"] == test_project_id
        assert "eval_id" in data
        assert data["summary"]["total"] == 1
        assert data["summary"]["passed"] == 1

    def test_multiple_test_cases(
        self,
        direct_client,
        sample_test_cases: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assessment with multiple test cases should process all."""
        from evaris_server.services import RunnerService
        from evaris_server.api.schemas import (
            EvaluateResponse,
            EvalSummary,
            EvalStatus,
            MetricScore,
            TestResultOutput,
        )
        from datetime import datetime, timezone

        async def mock_run_assessment(self, request, project_id):
            return EvaluateResponse(
                eval_id="run_multi123",
                project_id=project_id,
                name=request.name,
                status=EvalStatus.COMPLETED,
                summary=EvalSummary(
                    total=len(request.test_cases),
                    passed=len(request.test_cases),
                    failed=0,
                    accuracy=1.0,
                    metrics={},
                ),
                results=[
                    TestResultOutput(
                        input=tc.input,
                        expected=tc.expected,
                        actual_output=tc.actual_output,
                        scores=[
                            MetricScore(
                                name="exact_match",
                                score=1.0,
                                passed=True,
                                metadata={},
                            )
                        ],
                        passed=True,
                        metadata={},
                    )
                    for tc in request.test_cases
                ],
                created_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                metadata={},
            )

        monkeypatch.setattr(RunnerService, "run_assessment", mock_run_assessment)

        response = direct_client.post_assess(
            name="multi-test-assessment",
            test_cases=sample_test_cases,
            metrics=["exact_match"],
        )

        assert response.status_code == 201
        data = response.json()
        assert data["summary"]["total"] == 2
        assert len(data["results"]) == 2

    def test_metadata_is_preserved(
        self,
        direct_client,
        sample_test_case: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Metadata passed in request should be preserved in response."""
        from evaris_server.services import RunnerService
        from evaris_server.api.schemas import EvaluateResponse, EvalSummary, EvalStatus
        from datetime import datetime, timezone

        custom_metadata = {"version": "1.0", "experiment": "baseline"}

        async def mock_run_assessment(self, request, project_id):
            return EvaluateResponse(
                eval_id="run_meta123",
                project_id=project_id,
                name=request.name,
                status=EvalStatus.COMPLETED,
                summary=EvalSummary(
                    total=1,
                    passed=1,
                    failed=0,
                    accuracy=1.0,
                    metrics={},
                ),
                results=[],
                created_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                metadata=request.metadata,
            )

        monkeypatch.setattr(RunnerService, "run_assessment", mock_run_assessment)

        response = direct_client.post_assess(
            name="metadata-test",
            test_cases=[sample_test_case],
            metrics=["exact_match"],
            metadata=custom_metadata,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["metadata"] == custom_metadata


class TestAssessEndpointProjectIsolation:
    """Tests for RLS project isolation."""

    def test_different_projects_isolated(
        self,
        client: TestClient,
        valid_token: str,
        other_project_token: str,
        test_project_id: str,
        other_project_id: str,
        sample_test_case: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Requests with different project tokens should be isolated."""
        from evaris_server.services import RunnerService
        from evaris_server.api.schemas import EvaluateResponse, EvalSummary, EvalStatus
        from datetime import datetime, timezone

        captured_project_ids: list[str] = []

        async def mock_run_assessment(self, request, project_id):
            captured_project_ids.append(project_id)
            return EvaluateResponse(
                eval_id=f"run_{project_id}",
                project_id=project_id,
                name=request.name,
                status=EvalStatus.COMPLETED,
                summary=EvalSummary(
                    total=1,
                    passed=1,
                    failed=0,
                    accuracy=1.0,
                    metrics={},
                ),
                results=[],
                created_at=datetime.now(timezone.utc),
                metadata={},
            )

        monkeypatch.setattr(RunnerService, "run_assessment", mock_run_assessment)

        # Request with first project token
        response1 = client.post(
            "/internal/evaluate",
            headers={"X-Context-Token": valid_token},
            json={
                "name": "project1-test",
                "test_cases": [sample_test_case],
                "metrics": ["exact_match"],
            },
        )

        # Request with second project token
        response2 = client.post(
            "/internal/evaluate",
            headers={"X-Context-Token": other_project_token},
            json={
                "name": "project2-test",
                "test_cases": [sample_test_case],
                "metrics": ["exact_match"],
            },
        )

        assert response1.status_code == 201
        assert response2.status_code == 201

        # Verify different project IDs were used
        assert len(captured_project_ids) == 2
        assert test_project_id in captured_project_ids
        assert other_project_id in captured_project_ids

        # Verify response project IDs match
        assert response1.json()["project_id"] == test_project_id
        assert response2.json()["project_id"] == other_project_id
